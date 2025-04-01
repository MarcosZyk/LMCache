import abc
from typing import List, Optional, Tuple

import torch

from lmcache.experimental.memory_management import MemoryFormat, MemoryObj
from lmcache.utils import _lmcache_nvtx_annotate


class GPUConnectorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from 
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.
        """
        raise NotImplementedError


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self,
                 hidden_dim_size: int,
                 num_layers: int,
                 use_gpu: bool = False,
                 **kwargs):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this 
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError(
                "The memory object should be in KV_BLOB format in"
                " order to be processed by VLLMPagedMemGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        # NOTE(ApostaC): By default, detour from a GPU buffer is slower
        # than directly copying from the CPU.
        # So disabling it for now and use direct copy from CPU to GPU.

        self._multi_layer_kv_transfer(
            memory_obj.tensor,
            kvcaches,
            slot_mapping[start:end],
            False
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_BLOB.

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._multi_layer_kv_transfer(
            memory_obj.tensor,
            kvcaches,
            slot_mapping[start:end],
            True
        )

        memory_obj.metadata.fmt = MemoryFormat.KV_BLOB

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])

    def _multi_layer_kv_transfer(
            self,
            memory_obj_tensor: torch.Tensor,
            kv_caches: List[torch.Tensor],
            slot_mapping: torch.Tensor,
            direction: bool
    ):
        # memory_obj_tensor must be pinned cpu [2, num_layer, num_tokens, num_heads * head_size]
        # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
        #
        # slot mapping: (num_tokens,). The indices of the token slots that input tokens will be
        # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
        # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
        # in block 0, and 1st slot in block 1, respectively.
        num_layers = memory_obj_tensor.size(1)
        num_tokens = slot_mapping.size(0)

        for layer_idx in range(num_layers):
            page_buffer = kv_caches[layer_idx].view(2, -1, memory_obj_tensor.shape[3])

            for token_idx in range(num_tokens):
                slot = slot_mapping[token_idx].item()

                if direction:
                    # Load from page buffer to memory_obj_tensor
                    memory_obj_tensor[0, layer_idx, token_idx].copy_(page_buffer[0, slot])
                    memory_obj_tensor[1, layer_idx, token_idx].copy_(page_buffer[1, slot])
                else:
                    # Store from memory_obj_tensor to page buffer
                    page_buffer[0, slot].copy_(memory_obj_tensor[0, layer_idx, token_idx])
                    page_buffer[1, slot].copy_(memory_obj_tensor[1, layer_idx, token_idx])

IMAGE=vllm-cpu-lmcache:v1
docker run \
    --env "HUGGING_FACE_HUB_TOKEN=<>" \
    --env "LMCACHE_CONFIG_FILE=/workspace/lmcache-config/example.yaml" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "VLLM_MLA_DISABLE=1" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./example.yaml:/workspace/lmcache-config/example.yaml \
    --ipc=host \
    $IMAGE \
    --model Qwen/Qwen2.5-0.5B-Instruct \
	  --seed 0 \
	  --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}' \
    --enable-chunked-prefill false



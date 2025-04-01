from setuptools import find_packages, setup

setup(
    name="lmcache",
    version="0.1.4",
    description="LMCache: prefill your long contexts only once",
    author="LMCache team",
    author_email="lmcacheteam@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("csrc")),
    install_requires=[
        "torch == 2.6.0+cpu", "numpy==1.26.4", "aiofiles", "pyyaml", "redis",
        "nvtx", "safetensors", "transformers",
        "sortedcontainers", "prometheus_client", "infinistore", "msgspec"
    ],
    classifiers=[
        # Trove classifiers
        # Full list at https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # Add command-line scripts here
            # e.g., "my_command=my_package.module:function"
            "lmcache_server=lmcache.server.__main__:main",
            "lmcache_experimental_server=lmcache.experimental.server.__main__:main",
        ],
    },
)

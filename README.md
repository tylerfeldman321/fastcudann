# fastcudann

## What
Repo exploring neural network training optimizations with python and C++/CUDA based implementation

## Motivation

## Prerequisites
- Cuda 12.8+
- gcc 11.4+
- Python 3.12

## Installing Additional Dependencies
```bash
sudo apt update

# For torch.compile() to work
sudo apt install python3-dev

# Installing python dependencies into virtual environment
make setup_python_deps
```

## Building and Running

### C++/CUDA
```bash
make clean && make
```

### Python Code
```bash
make profile_python
```

## Resources
- https://developer.nvidia.com/blog/cuda-graphs/
- https://docs.nvidia.com/cuda/index.html
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH

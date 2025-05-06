# fastcudann

## What
Repo exploring neural network training optimizations with python and C++/CUDA based implementation. Training is done on the MNIST dataset with a fully connected classification neural network.

## Motivation
Neural network training can take a very long time, particularly for large scale foundation models like large language models or vision foundation models. Any improvements in runtime can yield significant time savings over many epochs of training. 

## Prerequisites
- Cuda 12.8+
- gcc 11.4+
- Python 3.12+

## Installing Additional Dependencies
```bash
sudo apt update

# Install cudnn
sudo apt-get install zlib1g
sudo apt-get -y install cudnn9-cuda-12

# For torch.compile()
sudo apt install python3-dev

# Creates virtual environment and installs python dependencies
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

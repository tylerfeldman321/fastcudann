# fastcudann

## What
Repo exploring neural network training optimizations with python and C++/CUDA based implementation. Training is done on the MNIST dataset with a fully connected classification neural network.

## Motivation
Neural network training can take a very long time, particularly for large scale foundation models like large language models or vision foundation models. Any improvements in runtime can yield significant time savings over many epochs of training. 

## Prerequisites
- Cuda 12.8+
- gcc 11.4+
- Python 3.12+
- Ubuntu 22.04+

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
This command will compile the program and run it. It will print out profiling information as it runs.
```bash
make clean && make train_mnist.exe
```

### Python Code
The command below will run each of the python implementations (numpy CPU, PyTorch GPU, PyTorch GPU optimized) one after the other. Note that this will take a while (around 5 minutes or so).
```bash
make profile_python
```

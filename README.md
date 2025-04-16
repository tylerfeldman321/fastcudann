# fastcudann

## What
This is a repo for fast neural network training using self updating CUDA graphs. This means we can launch a graph to do N epochs of training and only synchronize the GPU once those are done.

## Motivation - Why?
Minimize neural network training time! We often don't need to check weights and losses every epoch so instead we can use CUDA graphs to launch a task for training N epochs and synchronize once those epochs are done. This will greatly minimize launch latency of kernels compared to other frameworks that synchronize with the GPU very often, giving a tradeoff of close monitoring of losses vs. training time. The improved training time should make large differences for networks that take a long time to train.

## Prerequisites
- Cuda 12.8+
- gcc 11.4+

## Building and Running
```bash
make clean && make
```

## Resources
- https://developer.nvidia.com/blog/cuda-graphs/
- https://docs.nvidia.com/cuda/index.html
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH

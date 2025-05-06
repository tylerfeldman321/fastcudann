# Fast Neural Network Training Project Report

## 1. Purpose and Goals

Training large neural networks requires significant time, with large models potentially taking months. Even small performance optimizations can lead to substantial time savings over these long training periods. The primary goal of this project was to explore software-side optimizations to improve the runtime of neural network training. Other factors impacting training time, such as network architecture innovations, hardware improvements, and scaling techniques, are important but not the focus of this project.

## 2. Optimizations Identified

Several potential optimizations were identified for exploration:

* Kernel fusion (Python or C++)
* Reducing synchronization with the CPU (e.g., for logging losses) (Python or C++)
* Reducing launch latency via CUDA Graphs (Python or C++)
* Using pinned memory for CPU memory
* Maximizing batch size
* Leveraging optimized libraries (e.g., GEMM, cuBLAS, cuDNN)
* Optimizing memory access patterns (coalescing, shared memory, data layout)
* Adjusting data types (e.g., using FP16)
* Tuning launch configurations (grid and block dimensions)

## 3. Project Approach

* **Dataset:** This project utilizes the MNIST handwritten digit dataset, a standard machine learning benchmark consisting of 60,000 training and 10,000 testing $28\times28$ grayscale images.
* **Neural Network Architecture:** A simple single-layer, fully connected network with softmax activation and cross-entropy loss is implemented.
* **Implementation & Profiling:** The project involved iterative profiling and adding optimizations. Different implementations were created and compared:
    * Baseline CUDA/C++
    * Optimized CUDA/C++
    * Python NumPy (CPU)
    * Python PyTorch (GPU-accelerated)
    * Python PyTorch (GPU-accelerated, optimized)

## 4. Data Preprocessing

An [open-source parser](https://github.com/wichtounet/mnist/tree/master) is adapted to load the MNIST dataset into memory. The image data (originally `uint8_t`) is normalized to a 0-1 range and converted to `float`. For the fully connected network, each $28\times28$ image is represented as a 784-element float vector.

## 5. Implementation Details

### 5.1. Baseline CUDA / C++ Approach

* Used vanilla gradient descent with mini-batches.
* The core loop involved: Matrix multiplication -> softmax -> loss calculation -> accuracy calculation -> softmax backpropagation -> weight gradient backpropagation -> weight update.
* Loss and accuracy were computed every epoch.

### 5.2. Optimized CUDA / C++ Approach

* Reduced synchronization and memory copies between the host (CPU) and device (GPU), particularly for loss calculations.
* Implemented matrix multiplication using cuBLAS Sgemm for both forward and backward passes.
* Computed loss and accuracy only every 10 epochs to reduce overhead.
* Fused the loss and accuracy computation into a single kernel.
* Utilized cuDNN's optimized softmax function.

## 6. Timing Results

Runtime measurements were taken using an approximate average epoch time from the latter half of training (allowing for GPU warmup) and CPU wall clock time using `chrono`. Initialization times were excluded.

| Implementation                                                           | Average Epoch Runtime | Total Wall Clock Time |
| :----------------------------------------------------------------------- | :-------------------- | :-------------------- |
| Baseline CUDA/C++                                                        | ~38 ms                | 874 ms                |
| + reducing synchronization                                               | ~27 ms                | 554 ms                |
| + cuBLAS matrix multiplication                                           | ~20.73 ms             | 439 ms                |
| + only performing loss & accuracy calculation every 10 epochs            | ~19 ms                | 398 ms                |
| + CUDNN implementation of softmax                                        | ~18 ms                | 386 ms                |
| + CUDA Graphs to minimize kernel launch latencies                        | ~17 ms                | 367 ms                |
| Python, Numpy, CPU                                                       | ~12500 ms             | 251360 ms             |
| Simple Pytorch GPU                                                       | ~10500 ms             | 211260 ms             |
| + torch.compile(), pinned memory, AMP, preloading dataset                | ~600 ms               | 14610 ms              |

## 7. Conclusions

* Optimizing neural network training is crucial due to the potentially long training times.
* PyTorch provides easy-to-implement optimizations like `torch.compile` and Automatic Mixed Precision (AMP) that offer significant performance gains.
* There remains a considerable performance difference between optimized PyTorch code and optimized C++/CUDA code.
* Leveraging CUDA libraries like cuBLAS and cuDNN provides substantial speedups for relevant operations.
* Learned a lot about neural network training optimizations! 

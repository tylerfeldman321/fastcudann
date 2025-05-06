# Fast Neural Network Training Report

## 1. Purpose and Goals

Training large neural networks is a time-consuming process. For instance, estimates suggest it took approximately 70 days to train the Llama 3.1 405B parameter model. Furthermore, the training compute required for frontier AI models is growing by 4-5 times annually.

Given these long training times, performance optimizations become crucial. While various methods exist to reduce training runtime (including network architecture innovations, predictable scaling, and hardware improvements), this project focuses on software-side optimizations.

**Goal:** Explore various software optimizations to improve neural network training runtimes.

## 2. Optimizations Identified

Identified several potential optimization methods:

* **Kernel Fusion:** Combining multiple CUDA kernels into one to reduce launch overhead.
* **Reducing CPU Synchronization:** Minimizing points where the GPU must wait for the CPU.
* **CUDA Graphs:** Reducing kernel launch latency.
* **Pinned Memory:** Using pinned host memory for faster CPU-GPU data transfers.
* **Maximizing Batch Size:** Processing more data in parallel per iteration.
* **Optimized Libraries:** Leveraging libraries like cuBLAS (for GEMM) and cuDNN (for common NN operations like softmax).
* **Memory Access Patterns:** Efficiently using shared memory and optimizing data layout.
* **Data Types:** Using lower-precision types like FP16 for faster computation.
* **Launch Configuration Tuning:** Optimizing grid and block dimensions for kernel launches.

## 3. Project Approach: Iterative Improvement

The project employs an iterative approach to implement and evaluate optimizations.

* **Dataset:** MNIST handwritten digit dataset (60k training, 10k testing samples, $28 \times 28$ grayscale images).
* **Network Architecture:** A simple single-layer fully connected network with softmax activation and cross-entropy loss.
* **Methodology:**
    1.  Establish a working baseline network.
    2.  Iteratively profile the code and add optimization features.
    3.  Compare runtimes across different implementations.

* **Implementations Compared:**
    * Baseline CUDA/C++
    * Optimized CUDA/C++
    * Python + NumPy (CPU)
    * Python + PyTorch (GPU-accelerated)
    * Python + PyTorch (GPU-accelerated, optimized)

## 4. Data Preprocessing and Representation

* An open-source parser loads the MNIST dataset as `uint8_t` into contiguous memory.
* Data is normalized to a 0-1 range and converted to `float`.
* Each $28 \times 28$ image is flattened into a 784-element float vector for the fully connected network.

## 5. Implementation Details

### 5.1. Baseline CUDA / C++

* Uses vanilla mini-batch stochastic gradient descent (SGD).
* Training loop involves: Matrix Multiplication -> Softmax -> Loss Calculation -> Accuracy Calculation -> Softmax Backpropagation -> Weight Gradient Backpropagation -> Weight Update.
* Loss and accuracy are computed and reported every epoch.

### 5.2. Optimized CUDA / C++

* **Reduced Synchronization/memcpy:** Minimizes data transfers and waits between host and device, especially for loss calculations.
* **cuBLAS:** Implements forward and backward pass matrix multiplications using `Sgemm`.
* **Reduced Reporting:** Computes loss and accuracy only every 10 epochs.
* **Kernel Fusion:** Fuses the loss and accuracy computation into a single kernel.
* **cuDNN:** Utilizes cuDNN's optimized softmax function.
* **CUDA Graphs:** Employs CUDA Graphs to minimize kernel launch latency.

### 5.3. Python Implementations

* **NumPy CPU:** Basic gradient descent using vectorized NumPy operations on the CPU.
* **PyTorch GPU:** Uses a standard PyTorch `Logistic Regression` model, `CrossEntropyLoss`, SGD optimizer, and a `DataLoader` with a batch size of 128. Accesses loss every epoch.
* **PyTorch GPU Optimized:** Applies several PyTorch optimizations:
    * `torch.compile()`: JIT compilation for performance.
    * Automatic Mixed Precision (AMP): Uses FP16/FP32 for faster computation with minimal precision loss.
    * Data Preloading: Loads the entire dataset onto the GPU upfront.
    * Pinned Memory: Uses pinned host memory for faster data transfers if data loading from CPU is needed within the loop (though preloading mitigates this).

## 6. Timing Results Notes

* Timings represent an approximate average epoch runtime measured during the latter half of training (allowing for GPU warmup).
* CPU wall clock time is measured using `chrono`.
* Initialization times (e.g., data transfer to GPU, network building) are excluded.

## 7. Results Summary

| Implementation Description                                                               | Average Epoch Runtime | Total Wall Clock Time |
| :--------------------------------------------------------------------------------------- | :-------------------- | :-------------------- |
| Baseline CUDA/C++                                                                        | ~38 ms                | 874 ms                |
| + Reducing synchronization and memcpy to host                                            | ~27 ms                | 554 ms                |
| + cuBLAS matrix multiplication                                                           | ~20.73 ms             | 439 ms                |
| + Only performing loss & accuracy calculation every 10 epochs                            | ~19 ms                | 398 ms                |
| + cuDNN implementation of softmax                                                        | ~18 ms                | 386 ms                |
| + CUDA Graphs to minimize kernel launch latencies                                        | **~17 ms** | **367 ms** |
| Python + Numpy (CPU)                                                                     | ~12500 ms             | 251360 ms             |
| Python + Pytorch (GPU)                                                                   | ~10500 ms             | 211260 ms             |
| + torch.compile(), pinned memory, AMP, preloading dataset                                | **~600 ms** | **14610 ms** |

*(Note: Best times for C++/CUDA and Python/PyTorch highlighted in bold)*

## 8. CUDA Concepts and Libraries Used

* **CUDA Libraries:** cuRAND, cuDNN, cuBLAS
* **CUDA Events:** For timing kernel execution.
* **Grid/Block Dimensions:** Configured for 2D operations (e.g., matrix multiplication).
* **CUDA Streams:** For managing asynchronous operations.
* **CUDA Graphs:** To capture and replay sequences of CUDA operations, reducing launch overhead.

## 9. Conclusions

* Optimizing neural network training is vital due to the significant time investment required.
* High-level libraries like PyTorch provide easy-to-use, powerful optimization features (e.g., `torch.compile`, AMP) that offer substantial speedups over basic GPU implementations.
* Specialized CUDA libraries like cuBLAS and cuDNN are highly effective for accelerating core deep learning operations.
* There is a significant performance difference between the custom-optimized C++/CUDA implementation and the optimized PyTorch version, highlighting the effectiveness of both low-level control and high-level library optimizations.
* The project provided valuable insights into various optimization techniques applicable to both neural network training and inference.

## 10. Resources

* **Dataset:** MNIST Dataset, wichtounet/mnist C++ reader
* **CUDA Libraries:** cuBLAS Docs, cuDNN Docs (Softmax), cuRAND
* **CUDA Graphs:** CUDA Runtime API Docs, NVIDIA Blog Post, CUDA C++ Programming Guide
* **PyTorch:** torch.compile Docs, Automatic Mixed Precision (AMP) Docs

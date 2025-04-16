#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono> // For timing measurements

#define INPUT_SIZE 5     // Number of input features
#define OUTPUT_SIZE 1    // Binary classification (1 output)
#define BATCH_SIZE 3     // Number of samples in one batch
#define LEARNING_RATE 0.1
#define EPOCHS 100       // Number of epochs
#define WARMUP_EPOCHS 5  // Number of warmup epochs (not counted in timing)

// CUDA kernel to apply sigmoid activation function
__global__ void sigmoid_kernel(float *output, float *input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for the forward pass (matrix multiplication input * weights)
__global__ void matmul_kernel(float *output, float *input, float *weights, int input_size, int output_size, int batch_size) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < batch_size && col < output_size) {
        float value = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            value += input[row * input_size + k] * weights[k * output_size + col];
        }
        output[row * output_size + col] = value;
    }
}

// CUDA kernel to calculate error (output - target)
__global__ void calc_error_kernel(float *error, float *output, float *target, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        error[idx] = output[idx] - target[idx];
    }
}

// CUDA kernel to calculate the gradient for weights (during backpropagation)
__global__ void calc_gradient_kernel(float *gradient, float *input, float *error, int input_size, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < input_size) {
        float grad = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad += input[i * input_size + idx] * error[i];
        }
        gradient[idx] = grad / batch_size;
    }
}

// CUDA kernel for weight update
__global__ void update_weights_kernel(float *weights, float *gradient, float learning_rate, int weight_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < weight_size) {
        weights[idx] -= learning_rate * gradient[idx];
    }
}

// Wrapper function to handle CUDA error checking
void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Fake example data
    float h_input[BATCH_SIZE][INPUT_SIZE] = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
        {0.5f, 0.4f, 0.3f, 0.2f, 0.1f},
        {0.6f, 0.7f, 0.8f, 0.9f, 1.0f}
    };
    float h_weights[INPUT_SIZE][OUTPUT_SIZE] = {
        {0.1f},
        {0.2f},
        {0.3f},
        {0.4f},
        {0.5f}
    };
    float h_target[BATCH_SIZE][OUTPUT_SIZE] = {
        {1.0f},
        {0.0f},
        {1.0f}
    };
    float h_output[BATCH_SIZE][OUTPUT_SIZE] = {0};

    // Device memory allocation
    float *d_input, *d_weights, *d_output, *d_error, *d_gradient, *d_target;
    
    checkCudaError(cudaMalloc((void**)&d_input, sizeof(h_input)), "Allocating d_input");
    checkCudaError(cudaMalloc((void**)&d_weights, sizeof(h_weights)), "Allocating d_weights");
    checkCudaError(cudaMalloc((void**)&d_output, sizeof(h_output)), "Allocating d_output");
    checkCudaError(cudaMalloc((void**)&d_error, sizeof(h_output)), "Allocating d_error");
    checkCudaError(cudaMalloc((void**)&d_gradient, sizeof(float) * INPUT_SIZE), "Allocating d_gradient");
    checkCudaError(cudaMalloc((void**)&d_target, sizeof(h_target)), "Allocating d_target");

    checkCudaError(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice), "Copying input to device");
    checkCudaError(cudaMemcpy(d_weights, h_weights, sizeof(h_weights), cudaMemcpyHostToDevice), "Copying weights to device");
    checkCudaError(cudaMemcpy(d_target, h_target, sizeof(h_target), cudaMemcpyHostToDevice), "Copying target to device");

    // Kernel configurations
    dim3 blockSize(32);
    dim3 gridSize_batch((BATCH_SIZE + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_input((INPUT_SIZE + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_weights((INPUT_SIZE * OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_batch_output((BATCH_SIZE * OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);

    // Create CUDA streams
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Creating CUDA stream");

    // Initialize CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Capture graph
    checkCudaError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "Beginning stream capture");
    
    // Forward pass kernels
    matmul_kernel<<<BATCH_SIZE, OUTPUT_SIZE, 0, stream>>>(d_output, d_input, d_weights, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    sigmoid_kernel<<<gridSize_batch_output, blockSize, 0, stream>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);
    
    // Error calculation
    calc_error_kernel<<<gridSize_batch_output, blockSize, 0, stream>>>(d_error, d_output, d_target, BATCH_SIZE * OUTPUT_SIZE);
    
    // Backpropagation kernels
    calc_gradient_kernel<<<gridSize_input, blockSize, 0, stream>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);
    update_weights_kernel<<<gridSize_weights, blockSize, 0, stream>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);
    
    // End capture
    checkCudaError(cudaStreamEndCapture(stream, &graph), "Ending stream capture");
    
    // Create executable graph
    checkCudaError(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "Instantiating graph");
    
    // create a baseline version without CUDA graphs for comparison
    cudaStream_t baseline_stream;
    checkCudaError(cudaStreamCreate(&baseline_stream), "Creating baseline CUDA stream");
    
    std::cout << "Starting training using CUDA graph..." << std::endl;
    
    // Warmup runs for GPU
    std::cout << "Performing " << WARMUP_EPOCHS << " warmup epochs..." << std::endl;
    for (int warmup = 0; warmup < WARMUP_EPOCHS; ++warmup) {
        checkCudaError(cudaGraphLaunch(graphExec, stream), "Launching graph for warmup");
        checkCudaError(cudaStreamSynchronize(stream), "Synchronizing stream for warmup");
    }
    
    // timing variables
    auto total_start = std::chrono::high_resolution_clock::now();
    auto total_end = total_start;
    double total_time_ms = 0.0;
    
    double avg_epoch_time_ms = 0.0;
    double min_epoch_time_ms = std::numeric_limits<double>::max();
    double max_epoch_time_ms = 0.0;
    
    cudaDeviceSynchronize();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << std::endl;
        }
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Launch graph (this replaces all the kernel launches with a single launch)
        checkCudaError(cudaGraphLaunch(graphExec, stream), "Launching graph");
        checkCudaError(cudaStreamSynchronize(stream), "Synchronizing stream");
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        
        // timing
        avg_epoch_time_ms += epoch_time_ms;
        min_epoch_time_ms = std::min(min_epoch_time_ms, epoch_time_ms);
        max_epoch_time_ms = std::max(max_epoch_time_ms, epoch_time_ms);
        
        if (epoch % 10 == 0) {
            checkCudaError(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost), "Copying output to host");
            std::cout << "Sample outputs: " << h_output[0][0] << ", " << h_output[1][0] << ", " << h_output[2][0] << std::endl;
            std::cout << "Epoch time: " << epoch_time_ms << " ms" << std::endl;
        }
    }
    
    // Calculate final timing stats
    total_end = std::chrono::high_resolution_clock::now();
    total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    avg_epoch_time_ms /= EPOCHS;
    
    // Get final weights and output
    checkCudaError(cudaMemcpy(h_weights, d_weights, sizeof(h_weights), cudaMemcpyDeviceToHost), "Copying weights to host");
    checkCudaError(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost), "Copying output to host");
    
    // Print final results
    std::cout << "\nTraining completed." << std::endl;
    std::cout << "Final outputs: " << std::endl;
    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "Sample " << i << ": " << h_output[i][0] << " (Target: " << h_target[i][0] << ")" << std::endl;
    }
    
    std::cout << "\nFinal weights: " << std::endl;
    for (int i = 0; i < INPUT_SIZE; i++) {
        std::cout << "Weight " << i << ": " << h_weights[i][0] << std::endl;
    }
    
    // Print timing information
    std::cout << "\nTiming Information:" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "Total training time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average time per epoch: " << avg_epoch_time_ms << " ms" << std::endl;
    std::cout << "Minimum epoch time: " << min_epoch_time_ms << " ms" << std::endl;
    std::cout << "Maximum epoch time: " << max_epoch_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (EPOCHS * BATCH_SIZE * 1000.0 / total_time_ms) << " samples/second" << std::endl;
    
    // Run baseline version (without CUDA graphs) for comparison
    std::cout << "\nRunning baseline version (without CUDA graphs) for comparison..." << std::endl;
    
    // Reset weights for fair comparison
    checkCudaError(cudaMemcpy(d_weights, h_weights, sizeof(h_weights), cudaMemcpyHostToDevice), "Resetting weights for baseline");
    
    // Warmup
    for (int warmup = 0; warmup < WARMUP_EPOCHS; ++warmup) {
        matmul_kernel<<<BATCH_SIZE, OUTPUT_SIZE, 0, baseline_stream>>>(d_output, d_input, d_weights, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
        sigmoid_kernel<<<gridSize_batch_output, blockSize, 0, baseline_stream>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);
        calc_error_kernel<<<gridSize_batch_output, blockSize, 0, baseline_stream>>>(d_error, d_output, d_target, BATCH_SIZE * OUTPUT_SIZE);
        calc_gradient_kernel<<<gridSize_input, blockSize, 0, baseline_stream>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);
        update_weights_kernel<<<gridSize_weights, blockSize, 0, baseline_stream>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);
        cudaStreamSynchronize(baseline_stream);
    }
    
    // Timing variables for baseline
    auto baseline_start = std::chrono::high_resolution_clock::now();
    double baseline_time_ms = 0.0;
    double baseline_avg_epoch_ms = 0.0;
    
    cudaDeviceSynchronize();
    
    // Run baseline epochs
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        matmul_kernel<<<BATCH_SIZE, OUTPUT_SIZE, 0, baseline_stream>>>(d_output, d_input, d_weights, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
        sigmoid_kernel<<<gridSize_batch_output, blockSize, 0, baseline_stream>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);
        calc_error_kernel<<<gridSize_batch_output, blockSize, 0, baseline_stream>>>(d_error, d_output, d_target, BATCH_SIZE * OUTPUT_SIZE);
        calc_gradient_kernel<<<gridSize_input, blockSize, 0, baseline_stream>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);
        update_weights_kernel<<<gridSize_weights, blockSize, 0, baseline_stream>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);
        
        cudaStreamSynchronize(baseline_stream);
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        baseline_avg_epoch_ms += epoch_time_ms;
        
        if (epoch % 10 == 0) {
            std::cout << "Baseline epoch " << epoch + 1 << " time: " << epoch_time_ms << " ms" << std::endl;
        }
    }
    
    auto baseline_end = std::chrono::high_resolution_clock::now();
    baseline_time_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
    baseline_avg_epoch_ms /= EPOCHS;
    
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << "---------------------" << std::endl;
    std::cout << "CUDA Graph total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Baseline total time: " << baseline_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (baseline_time_ms / total_time_ms) << "x" << std::endl;
    std::cout << "CUDA Graph avg epoch: " << avg_epoch_time_ms << " ms" << std::endl;
    std::cout << "Baseline avg epoch: " << baseline_avg_epoch_ms << " ms" << std::endl;
    
    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(baseline_stream);
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_error);
    cudaFree(d_gradient);
    cudaFree(d_target);
    
    std::cout << "Resources cleaned up successfully." << std::endl;
    return 0;
}
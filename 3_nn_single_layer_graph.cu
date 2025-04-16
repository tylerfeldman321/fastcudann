#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define INPUT_SIZE 5  // Number of input features
#define OUTPUT_SIZE 1 // Binary classification (1 output)
#define BATCH_SIZE 3  // Number of samples in one batch
#define LEARNING_RATE 0.1
#define EPOCHS 100    // Number of epochs

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
        {0.6f}
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
    
    std::cout << "Starting training using CUDA graph..." << std::endl;
    
    // Loop for epochs
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        checkCudaError(cudaGraphLaunch(graphExec, stream), "Launching graph");
    }

    checkCudaError(cudaStreamSynchronize(stream), "Synchronizing stream");

    
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
    
    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_error);
    cudaFree(d_gradient);
    cudaFree(d_target);
    
    std::cout << "Resources cleaned up successfully." << std::endl;
    return 0;
}
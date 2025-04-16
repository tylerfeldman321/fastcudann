#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define INPUT_SIZE 5  // Number of input features
#define OUTPUT_SIZE 1 // Binary classification (1 output)
#define BATCH_SIZE 3  // Number of samples in one batch
#define LEARNING_RATE 0.1
#define EPOCHS 100  // Number of epochs

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
    float h_biases[OUTPUT_SIZE] = {0.0f};
    float h_output[BATCH_SIZE][OUTPUT_SIZE] = {0};
    float h_error[BATCH_SIZE][OUTPUT_SIZE] = {0};

    // Device memory allocation
    float *d_input, *d_weights, *d_output, *d_error, *d_gradient;
    cudaMalloc((void**)&d_input, sizeof(h_input));
    cudaMalloc((void**)&d_weights, sizeof(h_weights));
    cudaMalloc((void**)&d_output, sizeof(h_output));
    cudaMalloc((void**)&d_error, sizeof(h_error));
    cudaMalloc((void**)&d_gradient, sizeof(float) * INPUT_SIZE);

    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(h_weights), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << std::endl;

        // Forward pass (input * weights)
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (OUTPUT_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input, d_weights, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);

        // Apply sigmoid activation
        sigmoid_kernel<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);

        // Error Calculation (output - target)
        float target[BATCH_SIZE][OUTPUT_SIZE] = {
            {1.0f},
            {0.0f},
            {1.0f}
        };
        cudaMemcpy(d_error, target, sizeof(target), cudaMemcpyHostToDevice);

        // Backpropagation
        calc_gradient_kernel<<<(INPUT_SIZE + 255) / 256, 256>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);

        // Update weights
        update_weights_kernel<<<(INPUT_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_error);
    cudaFree(d_gradient);

    return 0;
}

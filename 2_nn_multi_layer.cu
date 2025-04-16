#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define INPUT_SIZE 5       // Number of input features
#define HIDDEN_SIZE 3      // Number of hidden layer neurons
#define OUTPUT_SIZE 1      // Binary classification (1 output)
#define BATCH_SIZE 3       // Number of samples in one batch
#define LEARNING_RATE 0.1
#define EPOCHS 100         // Number of epochs

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

    // Weights and biases for 2-layer network (hidden + output layer)
    float h_weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE] = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f},
        {1.0f, 1.1f, 1.2f},
        {1.3f, 1.4f, 1.5f}
    };

    float h_weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE] = {
        {0.1f},
        {0.2f},
        {0.3f}
    };

    float h_biases_hidden[HIDDEN_SIZE] = {0.0f, 0.0f, 0.0f};
    float h_biases_output[OUTPUT_SIZE] = {0.0f};

    float h_output[BATCH_SIZE][OUTPUT_SIZE] = {0};
    float h_error[BATCH_SIZE][OUTPUT_SIZE] = {0};  // error = output - target

    // Device memory allocation
    float *d_input, *d_weights_input_hidden, *d_weights_hidden_output;
    float *d_output, *d_hidden_output, *d_error, *d_gradient_input_hidden, *d_gradient_hidden_output;

    cudaMalloc((void**)&d_input, sizeof(h_input));
    cudaMalloc((void**)&d_weights_input_hidden, sizeof(h_weights_input_hidden));
    cudaMalloc((void**)&d_weights_hidden_output, sizeof(h_weights_hidden_output));
    cudaMalloc((void**)&d_output, sizeof(h_output));
    cudaMalloc((void**)&d_hidden_output, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE);
    cudaMalloc((void**)&d_error, sizeof(h_error));
    cudaMalloc((void**)&d_gradient_input_hidden, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc((void**)&d_gradient_hidden_output, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);

    // Copy data to device
    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_input_hidden, h_weights_input_hidden, sizeof(h_weights_input_hidden), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_hidden_output, h_weights_hidden_output, sizeof(h_weights_hidden_output), cudaMemcpyHostToDevice);

    // Loop for epochs
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << std::endl;

        // Forward pass - Input -> Hidden layer
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks_hidden((BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (HIDDEN_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks_hidden, threadsPerBlock>>>(d_hidden_output, d_input, d_weights_input_hidden,
                                                             INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);
        sigmoid_kernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden_output, d_hidden_output, BATCH_SIZE * HIDDEN_SIZE);

        // Forward pass - Hidden -> Output layer
        dim3 numBlocks_output((BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, (OUTPUT_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks_output, threadsPerBlock>>>(d_output, d_hidden_output, d_weights_hidden_output,
                                                              HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
        sigmoid_kernel<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);

        // Error Calculation (output - target)
        float target[BATCH_SIZE][OUTPUT_SIZE] = {
            {1.0f},
            {0.0f},
            {1.0f}
        };
        cudaMemcpy(d_error, target, sizeof(target), cudaMemcpyHostToDevice);

        // Backpropagation
        // Calculate gradients for the hidden-to-output layer
        calc_gradient_kernel<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_gradient_hidden_output, d_hidden_output, d_error, HIDDEN_SIZE, BATCH_SIZE);

        // Calculate gradients for the input-to-hidden layer
        calc_gradient_kernel<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_gradient_input_hidden, d_input, d_error, INPUT_SIZE, BATCH_SIZE);

        // Update weights
        update_weights_kernel<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_weights_hidden_output, d_gradient_hidden_output, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
        update_weights_kernel<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(d_weights_input_hidden, d_gradient_input_hidden, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
    }

    // After training, you can test or evaluate the model.
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    std::cout << "Final Output: \n";
    for (int i = 0; i < BATCH_SIZE; i++) {
        std::cout << h_output[i][0] << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_weights_input_hidden);
    cudaFree(d_weights_hidden_output);
    cudaFree(d_output);
    cudaFree(d_hidden_output);
    cudaFree(d_error);
    cudaFree(d_gradient_input_hidden);
    cudaFree(d_gradient_hidden_output);

    return 0;
}

#ifndef OPS_CUH
#define OPS_CUH

#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>

__global__ void convert_and_normalize(const uint8_t* input, float* output, int num_elements) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int data_idx = thread_idx; data_idx < num_elements; data_idx += stride) {
        output[data_idx] = (float)input[data_idx] / 255.0f;
    }
}

__global__ void init_weights_uniform(float* data, size_t size, size_t seed) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t data_idx = thread_idx; data_idx < size; data_idx+=stride) {
        curandState_t state;
        curand_init(data_idx+(size_t)seed, thread_idx, 0, &state);
        float rand_val = curand_uniform(&state);
        data[data_idx] = 2.0f * rand_val - 1.0f;
    }
}


__global__ void matmul_kernel(float *output, float *input, float *weights, int input_size, int output_size, int batch_size) {
    int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_y = blockDim.y * blockIdx.y + threadIdx.y;

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // Using x to index into sample
    // Using y to index into output_idx
    for (int sample_idx = thread_id_x; sample_idx < batch_size; sample_idx += stride_x) {
        for (int output_idx = thread_id_y; output_idx < output_size; output_idx += stride_y) {
            float value = 0.0f;
            for (int k = 0; k < input_size; ++k) {
                value += input[sample_idx * input_size + k] * weights[k * output_size + output_idx];
            }
            output[sample_idx * output_size + output_idx] = value;
        }
    }
}


__global__ void softmax(const float* logits, float* probabilities, int batch_size, int num_classes) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Iterating through samples in the batch
    for (size_t idx = thread_idx; idx < batch_size; idx+=stride) {
        int offset = idx * num_classes;
        const float* current_logits = logits + offset;
        float* current_probs = probabilities + offset;

        // 1. Find max logit for numerical stability
        float max_logit = -INFINITY;
        for (int j = 0; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, current_logits[j]);
        }

        // 2. Calculate sum of exponentials (shifted)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(current_logits[j] - max_logit);
        }

        // Add a small epsilon to prevent division by zero, though sum_exp should always be positive
        const float epsilon = 1e-12f;
        sum_exp += epsilon;

        // 3. Calculate probabilities
        for (int j = 0; j < num_classes; ++j) {
            current_probs[j] = expf(current_logits[j] - max_logit) / sum_exp;
        }
    }
}


__global__ void scce_loss_forward_kernel(const float* probabilities,
    const uint8_t* true_labels, float* losses,
    int batch_size, int num_classes) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Iterating through samples in the batch
    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx+=stride) {
        int prob_idx = sample_idx * num_classes + (int)true_labels[sample_idx];
        float prob_true_class = probabilities[prob_idx];

        // Add small epsilon for numerical stability (prevent log(0))
        const float epsilon = 1e-9f;
        losses[sample_idx] = -logf(prob_true_class + epsilon);
    }
}


__global__ void scce_softmax_backward_kernel(const float* probabilities,
    const uint8_t* true_labels,
    float* grad_logits,
    int batch_size, int num_classes) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float inv_batch_size = 1.0f / (float)batch_size;

    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx+=stride) {
        int offset = sample_idx * num_classes;
        const float* current_probs = probabilities + offset;
        float* current_grad = grad_logits + offset;
        int true_label = (int)true_labels[sample_idx];

        for (int j = 0; j < num_classes; ++j) {
            float indicator = (j == true_label) ? 1.0f : 0.0f;
            // Gradient = (Predicted Probability - True Label Indicator)
            current_grad[j] = (current_probs[j] - indicator) * inv_batch_size;
        }
    }
}


__global__ void calculate_weight_gradient_kernel(float* grad_weights,
    const float* input_images,
    const float* grad_logits,
    int input_size,
    int output_size,
    int batch_size) {
    // Each thread calculates one element of the grad_weights matrix
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;   // Index for input_size (row of weights)
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y; // Index for output_size (column of weights)

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int input_idx = thread_id_x; input_idx < input_size; input_idx += stride_x) {
        for (int output_idx = thread_id_y; output_idx < output_size; output_idx += stride_y) {
            float gradient_sum = 0.0f;
            for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                gradient_sum += input_images[sample_idx * input_size + input_idx] *
                grad_logits[sample_idx * output_size + output_idx];
            }

            int weight_grad_idx = input_idx * output_size + output_idx;
            grad_weights[weight_grad_idx] = gradient_sum;
        }
    }
}

/**
* @brief Updates the weights using gradient descent.
*
* Performs the update: weights = weights - learning_rate * grad_weights
*
* @param weights Weights to be updated (size: num_weights).
* @param grad_weights Calculated gradients for the weights (size: num_weights).
* @param learning_rate The step size for the gradient descent update.
* @param num_weights Total number of elements in the weights matrix (input_size * output_size).
*/
__global__ void update_weights_kernel(float* weights, const float* grad_weights, float learning_rate, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_weights; i += stride) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}

#endif
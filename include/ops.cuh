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
    // Initialize RNG state once per thread
    curandState_t state;
    // Ensure unique seed/sequence per thread even across blocks
    curand_init((size_t)seed + thread_idx, 0, 0, &state);

    for (size_t data_idx = thread_idx; data_idx < size; data_idx+=stride) {
        // Use the initialized state
        float rand_val = curand_uniform(&state);
        data[data_idx] = 2.0f * rand_val - 1.0f; // Range [-1.0, 1.0]
    }
}


__global__ void matmul_kernel(float *output, const float *input, const float *weights, int input_size, int output_size, int batch_size) {
    // Use 2D indexing for threads and blocks
    int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x; // Corresponds to sample_idx (batch dimension)
    int thread_id_y = blockDim.y * blockIdx.y + threadIdx.y; // Corresponds to output_idx (output features)

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // Using x to index into sample
    // Using y to index into output_idx
    for (int sample_idx = thread_id_x; sample_idx < batch_size; sample_idx += stride_x) {
        for (int output_idx = thread_id_y; output_idx < output_size; output_idx += stride_y) {
            float value = 0.0f;
            for (int k = 0; k < input_size; ++k) {
                // Access input: batch `sample_idx`, feature `k`
                // Access weights: input feature `k`, output feature `output_idx`
                value += input[sample_idx * input_size + k] * weights[k * output_size + output_idx];
            }
            // Write output: batch `sample_idx`, feature `output_idx`
            output[sample_idx * output_size + output_idx] = value;
        }
    }
}


__global__ void softmax(const float* logits, float* probabilities, int batch_size, int num_classes) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Iterating through samples in the batch (each thread handles one or more samples)
    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx+=stride) {
        int offset = sample_idx * num_classes;
        const float* current_logits = logits + offset;
        float* current_probs = probabilities + offset;

        // --- Find max logit for numerical stability ---
        // Note: Could be optimized with block-level reduction for large num_classes
        float max_logit = -INFINITY;
        for (int j = 0; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, current_logits[j]);
        }

        // --- Calculate sum of exponentials (shifted) ---
         // Note: Could be optimized with block-level reduction for large num_classes
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(current_logits[j] - max_logit);
        }

        // Add a small epsilon to prevent division by zero
        const float epsilon = 1e-12f;
        sum_exp = fmaxf(sum_exp, epsilon); // Ensure sum_exp is at least epsilon

        // --- Calculate probabilities ---
        float inv_sum_exp = 1.0f / sum_exp; // Compute inverse once
        for (int j = 0; j < num_classes; ++j) {
            current_probs[j] = expf(current_logits[j] - max_logit) * inv_sum_exp;
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
        int true_label_int = (int)true_labels[sample_idx];
        // Bounds check (optional but safe)
        if (true_label_int < 0 || true_label_int >= num_classes) {
             // Handle error or assign a default high loss, prevent out-of-bounds access
             losses[sample_idx] = 100.0f; // Example: Assign a high loss
             continue;
        }
        int prob_idx = sample_idx * num_classes + true_label_int;
        float prob_true_class = probabilities[prob_idx];

        // Add small epsilon for numerical stability (prevent log(0))
        const float epsilon = 1e-9f;
        losses[sample_idx] = -logf(fmaxf(prob_true_class, epsilon)); // Ensure argument to logf is positive
    }
}


__global__ void scce_softmax_backward_kernel(const float* probabilities,
    const uint8_t* true_labels,
    float* grad_logits,
    int batch_size, int num_classes, float scale_factor) { // Added scale_factor
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Iterating through samples in the batch
    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx+=stride) {
        int offset = sample_idx * num_classes;
        const float* current_probs = probabilities + offset;
        float* current_grad = grad_logits + offset;
        int true_label = (int)true_labels[sample_idx];

        for (int j = 0; j < num_classes; ++j) {
            float indicator = (j == true_label) ? 1.0f : 0.0f;
            // Gradient = (Predicted Probability - True Label Indicator) * scale_factor
            // Scale factor is typically 1.0 / batch_size for averaging
            current_grad[j] = (current_probs[j] - indicator) * scale_factor;
        }
    }
}


__global__ void calculate_weight_gradient_kernel(float* grad_weights,
    const float* input_images, // Should point to the start of the current batch's images
    const float* grad_logits,  // Should correspond to the current batch's gradients
    int input_size,
    int output_size,
    int batch_size) {          // Should be the current mini-batch size
    // Each thread calculates one element of the grad_weights matrix
    // Use 2D indexing
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x; // Index for input_size (row of weights)
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y; // Index for output_size (column of weights)

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // Iterate over the portion of the weight gradient matrix this thread is responsible for
    for (int input_idx = thread_id_x; input_idx < input_size; input_idx += stride_x) {
        for (int output_idx = thread_id_y; output_idx < output_size; output_idx += stride_y) {
            float gradient_sum = 0.0f;
            // Sum contributions from all samples in the *current mini-batch*
            for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                // grad_weights[in, out] = sum_{batch} ( input[batch, in] * grad_logits[batch, out] )
                gradient_sum += input_images[sample_idx * input_size + input_idx] *
                                grad_logits[sample_idx * output_size + output_idx];
            }

            // Write the final gradient for this weight element
            // Note: This overwrites previous gradient. For accumulation across batches (if needed), use atomicAdd or calculate per batch and sum later.
            // Standard SGD updates weights after each batch, so overwriting is correct here.
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
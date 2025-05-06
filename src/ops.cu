#include "../include/ops.cuh"

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
    curandState_t state;
    curand_init((size_t)seed + thread_idx, 0, 0, &state);

    for (size_t data_idx = thread_idx; data_idx < size; data_idx+=stride) {
        float rand_val = curand_uniform(&state);
        data[data_idx] = 2.0f * rand_val - 1.0f;
    }
}


__global__ void matmul_kernel(float *output, const float *input, const float *weights, int input_size, int output_size, int batch_size) {
    // Use 2D indexing for threads and blocks
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

    // Iterating through samples in the batch (each thread handles one or more samples)
    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx+=stride) {
        int offset = sample_idx * num_classes;
        const float* current_logits = logits + offset;
        float* current_probs = probabilities + offset;

        // --- Find max logit for numerical stability ---
        float max_logit = -INFINITY;
        for (int j = 0; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, current_logits[j]);
        }

        // --- Calculate sum of exponentials (shifted) ---
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(current_logits[j] - max_logit);
        }

        const float epsilon = 1e-12f;
        sum_exp = fmaxf(sum_exp, epsilon);

        // --- Calculate probabilities ---
        float inv_sum_exp = 1.0f / sum_exp;
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
    int batch_size, int num_classes, float scale_factor) {
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
            current_grad[j] = (current_probs[j] - indicator) * scale_factor;
        }
    }
}


__global__ void calculate_weight_gradient_kernel(float* grad_weights,
    const float* input_images,
    const float* grad_logits,
    int input_size,
    int output_size,
    int batch_size
) {
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void update_weights_kernel(float* weights, const float* grad_weights, float learning_rate, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_weights; i += stride) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}

__global__ void calculate_accuracy_kernel(
    const float* probabilities,
    const uint8_t* true_labels,
    int* correct_counts,
    int batch_size,
    int num_classes
) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t sample_idx = thread_idx; sample_idx < batch_size; sample_idx += stride) {
        int offset = sample_idx * num_classes;
        const float* current_probs = probabilities + offset;
        int true_label = (int)true_labels[sample_idx];

        // Find the index of the highest probability (predicted class)
        int predicted_label = 0;
        float max_prob = current_probs[0];
        for (int j = 1; j < num_classes; ++j) {
            if (current_probs[j] > max_prob) {
                max_prob = current_probs[j];
                predicted_label = j;
            }
        }

        // If prediction matches true label, atomically increment the counter
        if (predicted_label == true_label) {
            atomicAdd(correct_counts, 1);
        }
    }
}


__global__ void scce_loss_and_accuracy_kernel_accumulate(
    const float *probabilities,
    const uint8_t *labels,
    float *d_batch_losses,
    float *d_epoch_total_loss,
    int *d_epoch_total_correct,
    int batch_size,
    int num_classes
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int sample_idx = thread_id; sample_idx < batch_size; sample_idx += stride) {
        int label = labels[sample_idx];
        int offset = sample_idx * num_classes;

        float prob_true_class = probabilities[offset + label];
        prob_true_class = fmaxf(prob_true_class, 1e-9f);
        float sample_loss = -logf(prob_true_class);

        if (d_batch_losses != NULL) {
            d_batch_losses[sample_idx] = sample_loss;
        }

        atomicAdd(d_epoch_total_loss, sample_loss);

        float max_prob = -1.0f;
        int predicted_label = -1;
        for (int i = 0; i < num_classes; ++i) {
            float p = probabilities[offset + i];
            if (p > max_prob) {
                max_prob = p;
                predicted_label = i;
            }
        }

        if (predicted_label == label) {
            atomicAdd(d_epoch_total_correct, 1);
        }
    }
}


__global__ void compute_logit_gradient_kernel(
    const float* probabilities,
    const uint8_t* labels,
    float* d_grad_logits,
    int batch_size,
    int num_classes,
    float scale_factor
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch_size * num_classes;
         idx += gridDim.x * blockDim.x) {

        int sample_idx = idx / num_classes;
        int class_idx = idx % num_classes; 

        float prob = probabilities[idx];
        uint8_t true_label = labels[sample_idx];

        float target = (class_idx == true_label) ? 1.0f : 0.0f;

        d_grad_logits[idx] = (prob - target) * scale_factor;
    }
}


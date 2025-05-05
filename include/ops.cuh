#ifndef OPS_CUH
#define OPS_CUH

#include <cstdint>

__global__ void convert_and_normalize(const uint8_t* input, float* output, int num_elements);

__global__ void init_weights_uniform(float* data, size_t size, size_t seed);

__global__ void matmul_kernel(float *output, const float *input, const float *weights, int input_size, int output_size, int batch_size);

__global__ void softmax(const float* logits, float* probabilities, int batch_size, int num_classes);

__global__ void scce_loss_forward_kernel(const float* probabilities,
    const uint8_t* true_labels, float* losses,
    int batch_size, int num_classes);

__global__ void scce_softmax_backward_kernel(const float* probabilities,
    const uint8_t* true_labels,
    float* grad_logits,
    int batch_size, int num_classes, float scale_factor);

__global__ void calculate_weight_gradient_kernel(float* grad_weights,
    const float* input_images,
    const float* grad_logits,
    int input_size,
    int output_size,
    int batch_size);

__global__ void update_weights_kernel(float* weights, const float* grad_weights, float learning_rate, int num_weights);


__global__ void scce_loss_forward_kernel_accumulate(
    const float *probabilities,
    const uint8_t *labels,
    float *d_batch_losses,
    float *d_epoch_total_loss,
    int batch_size,
    int num_classes);

__global__ void calculate_accuracy_kernel_accumulate(
    const float *probabilities,
    const uint8_t *labels,
    int *d_epoch_total_correct,
    int batch_size,
    int num_classes);

#endif
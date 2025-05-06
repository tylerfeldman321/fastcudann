#ifndef TRAIN_CUH
#define TRAIN_CUH

#include <cstdint>

#define BLOCK_SIZE_1D 256
#define BLOCK_DIM_2D 16

__global__ void calculate_accuracy_kernel(
    const float* probabilities,
    const uint8_t* true_labels,
    int* correct_counts,
    int batch_size,
    int num_classes
);

bool run_training_basic_implementation(
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate
);

bool run_training_optimized(
    float *d_all_train_images_float, 
    uint8_t *d_all_train_labels,     
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate,
    int loss_print_period = 10       
);

bool run_training_optimized_cudnn_and_graphs(
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate,
    int loss_print_period = 10
);

#endif // TRAIN_CUH
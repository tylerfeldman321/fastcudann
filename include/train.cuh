#ifndef TRAIN_CUH
#define TRAIN_CUH

#include <cstdint>

#define BLOCK_SIZE_1D 256
#define BLOCK_DIM_2D 16

__global__ void calculate_accuracy_kernel(const float* probabilities, const uint8_t* true_labels,
    int* correct_counts,
    int batch_size, int num_classes);

bool run_training_basic_implementation(float *d_all_train_images_float, // Pointer to ALL training images on device
    uint8_t *d_all_train_labels,     // Pointer to ALL training labels on device
    int total_train_samples,         // Total number of training samples (e.g., 60000)
    int input_size,                  // Size of one input image (e.g., 784)
    int output_size,                 // Number of output classes (e.g., 10)
    int num_epochs,                  // Number of epochs to train
    int mini_batch_size,             // Mini-batch size
    float learning_rate              // Learning rate for optimizer
    );

bool run_training_optimized(
    float *d_all_train_images_float, // Pointer to ALL training images on device
    uint8_t *d_all_train_labels,     // Pointer to ALL training labels on device
    int total_train_samples,         // Total number of training samples (e.g., 60000)
    int input_size,                  // Size of one input image (e.g., 784)
    int output_size,                 // Number of output classes (e.g., 10)
    int num_epochs,                  // Number of epochs to train
    int mini_batch_size,             // Mini-batch size
    float learning_rate,             // Learning rate for optimizer
    int loss_print_period = 10       // How often to compute/print loss (epochs)
);

bool run_training_cudnn( // Renamed for clarity
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,      // Number of features per image (e.g., 784 for MNIST)
    int output_size,     // Number of classes (e.g., 10 for MNIST)
    int num_epochs,
    int mini_batch_size,
    float learning_rate,
    int loss_print_period = 10
);

#endif // TRAIN_CUH
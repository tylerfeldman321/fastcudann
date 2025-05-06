#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include "../include/mnist_reader_common.hpp"
#include "../include/utils.cuh"
#include "../include/ops.cuh"
#include "../include/train.cuh"


#define WARMUP_EPOCHS 0
#define NUM_EPOCHS 20
#define LEARNING_RATE 0.01f
#define MINI_BATCH_SIZE 128


void print_mnist_image(const uint8_t* image_data, int rows, int columns, uint8_t label) {
    std::cout << "Label: " << static_cast<int>(label) << std::endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            uint8_t pixel = image_data[r * columns + c];
            std::cout << (pixel > 128 ? '#' : ' ');
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------" << std::endl;
}


int main(int argc, char* argv[]) {
    std::string MNIST_DATA_LOCATION = "./data";
     if (argc > 1) {
        MNIST_DATA_LOCATION = argv[1];
    }
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Read train images
    auto mnist_train_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-images-idx3-ubyte", 0x803);
    int train_images_count   = static_cast<int>(read_header(mnist_train_data_buffer, 1));
    int train_images_rows    = static_cast<int>(read_header(mnist_train_data_buffer, 2));
    int train_images_columns = static_cast<int>(read_header(mnist_train_data_buffer, 3));
    uint8_t* train_images = reinterpret_cast<uint8_t*>(mnist_train_data_buffer.get() + 16);
    int input_feature_size = train_images_rows * train_images_columns; // Should be 784
    std::cout << "Train images: " << train_images_count << " [" << train_images_rows << "x" << train_images_columns << "]" << std::endl;

    // Read train labels
    auto mnist_train_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-labels-idx1-ubyte", 0x801);
    auto train_labels_count = read_header(mnist_train_labels_data_buffer, 1);
    auto train_labels = reinterpret_cast<uint8_t*>(mnist_train_labels_data_buffer.get() + 8);
    std::cout << "Train labels: " << train_labels_count << std::endl;

    // // Read test images (optional - not used in training, but good to have)
    // auto mnist_test_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-images-idx3-ubyte", 0x803);
    // int test_images_count   = static_cast<int>(read_header(mnist_test_data_buffer, 1));
    // int test_images_rows    = static_cast<int>(read_header(mnist_test_data_buffer, 2));
    // int test_images_columns = static_cast<int>(read_header(mnist_test_data_buffer, 3));
    // uint8_t* test_images = reinterpret_cast<uint8_t*>(mnist_test_data_buffer.get() + 16);
    // std::cout << "Test images: " << test_images_count << std::endl;

    // // Read test labels
    // auto mnist_test_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-labels-idx1-ubyte", 0x801);
    // auto test_labels_count = read_header(mnist_test_labels_data_buffer, 1);
    // auto test_labels = reinterpret_cast<uint8_t*>(mnist_test_labels_data_buffer.get() + 8);
    // std::cout << "Test labels: " << test_labels_count << std::endl;

    if (train_images_count != train_labels_count) {
        std::cerr << "Error: Mismatch between number of training images and labels." << std::endl;
        return 1;
    }

    // --- GPU Data Preparation ---
    uint8_t *d_train_images_uint8; // Temporary device buffer for uint8 images
    uint8_t *d_all_train_labels;   // Device buffer for all labels
    float *d_all_train_images_float; // Device buffer for all normalized float images

    size_t num_training_pixels = (size_t)train_images_count * input_feature_size;
    size_t training_images_bytes_uint8 = sizeof(uint8_t) * num_training_pixels;
    size_t training_images_bytes_float = sizeof(float) * num_training_pixels;
    size_t training_labels_bytes = sizeof(uint8_t) * train_labels_count;

    // Allocate temporary uint8 image buffer and final label/float image buffers
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_train_images_uint8, training_images_bytes_uint8));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_all_train_labels, training_labels_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_all_train_images_float, training_images_bytes_float));

    // Copy images (uint8) and labels to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_train_images_uint8, train_images, training_images_bytes_uint8, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_all_train_labels, train_labels, training_labels_bytes, cudaMemcpyHostToDevice));

    // Convert uint8_t image data to normalized floats on GPU
    int convert_grid_size = calculate_grid_size_1d(num_training_pixels, BLOCK_SIZE_1D);
    convert_and_normalize<<<convert_grid_size, BLOCK_SIZE_1D>>>(d_train_images_uint8, d_all_train_images_float, num_training_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Free the temporary uint8 buffer on device
    CHECK_CUDA_ERROR(cudaFree(d_train_images_uint8));

    int num_classes = 10;

    printf("------------ Running Basic NN Training Implementation ------------\n");
    using Clock = std::chrono::high_resolution_clock;
    auto overall_start_time = Clock::now();
    run_training_basic_implementation(d_all_train_images_float, d_all_train_labels,
                 train_images_count, input_feature_size, num_classes,
                 NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE);
    auto overall_end_time = Clock::now();
    auto overall_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);
    double overall_duration_s = overall_duration_ms.count() / 1000.0;
    printf("Overall Training Wall Time for basic MNIST training implementation: %lld ms (%.3f s)\n", overall_duration_ms.count(), overall_duration_s);

    printf("------------ Running Optimized NN  Training Implementation ------------\n");
    using Clock = std::chrono::high_resolution_clock;
    auto overall_start_time_optimized = Clock::now();
    run_training_optimized(d_all_train_images_float, d_all_train_labels,
                 train_images_count, input_feature_size, num_classes,
                 NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE);
    auto overall_end_time_optimized = Clock::now();
    auto overall_duration_ms_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time_optimized - overall_start_time_optimized);
    double overall_duration_s_optimized = overall_duration_ms_optimized.count() / 1000.0;
    printf("Overall Training Wall Time for optimized MNIST training implementation: %lld ms (%.3f s)\n", overall_duration_ms_optimized.count(), overall_duration_s_optimized);

    printf("------------ Running Optimized + CUDNN Softmax + Graphs NN Training Implementation ------------\n");
    using Clock = std::chrono::high_resolution_clock;
    auto overall_start_time_cudnn = Clock::now();
    run_training_optimized_cudnn_and_graphs(d_all_train_images_float, d_all_train_labels,
                 train_images_count, input_feature_size, num_classes,
                 NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE);
    auto overall_end_time_cudnn = Clock::now();
    auto overall_duration_ms_cudnn = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time_cudnn - overall_start_time_cudnn);
    double overall_duration_s_cudnn = overall_duration_ms_cudnn.count() / 1000.0;
    printf("Overall Training Wall Time for CUDNN MNIST training implementation: %lld ms (%.3f s)\n", overall_duration_ms_cudnn.count(), overall_duration_s_cudnn);

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaFree(d_all_train_images_float));
    CHECK_CUDA_ERROR(cudaFree(d_all_train_labels));

    printf("Exiting program.\n");
    return 0;
}
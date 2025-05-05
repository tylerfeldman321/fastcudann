#include <iostream>
#include "../include/mnist_reader_common.hpp"
#include "../include/utils.cuh"
#include "../include/ops.cuh"


#define WARMUP_EPOCHS 5  // Number of warmup epochs (not counted in timing)


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


bool run_basic_implementation(float *d_train_images, uint8_t *d_train_labels) {
    // Run my implementation w/o graphs
    size_t input_size = 784;
    size_t output_size = 10;
    size_t batch_size = 60000;
    float *h_weights = (float*)calloc(input_size*output_size, sizeof(float));

    float *d_weights, *d_output, *d_probabilities, *d_losses;
    checkCudaError(cudaMalloc((void**)&d_weights, sizeof(float)*input_size*output_size), "Allocating d_weights");
    checkCudaError(cudaMalloc((void**)&d_output, sizeof(float)*batch_size*output_size), "Allocating d_output");
    checkCudaError(cudaMalloc((void**)&d_probabilities, sizeof(float)*batch_size*output_size), "Allocating d_probabilities");
    checkCudaError(cudaMalloc((void**)&d_losses, sizeof(float)*batch_size), "Allocating d_losses");

    // float *d_error, *d_gradient;
    // checkCudaError(cudaMalloc((void**)&d_error, sizeof(float)*batch_size*output_size), "Allocating d_error");
    // checkCudaError(cudaMalloc((void**)&d_gradient, sizeof(float) * INPUT_SIZE), "Allocating d_gradient");
    // checkCudaError(cudaMalloc((void**)&d_train_labels, sizeof(h_target)), "Allocating d_target");


    dim3 gridSize(1, 1);
    dim3 blockSize(1, 1);
    init_weights_uniform<<<1, 1>>>(d_weights, input_size*output_size, 0);

    matmul_kernel<<<gridSize, blockSize>>>(d_output, d_train_images, d_weights, input_size, output_size, batch_size);
    softmax<<<1, 1>>>(d_output, d_probabilities, batch_size, output_size);
    scce_loss_forward_kernel<<<1, 1>>>(d_probabilities, d_train_labels, d_losses, batch_size, output_size);
    // TODO: Reduce d_losses

    // sigmoid_kernel<<<1, 1>>>(d_output, d_output, batch_size*output_size);
    // calc_error_kernel<<<1, 1>>>(d_error, d_output, d_train_labels, batch_size*output_size);
    // calc_gradient_kernel<<<gridSize_input, blockSize, 0, baseline_stream>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);
    // update_weights_kernel<<<gridSize_weights, blockSize, 0, baseline_stream>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);

	// checkCudaError( cudaDeviceSynchronize() , "Sync before return");
	// checkCudaError( cudaGetLastError() , "matmul_kernel");

    // matmul_kernel<<<BATCH_SIZE, OUTPUT_SIZE, 0, stream>>>(d_output, d_input, d_weights, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    // sigmoid_kernel<<<gridSize_batch_output, blockSize, 0, stream>>>(d_output, d_output, BATCH_SIZE * OUTPUT_SIZE);
    // calc_error_kernel<<<gridSize_batch_output, blockSize, 0, stream>>>(d_error, d_output, d_target, BATCH_SIZE * OUTPUT_SIZE);
    // calc_gradient_kernel<<<gridSize_input, blockSize, 0, stream>>>(d_gradient, d_input, d_error, INPUT_SIZE, BATCH_SIZE);
    // update_weights_kernel<<<gridSize_weights, blockSize, 0, stream>>>(d_weights, d_gradient, LEARNING_RATE, INPUT_SIZE * OUTPUT_SIZE);

    checkCudaError(cudaMemcpy(h_weights, d_weights, sizeof(float)*input_size*output_size, cudaMemcpyDeviceToHost), "Copying weights to host");
    checkCudaError( cudaDeviceSynchronize() , "Sync before accessing weights");
    std::cout << "Weights: ";
    for (int i = 0; i < 10; i++)
        std::cout << h_weights[i] << " ";
    std::cout << std::endl;
    free(h_weights);

    float* h_output = (float*)malloc(batch_size*output_size*sizeof(float));
    checkCudaError(cudaMemcpy(h_output, d_output, sizeof(float)*batch_size*output_size, cudaMemcpyDeviceToHost), "Copying output to host");
    checkCudaError( cudaDeviceSynchronize() , "Sync before accessing output");
    std::cout << "Logits: ";
    for (int i = 0; i < 10; i++)
        std::cout << h_output[i] << " ";
    std::cout << std::endl;
    free(h_output);

    float* h_prob = (float*)malloc(batch_size*output_size*sizeof(float));
    checkCudaError(cudaMemcpy(h_prob, d_probabilities, sizeof(float)*batch_size*output_size, cudaMemcpyDeviceToHost), "Copying prob to host");
    checkCudaError( cudaDeviceSynchronize() , "Sync before accessing prob");
    std::cout << "Probabilities: ";
    float sum = 0;
    for (int i = 0; i < 10; i++){
        std::cout << h_prob[i] << " ";
        sum += h_prob[i];}
    std::cout << std::endl;
    std::cout << sum << std::endl;
    free(h_prob);

    return true;
}


int main(int argc, char* argv[]) {
    std::string MNIST_DATA_LOCATION = "/home/ubuntu/fastcudann/data";
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    
    // Read train images
    auto mnist_train_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-images.idx3-ubyte", 0x803);
    int train_images_count   = static_cast<int>(read_header(mnist_train_data_buffer, 1));
    int train_images_rows    = static_cast<int>(read_header(mnist_train_data_buffer, 2));
    int train_images_columns = static_cast<int>(read_header(mnist_train_data_buffer, 3));
    uint8_t* train_images = reinterpret_cast<uint8_t*>(mnist_train_data_buffer.get() + 16);
    std::cout << train_images_count << std::endl;

    // Read train labels
    auto mnist_train_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-labels.idx1-ubyte", 0x801);
    auto train_labels_count = read_header(mnist_train_labels_data_buffer, 1);
    auto train_labels = reinterpret_cast<uint8_t*>(mnist_train_labels_data_buffer.get() + 8);
    std::cout << train_labels_count << std::endl;

    // Read test images
    auto mnist_test_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-images.idx3-ubyte", 0x803);
    int test_images_count   = static_cast<int>(read_header(mnist_test_data_buffer, 1));
    int test_images_rows    = static_cast<int>(read_header(mnist_test_data_buffer, 2));
    int test_images_columns = static_cast<int>(read_header(mnist_test_data_buffer, 3));
    uint8_t* test_images = reinterpret_cast<uint8_t*>(mnist_test_data_buffer.get() + 16);
    std::cout << test_images_count << std::endl;

    // Read test labels
    auto mnist_test_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-labels.idx1-ubyte", 0x801);
    auto test_labels_count = read_header(mnist_test_labels_data_buffer, 1);
    auto test_labels = reinterpret_cast<uint8_t*>(mnist_test_labels_data_buffer.get() + 8);
    std::cout << test_labels_count << std::endl;

    // // --- Print Samples ---
    // std::cout << "\n--- Printing Training Samples ---" << std::endl;
    // int image_size = train_images_rows * train_images_columns; // Should be 784 for MNIST
    // print_mnist_image(train_images + 0 * image_size, train_images_rows, train_images_columns, train_labels[0]);
    // print_mnist_image(train_images + 1 * image_size, train_images_rows, train_images_columns, train_labels[1]);
    // std::cout << "\n--- Printing Testing Samples ---" << std::endl;
    // int test_image_size = test_images_rows * test_images_columns;
    // print_mnist_image(test_images + 0 * test_image_size, test_images_rows, test_images_columns, test_labels[0]);
    // print_mnist_image(test_images + 1 * test_image_size, test_images_rows, test_images_columns, test_labels[1]);

    uint8_t *d_train_images_uint8, *d_train_labels;
    size_t num_training_pixels = (size_t)train_images_count*train_images_rows*train_images_columns;
    size_t training_labels_size = sizeof(uint8_t)*train_labels_count;
    checkCudaError(cudaMalloc((void**)&d_train_images_uint8, sizeof(uint8_t)*num_training_pixels), "Allocating d_train_images_uint8");
    checkCudaError(cudaMalloc((void**)&d_train_labels, training_labels_size), "Allocating d_train_labels");

    checkCudaError(cudaMemcpy(d_train_images_uint8, train_images, sizeof(uint8_t)*num_training_pixels, cudaMemcpyHostToDevice), "Copying training images to device");
    checkCudaError(cudaMemcpy(d_train_labels, train_labels, training_labels_size, cudaMemcpyHostToDevice), "Copying training labels to device");

    // Convert uint8_t data to normalized floats
    float *d_train_images_float;
    checkCudaError(cudaMalloc((void**)&d_train_images_float, sizeof(float) * num_training_pixels), "Allocating d_train_images_float");
    convert_and_normalize<<<256, 256>>>(d_train_images_uint8, d_train_images_float, num_training_pixels);
    checkCudaError(cudaFree(d_train_images_uint8), "Freeing d_train_images");

    run_basic_implementation(d_train_images_float, d_train_labels);
        
    // TODO: Run my implementation with graphs
    // TODO: Run my implementation with graphs, reduce synchronization
    // TODO: Run cudNN

    checkCudaError(cudaFree(d_train_labels), "Freeing d_train_labels");

    return 0;
}
#include <iostream>
#include "../include/mnist_reader_common.hpp"
#include "../include/utils.cuh"
#include "../include/ops.cuh"


#define WARMUP_EPOCHS 0
#define NUM_EPOCHS 20
#define LEARNING_RATE 0.01f
#define MINI_BATCH_SIZE 128
#define BLOCK_SIZE_1D 256
#define BLOCK_DIM_2D 16


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

// Function to calculate accuracy on the device (avoids transferring probabilities)
__global__ void calculate_accuracy_kernel(const float* probabilities, const uint8_t* true_labels,
                                          int* correct_counts,
                                          int batch_size, int num_classes) {
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
            atomicAdd(correct_counts, 1); // Atomically add 1 to the shared counter
        }
    }
}


bool run_training(float *d_all_train_images_float, // Pointer to ALL training images on device
                  uint8_t *d_all_train_labels,     // Pointer to ALL training labels on device
                  int total_train_samples,         // Total number of training samples (e.g., 60000)
                  int input_size,                  // Size of one input image (e.g., 784)
                  int output_size,                 // Number of output classes (e.g., 10)
                  int num_epochs,                  // Number of epochs to train
                  int mini_batch_size,             // Mini-batch size
                  float learning_rate              // Learning rate for optimizer
                  )
{
    printf("Starting training...\n");
    printf("Parameters:\n");
    printf("  Epochs: %d\n", num_epochs);
    printf("  Mini-batch Size: %d\n", mini_batch_size);
    printf("  Learning Rate: %f\n", learning_rate);
    printf("  Input Size: %d\n", input_size);
    printf("  Output Size (Classes): %d\n", output_size);
    printf("  Total Training Samples: %d\n", total_train_samples);

    int num_weights = input_size * output_size;
    size_t weights_bytes = sizeof(float) * num_weights;
    // Allocate intermediate buffers based on MINI_BATCH_SIZE
    size_t output_bytes = sizeof(float) * mini_batch_size * output_size;
    size_t loss_bytes = sizeof(float) * mini_batch_size;
    size_t accuracy_counter_bytes = sizeof(int); // For single atomic counter

    float *h_losses = (float*)malloc(loss_bytes); // Host buffer for losses of one mini-batch
    if (!h_losses) { fprintf(stderr, "Failed to allocate host memory for losses\n"); return false; }
    int h_correct_count = 0; // Host variable for accuracy count

    // --- Device Memory Allocation ---
    float *d_weights, *d_output, *d_probabilities, *d_losses, *d_grad_logits, *d_grad_weights;
    int *d_correct_count;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_weights, weights_bytes));
    // Allocate based on mini_batch_size for intermediate results
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_probabilities, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_losses, loss_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_logits, output_bytes));
    // Allocate and initialize accuracy counter on device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_correct_count, accuracy_counter_bytes));

    // --- Initialize Weights ---
    int init_grid_size = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
    init_weights_uniform<<<init_grid_size, BLOCK_SIZE_1D>>>(d_weights, num_weights, time(0));
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- Training Loop ---
    int num_batches = (total_train_samples + mini_batch_size - 1) / mini_batch_size;
    printf("Total mini-batches per epoch: %d\n", num_batches);

    // Define block sizes
    dim3 block_1d(BLOCK_SIZE_1D);
    dim3 block_2d(BLOCK_DIM_2D, BLOCK_DIM_2D);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        double epoch_total_loss = 0.0;
        long long epoch_total_correct = 0;
        long long epoch_total_processed = 0;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start_idx = batch_idx * mini_batch_size;
            // Determine the actual number of samples in this batch (can be smaller for the last one)
            int current_batch_size = (batch_start_idx + mini_batch_size > total_train_samples) ?
                                      (total_train_samples - batch_start_idx) :
                                      mini_batch_size;

            if (current_batch_size <= 0) continue; // Should not happen with correct num_batches calculation

            // Calculate pointers to the current mini-batch data within the full dataset buffers
            // NOTE: For true SGD with shuffling, you'd ideally copy shuffled data to contiguous
            // batch buffers or use indirect addressing via shuffled indices.
            // This simplified approach uses contiguous slices but iterates in shuffled batch order.
            // We will use the shuffled indices 'p' to get the *starting* sample index for the slice.
            // This isn't perfect shuffling *within* the slice but better than nothing.
            // A truly robust implementation would gather data based on 'p'. Let's keep it simple for now.
            // int effective_start_idx = p[batch_start_idx]; // Index of the first sample for this batch slice
            // float* d_current_batch_images = d_all_train_images_float + effective_start_idx * input_size;
            // uint8_t* d_current_batch_labels = d_all_train_labels + effective_start_idx;
            // ^^^ This simple slicing with shuffled start isn't ideal as slices might overlap/miss data if not careful.

            // --- Alternative: Process batches sequentially but shuffle order ---
            // This is simpler to implement correctly with pointer offsets.
            // The order of batches is randomized by the outer loop's access to `p`.
            // We'll process the data corresponding to indices p[batch_start_idx] through p[batch_start_idx + current_batch_size - 1].
            // This requires **gathering** data, which is more complex.

            // --- Simplest Approach for this Example: Sequential Slice Access ---
            // We will iterate through batches sequentially but rely on the fact
            // that weight updates happen frequently. We *won't* use the shuffled indices `p` directly
            // for memory access here to avoid complexity, but acknowledge it's less ideal than true shuffling.
            float* d_current_batch_images = d_all_train_images_float + batch_start_idx * input_size;
            uint8_t* d_current_batch_labels = d_all_train_labels + batch_start_idx;


            // --- Forward Pass ---
            // 1. Calculate Logits (d_output = d_current_batch_images * d_weights)
            dim3 matmul_grid = calculate_grid_size_2d(current_batch_size, output_size, block_2d);
            matmul_kernel<<<matmul_grid, block_2d>>>(d_output, d_current_batch_images, d_weights, input_size, output_size, current_batch_size);

            // 2. Calculate Probabilities (d_probabilities = softmax(d_output))
            int softmax_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
            softmax<<<softmax_grid, block_1d>>>(d_output, d_probabilities, current_batch_size, output_size);

            // 3. Calculate Loss (per sample for the current batch)
            int loss_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
            scce_loss_forward_kernel<<<loss_grid, block_1d>>>(d_probabilities, d_current_batch_labels, d_losses, current_batch_size, output_size);


            // --- Loss Calculation & Logging ---
            // Copy losses for the current batch from Device to Host
            // Use cudaMemcpyAsync with a stream for potential overlap if needed later.
            CHECK_CUDA_ERROR(cudaMemcpy(h_losses, d_losses, sizeof(float) * current_batch_size, cudaMemcpyDeviceToHost));
            // Synchronize *only* to ensure the loss copy is complete before CPU access.
            // Kernels for this batch might still be running.
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Essential before CPU access to h_losses

            // Sum losses on the CPU for this batch
            double current_batch_total_loss = 0.0;
            for (int i = 0; i < current_batch_size; ++i) {
                current_batch_total_loss += h_losses[i];
            }
            epoch_total_loss += current_batch_total_loss;
            epoch_total_processed += current_batch_size;


             // --- Calculate Accuracy (on GPU) ---
            CHECK_CUDA_ERROR(cudaMemset(d_correct_count, 0, accuracy_counter_bytes)); // Reset counter for the batch
            int accuracy_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
            calculate_accuracy_kernel<<<accuracy_grid, block_1d>>>(d_probabilities, d_current_batch_labels, d_correct_count, current_batch_size, output_size);
            // Copy the result back from GPU
            CHECK_CUDA_ERROR(cudaMemcpy(&h_correct_count, d_correct_count, accuracy_counter_bytes, cudaMemcpyDeviceToHost));
            // Synchronize *only* to ensure the accuracy count copy is complete.
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Essential before CPU access to h_correct_count
            epoch_total_correct += h_correct_count;


            // --- Backward Pass ---
            // 4. Calculate Gradient of Loss w.r.t. Logits (dL/dZ)
            float grad_scale_factor = 1.0f / (float)current_batch_size; // Average gradient over the batch
            int backward_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D); // Based on batch size
            scce_softmax_backward_kernel<<<backward_grid, block_1d>>>(d_probabilities, d_current_batch_labels, d_grad_logits, current_batch_size, output_size, grad_scale_factor);

            // 5. Calculate Gradient of Loss w.r.t Weights (dL/dW = X^T * dL/dZ)
            // Grid depends on weight matrix dimensions (input_size x output_size)
            dim3 grad_weights_grid = calculate_grid_size_2d(input_size, output_size, block_2d);
            calculate_weight_gradient_kernel<<<grad_weights_grid, block_2d>>>(d_grad_weights, d_current_batch_images, d_grad_logits, input_size, output_size, current_batch_size);

            // --- Update Weights ---
            // 6. Apply gradient descent step (Weights = Weights - LR * dL/dW)
            int update_grid = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
            update_weights_kernel<<<update_grid, block_1d>>>(d_weights, d_grad_weights, learning_rate, num_weights);

            // Check for errors periodically (e.g., end of batch) - essential for debugging
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Calculate and log average loss and accuracy for the epoch
        float average_loss = (epoch_total_processed > 0) ? (float)(epoch_total_loss / epoch_total_processed) : 0.0f;
        float accuracy = (epoch_total_processed > 0) ? (float)(epoch_total_correct * 100.0 / epoch_total_processed) : 0.0f;

        printf("Epoch [%d/%d], Average Loss: %.6f, Accuracy: %.2f%%\n",
               epoch + 1, num_epochs, average_loss, accuracy);


    }

    printf("Training finished.\n");

    // // --- Optional: Print some final weights/outputs (from the last batch state) ---
    // // Note: These outputs (logits, probs) only reflect the *last* mini-batch processed.
    // float* h_weights = (float*)malloc(weights_bytes);
    // CHECK_CUDA_ERROR(cudaMemcpy(h_weights, d_weights, weights_bytes, cudaMemcpyDeviceToHost));
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // std::cout << "Final Weights (first 10): ";
    // for (int i = 0; i < 10 && i < num_weights; i++)
    //     std::cout << std::fixed << std::setprecision(4) << h_weights[i] << " ";
    // std::cout << std::endl;
    // free(h_weights);

    // --- Cleanup ---
    free(h_losses);
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_probabilities));
    CHECK_CUDA_ERROR(cudaFree(d_losses));
    CHECK_CUDA_ERROR(cudaFree(d_grad_logits));
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_correct_count));

    return true;
}


int main(int argc, char* argv[]) {
    std::string MNIST_DATA_LOCATION = "./data";
     if (argc > 1) {
        MNIST_DATA_LOCATION = argv[1];
    }
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Read train images
    auto mnist_train_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-images.idx3-ubyte", 0x803);
    int train_images_count   = static_cast<int>(read_header(mnist_train_data_buffer, 1));
    int train_images_rows    = static_cast<int>(read_header(mnist_train_data_buffer, 2));
    int train_images_columns = static_cast<int>(read_header(mnist_train_data_buffer, 3));
    uint8_t* train_images = reinterpret_cast<uint8_t*>(mnist_train_data_buffer.get() + 16);
    int input_feature_size = train_images_rows * train_images_columns; // Should be 784
    std::cout << "Train images: " << train_images_count << " [" << train_images_rows << "x" << train_images_columns << "]" << std::endl;

    // Read train labels
    auto mnist_train_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/train-labels.idx1-ubyte", 0x801);
    auto train_labels_count = read_header(mnist_train_labels_data_buffer, 1);
    auto train_labels = reinterpret_cast<uint8_t*>(mnist_train_labels_data_buffer.get() + 8);
    std::cout << "Train labels: " << train_labels_count << std::endl;

    // Read test images (optional - not used in training, but good to have)
    auto mnist_test_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-images.idx3-ubyte", 0x803);
    int test_images_count   = static_cast<int>(read_header(mnist_test_data_buffer, 1));
    int test_images_rows    = static_cast<int>(read_header(mnist_test_data_buffer, 2));
    int test_images_columns = static_cast<int>(read_header(mnist_test_data_buffer, 3));
    uint8_t* test_images = reinterpret_cast<uint8_t*>(mnist_test_data_buffer.get() + 16);
    std::cout << "Test images: " << test_images_count << std::endl;

    // Read test labels
    auto mnist_test_labels_data_buffer = read_mnist_file(MNIST_DATA_LOCATION + "/t10k-labels.idx1-ubyte", 0x801);
    auto test_labels_count = read_header(mnist_test_labels_data_buffer, 1);
    auto test_labels = reinterpret_cast<uint8_t*>(mnist_test_labels_data_buffer.get() + 8);
    std::cout << "Test labels: " << test_labels_count << std::endl;

    if (train_images_count != train_labels_count) {
        std::cerr << "Error: Mismatch between number of training images and labels." << std::endl;
        return 1;
    }

    // --- GPU Data Preparation ---
    uint8_t *d_train_images_uint8; // Temporary device buffer for uint8 images
    uint8_t *d_all_train_labels;   // Device buffer for ALL labels
    float *d_all_train_images_float; // Device buffer for ALL normalized float images

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

    // --- Run Training ---
    int num_classes = 10;
    run_training(d_all_train_images_float, d_all_train_labels,
                 train_images_count, input_feature_size, num_classes,
                 NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE);

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaFree(d_all_train_images_float));
    CHECK_CUDA_ERROR(cudaFree(d_all_train_labels));

    printf("Exiting program.\n");
    return 0;
}
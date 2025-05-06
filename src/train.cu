#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "../include/train.cuh"
#include "../include/utils.cuh"
#include "../include/ops.cuh"

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


bool run_training_basic_implementation(float *d_all_train_images_float, // Pointer to ALL training images on device
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

    // --- Timing Setup ---
    cudaEvent_t epoch_start_event, epoch_stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_stop_event));
    float epoch_gpu_time_ms = 0.0f;

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

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_start_event, 0));

        double epoch_total_loss = 0.0;
        long long epoch_total_correct = 0;
        long long epoch_total_processed = 0;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start_idx = batch_idx * mini_batch_size;
            int current_batch_size = (batch_start_idx + mini_batch_size > total_train_samples) ?
                                      (total_train_samples - batch_start_idx) :
                                      mini_batch_size;

            if (current_batch_size <= 0) continue;
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
            CHECK_CUDA_ERROR(cudaMemcpy(h_losses, d_losses, sizeof(float) * current_batch_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // Synchronize ensure the accuracy count copy is complete.
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

        // --- Record Epoch Stop Time (GPU) and Calculate Duration ---
        CHECK_CUDA_ERROR(cudaEventRecord(epoch_stop_event, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(epoch_stop_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&epoch_gpu_time_ms, epoch_start_event, epoch_stop_event));

        // Calculate and log average loss and accuracy for the epoch
        float average_loss = (epoch_total_processed > 0) ? (float)(epoch_total_loss / epoch_total_processed) : 0.0f;
        float accuracy = (epoch_total_processed > 0) ? (float)(epoch_total_correct * 100.0 / epoch_total_processed) : 0.0f;

        printf("Epoch [%d/%d], Average Loss: %.6f, Accuracy: %.2f%%, Epoch GPU Time: %.2f ms (%.3f s)\n",
                epoch + 1, num_epochs, average_loss, accuracy, epoch_gpu_time_ms, epoch_gpu_time_ms / 1000.0f);
    }
    printf("Training complete!\n");

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


bool run_training_optimized(
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate,
    int loss_print_period
) {
    printf("Starting training (Periodic Loss Reporting - cuBLAS)...\n"); // Indicate cuBLAS usage
    printf("Parameters:\n");
    printf("  Epochs: %d\n", num_epochs);
    printf("  Mini-batch Size: %d\n", mini_batch_size);
    printf("  Learning Rate: %f\n", learning_rate);
    printf("  Input Size: %d\n", input_size);
    printf("  Output Size (Classes): %d\n", output_size);
    printf("  Total Training Samples: %d\n", total_train_samples);
    printf("  Loss Print Period: %d epochs\n", loss_print_period);

    // --- cuBLAS Setup ---
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // --- Timing Setup ---
    cudaEvent_t epoch_start_event, epoch_stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_stop_event));
    float epoch_gpu_time_ms = 0.0f;

    // --- Memory Sizes ---
    int num_weights = input_size * output_size;
    size_t weights_bytes = sizeof(float) * num_weights;
    size_t output_bytes = sizeof(float) * mini_batch_size * output_size;
    size_t batch_loss_bytes = sizeof(float) * mini_batch_size;
    size_t scalar_float_bytes = sizeof(float);
    size_t scalar_int_bytes = sizeof(int);

    // --- Host Variables for Periodic Reporting ---
    float h_epoch_total_loss = 0.0f;
    int h_epoch_total_correct = 0;

    // --- Device Memory Allocation ---
    float *d_weights, *d_grad_weights;
    float *d_output, *d_probabilities, *d_grad_logits;
    float *d_batch_losses;
    float *d_epoch_total_loss;
    int   *d_epoch_total_correct;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_probabilities, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_logits, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_losses, batch_loss_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_epoch_total_loss, scalar_float_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_epoch_total_correct, scalar_int_bytes));

    // --- Initialize Weights ---
    int init_grid_size = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
    init_weights_uniform<<<init_grid_size, BLOCK_SIZE_1D>>>(d_weights, num_weights, time(0));
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- Training Loop ---
    int num_batches = (total_train_samples + mini_batch_size - 1) / mini_batch_size;
    printf("Total mini-batches per epoch: %d\n", num_batches);

    dim3 block_1d(BLOCK_SIZE_1D);

    // --- cuBLAS Scalars ---
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_start_event, 0));

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_loss, 0, scalar_float_bytes, 0));
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_correct, 0, scalar_int_bytes, 0));

        bool should_print_loss = ((epoch + 1) % loss_print_period == 0) || (epoch == num_epochs - 1);
        long long epoch_total_processed = 0;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start_idx = batch_idx * mini_batch_size;
            int current_batch_size = (batch_start_idx + mini_batch_size > total_train_samples) ?
                                      (total_train_samples - batch_start_idx) :
                                      mini_batch_size;

            if (current_batch_size <= 0) continue;

            float* d_current_batch_images = d_all_train_images_float + batch_start_idx * input_size;
            uint8_t* d_current_batch_labels = d_all_train_labels + batch_start_idx;

            // --- Forward Pass ---
            // 1. Calculate Logits (d_output = d_current_batch_images * d_weights)
            cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_size, current_batch_size, input_size,
                &alpha,
                d_weights, output_size,
                d_current_batch_images, input_size,
                &beta,
                d_output, output_size);

            // 2. Calculate Probabilities (d_probabilities = softmax(d_output))
            int softmax_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
            softmax<<<softmax_grid, block_1d>>>(d_output, d_probabilities, current_batch_size, output_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            if (should_print_loss) {
                // --- Loss and Accuracy Calculation (GPU Accumulation) ---
                // 3 & 4. Calculate total loss and accuracy and accumulate total correct count on GPU
                int accuracy_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
                scce_loss_and_accuracy_kernel_accumulate<<<accuracy_grid, block_1d>>>(
                    d_probabilities,
                    d_current_batch_labels,
                    d_batch_losses,
                    d_epoch_total_loss,
                    d_epoch_total_correct,
                    current_batch_size,
                    output_size
                );
                CHECK_CUDA_ERROR(cudaGetLastError());
            }

            // --- Backward Pass ---
            // 5. Calculate Gradient of Loss w.r.t. Logits (dL/dZ)
            float grad_scale_factor = 1.0f / (float)current_batch_size;
            int backward_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
            scce_softmax_backward_kernel<<<backward_grid, block_1d>>>(d_probabilities, d_current_batch_labels, d_grad_logits, current_batch_size, output_size, grad_scale_factor);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // 6. Calculate Gradient of Loss w.r.t Weights (dL/dW = X^T * dL/dZ)
            cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                output_size, input_size, current_batch_size,
                &alpha,
                d_grad_logits, output_size,
                d_current_batch_images, input_size,
                &beta,
                d_grad_weights, output_size);

            // --- Update Weights ---
            // 7. Apply gradient descent step (Weights = Weights - LR * dL/dW)
            int update_grid = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
            update_weights_kernel<<<update_grid, block_1d>>>(d_weights, d_grad_weights, learning_rate, num_weights);
            CHECK_CUDA_ERROR(cudaGetLastError()); // Check after kernel launch

            // --- Update CPU counter for total processed samples ---
            epoch_total_processed += current_batch_size;
        }

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_stop_event, 0));

        // --- Conditional Loss/Accuracy Reporting ---
        if (should_print_loss) {
            CHECK_CUDA_ERROR(cudaEventSynchronize(epoch_stop_event));

            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_loss, d_epoch_total_loss, scalar_float_bytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_correct, d_epoch_total_correct, scalar_int_bytes, cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaEventElapsedTime(&epoch_gpu_time_ms, epoch_start_event, epoch_stop_event));

            long long total_samples_in_epoch = total_train_samples;

            float average_loss = (total_samples_in_epoch > 0) ? (h_epoch_total_loss / total_samples_in_epoch) : 0.0f;
            float accuracy = (total_samples_in_epoch > 0) ? (float)(h_epoch_total_correct * 100.0 / total_samples_in_epoch) : 0.0f;

            printf("Epoch [%d/%d], Average Loss: %.6f, Accuracy: %.2f%%, Epoch GPU Time: %.2f ms (%.3f s)\n",
                   epoch + 1, num_epochs, average_loss, accuracy, epoch_gpu_time_ms, epoch_gpu_time_ms / 1000.0f);

        }
        CHECK_CUDA_ERROR(cudaGetLastError());

    }

    printf("Training complete!\n");

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_stop_event));
    cublasDestroy(cublas_handle);

    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_probabilities));
    CHECK_CUDA_ERROR(cudaFree(d_batch_losses)); // Free if allocated
    CHECK_CUDA_ERROR(cudaFree(d_grad_logits));
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_correct));

    return true;
}


bool run_training_cudnn(
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate,
    int loss_print_period
) {
    printf("Starting training (Periodic Loss Reporting - cuBLAS + cuDNN)...\n");
    printf("Parameters:\n");
    printf("  Epochs: %d\n", num_epochs);
    printf("  Mini-batch Size: %d\n", mini_batch_size);
    printf("  Learning Rate: %f\n", learning_rate);
    printf("  Input Size: %d\n", input_size);
    printf("  Output Size (Classes): %d\n", output_size);
    printf("  Total Training Samples: %d\n", total_train_samples);
    printf("  Loss Print Period: %d epochs\n", loss_print_period);

    // --- cuBLAS & cuDNN Setup ---
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cublasCreate(&cublas_handle);
    cudnnCreate(&cudnn_handle);

    // --- Timing Setup ---
    cudaEvent_t epoch_start_event, epoch_stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_stop_event));
    float epoch_gpu_time_ms = 0.0f;

    // --- Memory Sizes ---
    int num_weights = input_size * output_size;
    size_t weights_bytes = sizeof(float) * num_weights;
    size_t max_output_bytes = sizeof(float) * mini_batch_size * output_size;
    size_t batch_loss_bytes = sizeof(float) * mini_batch_size;
    size_t scalar_float_bytes = sizeof(float);
    size_t scalar_int_bytes = sizeof(int);

    // --- Host Variables for Periodic Reporting ---
    float h_epoch_total_loss = 0.0f;
    int h_epoch_total_correct = 0;

    // --- Device Memory Allocation ---
    float *d_weights, *d_grad_weights;
    float *d_logits;
    float *d_probabilities;
    float *d_grad_logits;
    float *d_batch_losses;
    float *d_epoch_total_loss;
    int   *d_epoch_total_correct;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_logits, max_output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_probabilities, max_output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_logits, max_output_bytes));
    // Allocate batch losses only if the kernel uses it (can be NULL otherwise)
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_losses, batch_loss_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_epoch_total_loss, scalar_float_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_epoch_total_correct, scalar_int_bytes));

    // --- Initialize Weights ---
    int init_grid_size = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
    init_weights_uniform<<<init_grid_size, BLOCK_SIZE_1D>>>(d_weights, num_weights, time(0));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // --- Training Loop ---
    int num_batches = (total_train_samples + mini_batch_size - 1) / mini_batch_size;
    printf("Total mini-batches per epoch: %d\n", num_batches);

    dim3 block_1d(BLOCK_SIZE_1D);

    // --- cuBLAS Scalars ---
    const float alpha_gemm = 1.0f;
    const float beta_gemm = 0.0f;
    // --- cuDNN Scalars ---
    const float alpha_softmax = 1.0f;
    const float beta_softmax = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_start_event, 0));

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_loss, 0, scalar_float_bytes, 0));
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_correct, 0, scalar_int_bytes, 0));

        bool should_print_loss = ((epoch + 1) % loss_print_period == 0) || (epoch == num_epochs - 1);
        long long epoch_total_processed = 0;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start_idx = batch_idx * mini_batch_size;
            int current_batch_size = (batch_start_idx + mini_batch_size > total_train_samples) ?
                                      (total_train_samples - batch_start_idx) :
                                      mini_batch_size;
            if (current_batch_size <= 0) continue;

            float* d_current_batch_images = d_all_train_images_float + batch_start_idx * input_size;
            uint8_t* d_current_batch_labels = d_all_train_labels + batch_start_idx;

            cudnnTensorDescriptor_t logits_probs_desc;
            CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&logits_probs_desc));
            CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(logits_probs_desc,
                                                        CUDNN_TENSOR_NCHW,
                                                        CUDNN_DATA_FLOAT,
                                                        current_batch_size,
                                                        output_size,       
                                                        1,                 
                                                        1));               

            // --- Forward Pass ---
            // 1. Calculate Logits
            cublasSgemm(cublas_handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           output_size, current_batch_size, input_size,
                           &alpha_gemm,
                           d_weights, output_size,
                           d_current_batch_images, input_size,
                           &beta_gemm,
                           d_logits, output_size);

            // 2. Calculate Probabilities using cuDNN Softmax
            CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle,
                                  CUDNN_SOFTMAX_ACCURATE,      
                                  CUDNN_SOFTMAX_MODE_INSTANCE,
                                  &alpha_softmax,              
                                  logits_probs_desc,           
                                  d_logits,                    
                                  &beta_softmax,               
                                  logits_probs_desc,           
                                  d_probabilities));           

            if (should_print_loss) {
                 // 3 & 4. Calculate total loss and accuracy using the probabilities
                int accuracy_grid = calculate_grid_size_1d(current_batch_size, BLOCK_SIZE_1D);
                scce_loss_and_accuracy_kernel_accumulate<<<accuracy_grid, block_1d>>>(
                    d_probabilities,
                    d_current_batch_labels,
                    d_batch_losses,
                    d_epoch_total_loss,
                    d_epoch_total_correct,
                    current_batch_size,
                    output_size
                );
                CHECK_CUDA_ERROR(cudaGetLastError());
                epoch_total_processed += current_batch_size;
            }

            // --- Backward Pass ---
            // 5. Calculate Gradient of Loss w.r.t. Logits (dL/dZ = (P - T) / N)
            float grad_scale_factor = 1.0f / (float)current_batch_size;
            int logit_grad_grid = calculate_grid_size_1d(current_batch_size * output_size, BLOCK_SIZE_1D);
            compute_logit_gradient_kernel<<<logit_grad_grid, block_1d>>>(
                d_probabilities,
                d_current_batch_labels,
                d_grad_logits,
                current_batch_size,
                output_size,
                grad_scale_factor);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // 6. Calculate Gradient of Loss w.r.t Weights (dL/dW = dL/dZ * X^T)
            cublasSgemm(cublas_handle,
                          CUBLAS_OP_N, CUBLAS_OP_T,
                          output_size, input_size, current_batch_size,
                          &alpha_gemm,
                          d_grad_logits, output_size,
                          d_current_batch_images, input_size,
                          &beta_gemm,
                          d_grad_weights, output_size);


            // --- Update Weights ---
            // 7. Apply gradient descent step (Weights = Weights - LR * dL/dW)
            int update_grid = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
            update_weights_kernel<<<update_grid, block_1d>>>(
                d_weights,
                d_grad_weights,
                learning_rate,
                num_weights);
            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(logits_probs_desc));
        }

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_stop_event, 0));

        // --- Conditional Loss/Accuracy Reporting ---
        if (should_print_loss) {
            CHECK_CUDA_ERROR(cudaEventSynchronize(epoch_stop_event));

            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_loss, d_epoch_total_loss, scalar_float_bytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_correct, d_epoch_total_correct, scalar_int_bytes, cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaEventElapsedTime(&epoch_gpu_time_ms, epoch_start_event, epoch_stop_event));

            long long total_samples_in_epoch = total_train_samples;

            float average_loss = (total_samples_in_epoch > 0) ? (h_epoch_total_loss / total_samples_in_epoch) : 0.0f;
            float accuracy = (total_samples_in_epoch > 0) ? (float)(h_epoch_total_correct * 100.0 / total_samples_in_epoch) : 0.0f;

            printf("Epoch [%d/%d], Average Loss: %.6f, Accuracy: %.2f%%, Epoch GPU Time: %.2f ms (%.3f s)\n",
                   epoch + 1, num_epochs, average_loss, accuracy, epoch_gpu_time_ms, epoch_gpu_time_ms / 1000.0f);

             if (isnan(average_loss) || isinf(average_loss)) {
                 fprintf(stderr, "Warning: Loss is NaN or Inf. Training might be unstable.\n");
             }
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    } 

    printf("Training complete!\n");

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_stop_event));

    CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle));
    cublasDestroy(cublas_handle);

    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_logits));
    CHECK_CUDA_ERROR(cudaFree(d_probabilities));
    CHECK_CUDA_ERROR(cudaFree(d_batch_losses));
    CHECK_CUDA_ERROR(cudaFree(d_grad_logits));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_correct));

    return true;
}
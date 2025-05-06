#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "../include/train.cuh"
#include "../include/utils.cuh"
#include "../include/ops.cuh"


bool run_training_basic_implementation(
    float *d_all_train_images_float,
    uint8_t *d_all_train_labels,
    int total_train_samples,
    int input_size,
    int output_size,
    int num_epochs,
    int mini_batch_size,
    float learning_rate
) {
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
    size_t output_bytes = sizeof(float) * mini_batch_size * output_size;
    size_t loss_bytes = sizeof(float) * mini_batch_size;
    size_t accuracy_counter_bytes = sizeof(int);

    float *h_losses = (float*)malloc(loss_bytes);
    if (!h_losses) { fprintf(stderr, "Failed to allocate host memory for losses\n"); return false; }
    int h_correct_count = 0;

    // --- Device Memory Allocation ---
    float *d_weights, *d_output, *d_probabilities, *d_losses, *d_grad_logits, *d_grad_weights;
    int *d_correct_count;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_probabilities, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_losses, loss_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_logits, output_bytes));
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
    CHECK_CUDA_ERROR(cudaFree(d_batch_losses));
    CHECK_CUDA_ERROR(cudaFree(d_grad_logits));
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_correct));

    return true;
}


bool run_training_optimized_cudnn_and_graphs(
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
    printf("Starting training (CUDA Graphs - Padding + Separate Graphs)...\n");
    printf("Parameters:\n");
    printf("  Epochs: %d\n", num_epochs);
    printf("  Mini-batch Size: %d\n", mini_batch_size);
    printf("  Learning Rate: %f\n", learning_rate);
    printf("  Input Size: %d\n", input_size);
    printf("  Output Size (Classes): %d\n", output_size);
    printf("  Original Training Samples: %d\n", total_train_samples);
    printf("  Loss Print Period: %d epochs\n", loss_print_period);

    // --- CUDA Stream & Events ---
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t epoch_start_event, epoch_stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&epoch_stop_event));
    float epoch_gpu_time_ms = 0.0f;

    // --- cuBLAS & cuDNN Handles ---
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    CHECK_CUDNN_ERROR(cudnnSetStream(cudnn_handle, stream));

    // --- Padding Calculation and Data Preparation ---
    int padded_total_samples = ((total_train_samples + mini_batch_size - 1) / mini_batch_size) * mini_batch_size;
    int num_padding_samples = padded_total_samples - total_train_samples;
    size_t padded_images_bytes = sizeof(float) * padded_total_samples * input_size;
    size_t padded_labels_bytes = sizeof(uint8_t) * padded_total_samples;

    printf("Padding %d samples to reach %d total samples (multiple of %d).\n",
           num_padding_samples, padded_total_samples, mini_batch_size);

    float* d_padded_images = nullptr;
    uint8_t* d_padded_labels = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_padded_images, padded_images_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_padded_labels, padded_labels_bytes));

    // Copy original data to padded buffers
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_padded_images, d_all_train_images_float, sizeof(float) * total_train_samples * input_size, cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_padded_labels, d_all_train_labels, sizeof(uint8_t) * total_train_samples, cudaMemcpyDeviceToDevice, stream));

    // Zero out padding area
    if (num_padding_samples > 0) {
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_padded_images + total_train_samples * input_size, 0, sizeof(float) * num_padding_samples * input_size, stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_padded_labels + total_train_samples, 0, sizeof(uint8_t) * num_padding_samples, stream));
    }
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // --- Memory Allocation (Graph Buffers & Persistent Weights/Accumulators) ---
    float *d_graph_batch_images = nullptr, *d_logits = nullptr, *d_probabilities = nullptr;
    float *d_grad_logits = nullptr, *d_grad_weights = nullptr, *d_weights = nullptr;
    uint8_t* d_graph_batch_labels = nullptr;
    float *d_batch_losses = nullptr;
    float *d_epoch_total_loss = nullptr;
    int   *d_epoch_total_correct = nullptr;

    size_t batch_images_bytes = sizeof(float) * mini_batch_size * input_size;
    size_t batch_labels_bytes = sizeof(uint8_t) * mini_batch_size;
    size_t max_output_bytes = sizeof(float) * mini_batch_size * output_size;
    size_t batch_loss_bytes = sizeof(float) * mini_batch_size;
    int num_weights = input_size * output_size;
    size_t weights_bytes = sizeof(float) * num_weights;
    size_t scalar_float_bytes = sizeof(float);
    size_t scalar_int_bytes = sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_graph_batch_images, batch_images_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_graph_batch_labels, batch_labels_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_logits, max_output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_probabilities, max_output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_logits, max_output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_losses, batch_loss_bytes));

    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_epoch_total_loss, scalar_float_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_epoch_total_correct, scalar_int_bytes));

    // Host variables for reporting
    float h_epoch_total_loss = 0.0f;
    int h_epoch_total_correct = 0;

    // --- Initialize Weights ---
    int init_grid_size = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);
    init_weights_uniform<<<init_grid_size, BLOCK_SIZE_1D, 0, stream>>>(d_weights, num_weights, time(0));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // --- cuDNN Tensor Descriptor (created once for mini_batch_size) ---
    cudnnTensorDescriptor_t logits_probs_desc;
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&logits_probs_desc));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(logits_probs_desc,
                                             CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                             mini_batch_size,
                                             output_size, 1, 1));

    // --- Graph Capture ---
    printf("Capturing CUDA graphs...\n");
    cudaGraph_t graph_train = nullptr, graph_train_and_report = nullptr;
    cudaGraphExec_t graph_exec_train = nullptr, graph_exec_train_and_report = nullptr;

    const float alpha_gemm = 1.0f;
    const float beta_gemm = 0.0f;
    const float alpha_softmax = 1.0f;
    const float beta_softmax = 0.0f;
    float grad_scale_factor = 1.0f / (float)mini_batch_size;
    dim3 block_1d(BLOCK_SIZE_1D);

    int accuracy_grid = calculate_grid_size_1d(mini_batch_size, BLOCK_SIZE_1D);
    int logit_grad_grid = calculate_grid_size_1d(mini_batch_size * output_size, BLOCK_SIZE_1D);
    int update_grid = calculate_grid_size_1d(num_weights, BLOCK_SIZE_1D);

    // --- Capture Graph 1: Training Only ---
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 1. Forward GEMM (Logits): W * X -> Z
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, mini_batch_size, input_size,
                &alpha_gemm, d_weights, output_size, d_graph_batch_images, input_size,
                &beta_gemm, d_logits, output_size));

    // 2. Softmax: softmax(Z) -> P
    CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha_softmax, logits_probs_desc, d_logits,
                                        &beta_softmax, logits_probs_desc, d_probabilities));

    // 3. Logit Gradient: (P - T) / N -> dL/dZ
    compute_logit_gradient_kernel<<<logit_grad_grid, block_1d, 0, stream>>>(
        d_probabilities, d_graph_batch_labels, d_grad_logits,
        mini_batch_size, output_size, grad_scale_factor);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 4. Backward GEMM (Weight Gradient): dL/dZ * X^T -> dL/dW
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, output_size, input_size, mini_batch_size,
                &alpha_gemm, d_grad_logits, output_size, d_graph_batch_images, input_size,
                &beta_gemm, d_grad_weights, output_size));

    // 5. Weight Update: W = W - lr * dL/dW
    update_weights_kernel<<<update_grid, block_1d, 0, stream>>>(
        d_weights, d_grad_weights, learning_rate, num_weights);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph_train));
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec_train, graph_train, NULL, NULL, 0));
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph_train));
    graph_train = nullptr;

    // --- Capture Graph 2: Training + Reporting ---
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 1. Forward GEMM (Logits): W * X -> Z
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, mini_batch_size, input_size,
                &alpha_gemm, d_weights, output_size, d_graph_batch_images, input_size,
                &beta_gemm, d_logits, output_size));

    // 2. Softmax: softmax(Z) -> P
    CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha_softmax, logits_probs_desc, d_logits,
                                        &beta_softmax, logits_probs_desc, d_probabilities));

    // 3. Loss & Accuracy Accumulation
    scce_loss_and_accuracy_kernel_accumulate<<<accuracy_grid, block_1d, 0, stream>>>(
        d_probabilities, d_graph_batch_labels, d_batch_losses,
        d_epoch_total_loss, d_epoch_total_correct,
        mini_batch_size, output_size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 4. Logit Gradient: (P - T) / N -> dL/dZ
    compute_logit_gradient_kernel<<<logit_grad_grid, block_1d, 0, stream>>>(
        d_probabilities, d_graph_batch_labels, d_grad_logits,
        mini_batch_size, output_size, grad_scale_factor);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 5. Backward GEMM (Weight Gradient): dL/dZ * X^T -> dL/dW
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, output_size, input_size, mini_batch_size,
                &alpha_gemm, d_grad_logits, output_size, d_graph_batch_images, input_size,
                &beta_gemm, d_grad_weights, output_size));

    // 6. Weight Update: W = W - lr * dL/dW
    update_weights_kernel<<<update_grid, block_1d, 0, stream>>>(
        d_weights, d_grad_weights, learning_rate, num_weights);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph_train_and_report));
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec_train_and_report, graph_train_and_report, NULL, NULL, 0));
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph_train_and_report));
    graph_train_and_report = nullptr;

    printf("Graph capture complete.\n");

    // --- Training Loop ---
    int num_batches = padded_total_samples / mini_batch_size;
    printf("Total mini-batches per epoch: %d\n", num_batches);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        CHECK_CUDA_ERROR(cudaEventRecord(epoch_start_event, stream));

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_loss, 0, scalar_float_bytes, stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_epoch_total_correct, 0, scalar_int_bytes, stream));

        bool should_print_loss = ((epoch + 1) % loss_print_period == 0) || (epoch == num_epochs - 1);
        cudaGraphExec_t active_graph_exec = should_print_loss ? graph_exec_train_and_report : graph_exec_train;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start_idx = batch_idx * mini_batch_size;

            float* d_current_padded_batch_images = d_padded_images + batch_start_idx * input_size;
            uint8_t* d_current_padded_batch_labels = d_padded_labels + batch_start_idx;

            // --- Graph Input Update ---
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_graph_batch_images, d_current_padded_batch_images, batch_images_bytes, cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_graph_batch_labels, d_current_padded_batch_labels, batch_labels_bytes, cudaMemcpyDeviceToDevice, stream));

            // --- Graph Launch ---
            CHECK_CUDA_ERROR(cudaGraphLaunch(active_graph_exec, stream));
        }

        CHECK_CUDA_ERROR(cudaEventRecord(epoch_stop_event, stream));

        if (should_print_loss) {
            CHECK_CUDA_ERROR(cudaEventSynchronize(epoch_stop_event));
            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_loss, d_epoch_total_loss, scalar_float_bytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&h_epoch_total_correct, d_epoch_total_correct, scalar_int_bytes, cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaEventElapsedTime(&epoch_gpu_time_ms, epoch_start_event, epoch_stop_event));

            float average_loss = (total_train_samples > 0) ? (h_epoch_total_loss / total_train_samples) : 0.0f;
            float accuracy = (total_train_samples > 0) ? (float)(h_epoch_total_correct * 100.0 / total_train_samples) : 0.0f;

            printf("Epoch [%d/%d], Average Loss: %.6f, Accuracy: %.2f%%, Epoch GPU Time: %.2f ms (%.3f s)\n",
                   epoch + 1, num_epochs, average_loss, accuracy, epoch_gpu_time_ms, epoch_gpu_time_ms / 1000.0f);
             if (isnan(average_loss) || isinf(average_loss)) {
                 fprintf(stderr, "Warning: Loss is NaN or Inf at epoch %d. Training might be unstable.\n", epoch + 1);
             }
        } else {
             CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }

    }

    printf("Training complete!\n");

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(epoch_stop_event));

    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(logits_probs_desc));
    CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));

    if (graph_exec_train) CHECK_CUDA_ERROR(cudaGraphExecDestroy(graph_exec_train));
    if (graph_exec_train_and_report) CHECK_CUDA_ERROR(cudaGraphExecDestroy(graph_exec_train_and_report));

    CHECK_CUDA_ERROR(cudaFree(d_padded_images));
    CHECK_CUDA_ERROR(cudaFree(d_padded_labels));
    CHECK_CUDA_ERROR(cudaFree(d_graph_batch_images));
    CHECK_CUDA_ERROR(cudaFree(d_graph_batch_labels));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_grad_weights));
    CHECK_CUDA_ERROR(cudaFree(d_logits));
    CHECK_CUDA_ERROR(cudaFree(d_probabilities));
    CHECK_CUDA_ERROR(cudaFree(d_batch_losses));
    CHECK_CUDA_ERROR(cudaFree(d_grad_logits));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_epoch_total_correct));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    printf("Cleanup complete.\n");
    return true;
}
#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>

static void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

#define CHECK_CUDNN_ERROR(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error: %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, __LINE__); \
        /* Consider adding exit(EXIT_FAILURE); */ \
    }

// Helper function to calculate grid size for 1D kernels
inline int calculate_grid_size_1d(int num_elements, int block_size) {
    return (num_elements + block_size - 1) / block_size;
}

// Helper function to calculate grid size for 2D kernels
inline dim3 calculate_grid_size_2d(int dim_x, int dim_y, const dim3& block_size) {
    return dim3(
        (dim_x + block_size.x - 1) / block_size.x,
        (dim_y + block_size.y - 1) / block_size.y
    );
}


#endif
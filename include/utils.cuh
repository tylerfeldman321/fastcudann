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

#endif
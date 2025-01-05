#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include "cuda_runtime.h"
#include <iostream>


namespace cuda_o {

    #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
    void inline check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }
}
#endif
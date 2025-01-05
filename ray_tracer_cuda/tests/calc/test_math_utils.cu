#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../calc/math_utils.cuh"

__global__ void test_degrees_to_radians() {
    float degrees = 90.0;
    float expected = 3.1415926535897932385 / 2; // pi/2
    float result = calc::degrees_to_radians(degrees);
    assert(abs(result - expected) < 1e-5);
}

__global__ void test_clamp() {
    float result = calc::clamp(5.0, 1.0, 10.0);
    assert(result == 5.0);

    result = calc::clamp(-1.0, 1.0, 10.0);
    assert(result == 1.0);

    result = calc::clamp(15.0, 1.0, 10.0);
    assert(result == 10.0);
}

__global__ void test_random_float() {
    uint32_t state = 42;
    float result = calc::random_float(state);
    assert(result >= 0.0f && result <= 1.0f);
}

void run_math_util_tests() {
    test_degrees_to_radians<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_clamp<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_random_float<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "math_util tests passed!" << std::endl;
}

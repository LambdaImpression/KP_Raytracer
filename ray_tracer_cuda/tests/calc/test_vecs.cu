#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../calc/vecs.cuh"

// Test the default constructor and basic accessor functions
__global__ void test_vecs_constructors_and_accessors() {
    calc::vecs v1;
    assert(v1.x() == 0.0f && v1.y() == 0.0f && v1.z() == 0.0f);

    calc::vecs v2(1.0f, 2.0f, 3.0f);
    assert(v2.x() == 1.0f && v2.y() == 2.0f && v2.z() == 3.0f);

    v2[0] = 4.0f;
    v2[1] = 5.0f;
    v2[2] = 6.0f;
    assert(v2[0] == 4.0f && v2[1] == 5.0f && v2[2] == 6.0f);
}

// Test vector addition
__global__ void test_vecs_addition() {
    calc::vecs v1(1.0f, 2.0f, 3.0f);
    calc::vecs v2(4.0f, 5.0f, 6.0f);
    calc::vecs v3 = v1 + v2;

    assert(v3.x() == 5.0f && v3.y() == 7.0f && v3.z() == 9.0f);
}

// Test vector subtraction
__global__ void test_vecs_subtraction() {
    calc::vecs v1(4.0f, 5.0f, 6.0f);
    calc::vecs v2(1.0f, 2.0f, 3.0f);
    calc::vecs v3 = v1 - v2;

    assert(v3.x() == 3.0f && v3.y() == 3.0f && v3.z() == 3.0f);
}

// Test dot product
__global__ void test_vecs_dot_product() {
    calc::vecs v1(1.0f, 2.0f, 3.0f);
    calc::vecs v2(4.0f, -5.0f, 6.0f);
    float dot_product = calc::dot(v1, v2);

    assert(dot_product == (1.0f * 4.0f + 2.0f * -5.0f + 3.0f * 6.0f));
}

// Test cross product
__global__ void test_vecs_cross_product() {
    calc::vecs v1(1.0f, 0.0f, 0.0f);
    calc::vecs v2(0.0f, 1.0f, 0.0f);
    calc::vecs v3 = calc::cross(v1, v2);

    assert(v3.x() == 0.0f && v3.y() == 0.0f && v3.z() == 1.0f);
}

// Test normalization
__global__ void test_vecs_normalization() {
    calc::vecs v1(3.0f, 0.0f, 4.0f);
    calc::vecs v2 = calc::normalized(v1);

    assert(abs(v2.length() - 1.0f) < 1e-5);
}

// Test random unit vector
__global__ void test_vecs_random_unit_vector(curandState* state) {
    curand_init(1234, 0, 0, state);
    calc::vecs v = calc::random_unit_vector(state);
    assert(abs(v.length() - 1.0f) < 1e-5);
}

// Test random unit sphere
__global__ void test_vecs_random_unit_sphere(curandState* state) {
    curand_init(1234, 0, 0, state);
    calc::vecs v = calc::random_unit_sphere(state);
    assert(v.length_squared() < 1.0f);
}

// Run all tests
void run_vecs_tests() {
    // Initialize random state for tests requiring randomness
    curandState* dev_rand_state;
    cudaMalloc(&dev_rand_state, sizeof(curandState));

    test_vecs_constructors_and_accessors<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_addition<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_subtraction<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_dot_product<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_cross_product<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_normalization<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_vecs_random_unit_vector<<<1, 1>>>(dev_rand_state);
    cudaDeviceSynchronize();

    test_vecs_random_unit_sphere<<<1, 1>>>(dev_rand_state);
    cudaDeviceSynchronize();

    cudaFree(dev_rand_state);

    std::cout << "Vecs tests passed!" << std::endl;
}

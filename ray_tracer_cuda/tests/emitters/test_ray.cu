#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../emitters/ray.cuh"
#include "../../calc/vecs.cuh"
#include "../../calc/point.cuh"

// Test the ray constructor and accessors
__global__ void test_ray_constructor_and_accessors() {
    calc::point origin(1.0f, 2.0f, 3.0f);
    calc::vecs direction(4.0f, 5.0f, 6.0f);
    emit::ray r(origin, direction);

    // Validate origin
    assert(r.origin().x() == 1.0f && r.origin().y() == 2.0f && r.origin().z() == 3.0f);

    // Validate direction
    assert(r.direction().x() == 4.0f && r.direction().y() == 5.0f && r.direction().z() == 6.0f);
}

// Test the `at` method of the ray
__global__ void test_ray_at() {
    calc::point origin(0.0f, 0.0f, 0.0f);
    calc::vecs direction(1.0f, 1.0f, 1.0f);
    emit::ray r(origin, direction);

    calc::point point_at_2 = r.at(2.0f);
    assert(point_at_2.x() == 2.0f && point_at_2.y() == 2.0f && point_at_2.z() == 2.0f);

    calc::point point_at_minus1 = r.at(-1.0f);
    assert(point_at_minus1.x() == -1.0f && point_at_minus1.y() == -1.0f && point_at_minus1.z() == -1.0f);
}

// Run all ray tests
void run_ray_tests() {
    test_ray_constructor_and_accessors<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_ray_at<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "ray tests passed!" << std::endl;
}

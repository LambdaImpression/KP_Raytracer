#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../emitters/camera.cuh"
#include "../../calc/vecs.cuh"
#include "../../calc/point.cuh"

// Test get_ray method
__global__ void test_camera_get_ray() {
    calc::point lookfrom(0.0f, 0.0f, 0.0f);
    calc::point lookat(0.0f, 0.0f, -1.0f);
    calc::vecs vup(0.0f, 1.0f, 0.0f);
    float vfov = 90.0f;
    float aspect_ratio = 16.0f / 9.0f;
    float aperture = 0.1f;
    float focus_dist = 1.0f;

    emit::camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);

    // Initialize random state
    curandState rand_state;
    curand_init(1337, 0, 0, &rand_state);

    emit::ray r = cam.get_ray(0.5f, 0.5f, &rand_state);
    calc::point origin = r.origin();
    calc::vecs direction = r.direction();

    // Validate ray origin
    assert(origin.x() >= -0.05f && origin.x() <= 0.05f); // Origin should be near the camera
    assert(origin.y() >= -0.05f && origin.y() <= 0.05f);
    assert(origin.z() == 0.0f);

    // Validate ray direction
    assert(direction.z() < 0.0f); // Direction should point "into" the scene
}

// Run all camera tests
void run_camera_tests() {

    test_camera_get_ray<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "camera tests passed!" << std::endl;
}

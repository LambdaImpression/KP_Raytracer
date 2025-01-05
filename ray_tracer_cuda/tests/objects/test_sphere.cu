#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../objects/sphere.cuh"
#include "../../emitters/ray.cuh"
#include "../../canvas/color.cuh"
#include "../../calc/vecs.cuh"
#include "../../material/materials.cuh"

// Test sphere hit with a ray
__global__ void test_sphere_hit() {
    calc::point sphere_center(0.0f, 0.0f, -1.0f);
    float sphere_radius = 0.5f;
    mat::lambertian material(canvas::color(0.8f, 0.3f, 0.3f));

    objects::sphere sphere(sphere_center, sphere_radius, &material);

    emit::ray test_ray(calc::point(0.0f, 0.0f, 0.0f), calc::vecs(0.0f, 0.0f, -1.0f));
    canvas::hit_record record;

    bool result = sphere.hit(test_ray, 0.001f, FLT_MAX, record);

    assert(result); // Should hit the sphere
    assert(abs(record.t - 0.5f) < 1e-5); // Hit at t = 0.5
    assert(record.hit_point.x() == 0.0f && record.hit_point.y() == 0.0f && record.hit_point.z() == -0.5f);
    assert(record.normal.x() == 0.0f && record.normal.y() == 0.0f && record.normal.z() == 1.0f); // Normal points outward
}

// Test sphere miss with a ray
__global__ void test_sphere_miss() {
    calc::point sphere_center(0.0f, 0.0f, -1.0f);
    float sphere_radius = 0.5f;
    mat::lambertian material(canvas::color(0.8f, 0.3f, 0.3f));

    objects::sphere sphere(sphere_center, sphere_radius, &material);

    emit::ray test_ray(calc::point(0.0f, 0.0f, 0.0f), calc::vecs(0.0f, 1.0f, 0.0f)); // Ray points away
    canvas::hit_record record;

    bool result = sphere.hit(test_ray, 0.001f, FLT_MAX, record);

    assert(!result); // Should not hit the sphere
}

// Test sphere hit with multiple roots
__global__ void test_sphere_multiple_roots() {
    calc::point sphere_center(0.0f, 0.0f, -1.0f);
    float sphere_radius = 0.5f;
    mat::lambertian material(canvas::color(0.8f, 0.3f, 0.3f));

    objects::sphere sphere(sphere_center, sphere_radius, &material);

    emit::ray test_ray(calc::point(
        0.0f, 0.0f, -2.0f),
        calc::vecs(0.0f, 0.0f, 1.0f)); // Ray enters and exits the sphere
    canvas::hit_record record;

    bool result = sphere.hit(test_ray, 0.001f, FLT_MAX, record);

    assert(result); // Should hit the sphere
}

// Run all sphere tests
void run_sphere_tests() {
    test_sphere_hit<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_sphere_miss<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_sphere_multiple_roots<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "sphere tests passed!" << std::endl;
}

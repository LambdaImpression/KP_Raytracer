#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../material/materials.cuh"
#include "../../emitters/ray.cuh"
#include "../../canvas/color.cuh"
#include "../../canvas/intersectable.cuh"
#include "../../calc/vecs.cuh"

// Mock hit_record for testing
__device__ canvas::hit_record mock_hit_record(const calc::vecs& normal, const calc::point& hit_point) {
    canvas::hit_record rec;
    rec.normal = normal;
    rec.hit_point = hit_point;
    rec.front_face = true; // Assume front face for simplicity
    return rec;
}

// Test lambertian scatter
__global__ void test_lambertian_scatter() {
    canvas::color albedo(0.8f, 0.3f, 0.3f);
    mat::lambertian lambertian_material(albedo);

    emit::ray incoming_ray(calc::point(0.0f, 0.0f, 0.0f), calc::vecs(1.0f, 1.0f, 1.0f));
    canvas::hit_record rec = mock_hit_record(calc::vecs(0.0f, 1.0f, 0.0f), calc::point(0.0f, 0.0f, 0.0f));
    emit::ray scattered;
    canvas::color attenuation;

    curandState rand_state;
    curand_init(1337, 0, 0, &rand_state);

    bool scatter_result = lambertian_material.scatter(incoming_ray, rec, attenuation, scattered, &rand_state);
    assert(scatter_result);
}

// Test metal scatter
__global__ void test_metal_scatter() {
    calc::vecs albedo(0.8f, 0.8f, 0.8f);
    float fuzz = 0.5f;
    mat::metal metal_material(albedo, fuzz);

    emit::ray incoming_ray(calc::point(0.0f, 0.0f, 0.0f), calc::vecs(1.0f, 1.0f, 1.0f));
    canvas::hit_record rec = mock_hit_record(calc::vecs(0.0f, 1.0f, 0.0f), calc::point(0.0f, 0.0f, 0.0f));
    emit::ray scattered;
    canvas::color attenuation;

    curandState rand_state;
    curand_init(1337, 0, 0, &rand_state);

    bool scatter_result = metal_material.scatter(incoming_ray, rec, attenuation, scattered, &rand_state);
    assert(scatter_result == false);
    assert(dot(scattered.direction(), rec.normal) < 0); // Scattered ray should be above the surface
}

// Test dielectric scatter
__global__ void test_dielectric_scatter() {
    float refractive_index = 1.5f;
    mat::dielectric dielectric_material(refractive_index);

    emit::ray incoming_ray(calc::point(0.0f, 0.0f, 0.0f), calc::vecs(0.0f, 0.0f, -1.0f));
    canvas::hit_record rec = mock_hit_record(calc::vecs(0.0f, 0.0f, 1.0f), calc::point(0.0f, 0.0f, 0.0f));
    emit::ray scattered;
    canvas::color attenuation;

    curandState rand_state;
    curand_init(1337, 0, 0, &rand_state);

    bool scatter_result = dielectric_material.scatter(incoming_ray, rec, attenuation, scattered, &rand_state);
    assert(scatter_result);
    // Scattered ray direction should be calculated (reflection or refraction)
    assert(scattered.direction().length() > 0);
}

// Run all material tests
void run_material_tests() {
    test_lambertian_scatter<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_metal_scatter<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_dielectric_scatter<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "material tests passed!" << std::endl;
}

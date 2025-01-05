#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../canvas/intersectable.cuh"
#include "../../calc/point.cuh"
#include "../../emitters/ray.cuh"
#include "../../material/materials.cuh"

// Mock material for testing
class mock_material : public mat::material {
    // Empty implementation for testing purposes
};

// Mock intersectable for testing
class mock_intersectable : public canvas::intersectable {
public:
    __device__ mock_intersectable(bool should_hit, float t_value, const calc::vecs& normal)
        : should_hit(should_hit), t_value(t_value), normal(normal) {}

    __device__ virtual bool hit(const emit::ray& ray, float t_min, float t_max, canvas::hit_record& rec) const override {
        if (should_hit && t_value >= t_min && t_value <= t_max) {
            rec.t = t_value;
            rec.hit_point = ray.origin() + t_value * ray.direction();
            rec.face_normal(ray, normal);
            return true;
        }
        return false;
    }

private:
    bool should_hit;
    float t_value;
    calc::vecs normal;
};

// Test hit_record's face normal calculation
__global__ void test_hit_record_face_normal() {
    canvas::hit_record record;
    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(1, 0, 0));
    calc::vecs outward_normal(1.0f, 0.0f, 0.0f);

    record.face_normal(test_ray, outward_normal);
    assert(!record.front_face); // Ray and normal are in the same direction
    assert(record.normal.x() == -1.0f && record.normal.y() == 0.0f && record.normal.z() == 0.0f);

    outward_normal = calc::vecs(-1.0f, 0.0f, 0.0f);
    record.face_normal(test_ray, outward_normal);
    assert(record.front_face); // Ray and normal are in opposite directions
    assert(record.normal.x() == -1.0f && record.normal.y() == 0.0f && record.normal.z() == 0.0f);
}

// Test mock intersectable hit
__global__ void test_mock_intersectable_hit() {
    calc::vecs normal(0.0f, 1.0f, 0.0f);
    mock_intersectable obj(true, 5.0f, normal);

    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(0, 1, 0));
    canvas::hit_record record;

    bool result = obj.hit(test_ray, 0.0f, 10.0f, record);
    assert(result);          // Should hit
    assert(record.t == 5.0f); // t-value should be 5.0
    assert(record.hit_point.x() == 0.0f && record.hit_point.y() == 5.0f && record.hit_point.z() == 0.0f);
}

// Test mock intersectable miss
__global__ void test_mock_intersectable_miss() {
    calc::vecs normal(0.0f, 1.0f, 0.0f);
    mock_intersectable obj(false, 5.0f, normal);

    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(0, 1, 0));
    canvas::hit_record record;

    bool result = obj.hit(test_ray, 0.0f, 10.0f, record);
    assert(!result); // Should not hit
}

// Run all intersectable tests
void run_intersectable_tests() {
    test_hit_record_face_normal<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_mock_intersectable_hit<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_mock_intersectable_miss<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "intersectable tests passed!" << std::endl;
}

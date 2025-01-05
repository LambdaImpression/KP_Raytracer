#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "../../canvas/hitlist.cuh"
#include "../../emitters/ray.cuh"
#include "../../canvas/intersectable.cuh"

// Mock intersectable for testing
class mock_intersectable : public canvas::intersectable {
public:
    __device__ mock_intersectable(bool should_hit, float t_value)
        : should_hit(should_hit), t_value(t_value) {}

    __device__ virtual bool hit(const emit::ray& ray, float t_min, float t_max, canvas::hit_record& record) const override {
        if (should_hit && t_value >= t_min && t_value <= t_max) {
            record.t = t_value;
            return true;
        }
        return false;
    }

private:
    bool should_hit;
    float t_value;
};

// Test hitlist with no objects
__global__ void test_hitlist_empty() {
    canvas::hitlist list(nullptr, 0);
    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(1, 0, 0));
    canvas::hit_record record;

    bool result = list.hit(test_ray, 0.0f, 100.0f, record);
    assert(!result); // Should not hit anything
}

// Test hitlist with a single object
__global__ void test_hitlist_single_hit() {
    mock_intersectable obj(true, 10.0f);
    canvas::intersectable* objects[] = { &obj };
    canvas::hitlist list(objects, 1);

    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(1, 0, 0));
    canvas::hit_record record;

    bool result = list.hit(test_ray, 0.0f, 100.0f, record);
    assert(result);          // Should hit
    assert(record.t == 10.0f); // Closest hit at t = 10.0
}

// Test hitlist with multiple objects
__global__ void test_hitlist_multiple_hits() {
    mock_intersectable obj1(true, 10.0f);
    mock_intersectable obj2(true, 5.0f); // Closer hit
    mock_intersectable obj3(false, 20.0f); // Won't hit

    canvas::intersectable* objects[] = { &obj1, &obj2, &obj3 };
    canvas::hitlist list(objects, 3);

    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(1, 0, 0));
    canvas::hit_record record;

    bool result = list.hit(test_ray, 0.0f, 100.0f, record);
    assert(result);          // Should hit
    assert(record.t == 5.0f); // Closest hit at t = 5.0
}

// Test hitlist when no objects hit
__global__ void test_hitlist_no_hit() {
    mock_intersectable obj1(false, 10.0f);
    mock_intersectable obj2(false, 5.0f);
    canvas::intersectable* objects[] = { &obj1, &obj2 };
    canvas::hitlist list(objects, 2);

    emit::ray test_ray(calc::point(0, 0, 0), calc::vecs(1, 0, 0));
    canvas::hit_record record;

    bool result = list.hit(test_ray, 0.0f, 100.0f, record);
    assert(!result); // Should not hit anything
}

// Run all hitlist tests
void run_hitlist_tests() {
    test_hitlist_empty<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_hitlist_single_hit<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_hitlist_multiple_hits<<<1, 1>>>();
    cudaDeviceSynchronize();

    test_hitlist_no_hit<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "hitlist tests passed!" << std::endl;
}

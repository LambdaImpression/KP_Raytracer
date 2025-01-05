#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "../canvas/intersectable.cuh"
#include "../calc/vecs.cuh"
#include "cuda_runtime.h"
#include <curand_kernel.h>

namespace objects {

    class sphere : public canvas::intersectable {
    public:
        __device__ sphere(calc::point cen, float r, mat::material* m) : center(cen), radius(r), mat_ptr(m) {};
        __device__ virtual bool hit(const emit::ray& r, float t_min, float t_max, canvas::hit_record& record) const;

        calc::point center;
        float radius;
        mat::material* mat_ptr;

    };

    __device__ inline bool sphere::hit(const emit::ray& r, float t_min, float t_max, canvas::hit_record& record) const {
        calc::vecs origin_center = r.origin() - center;
        float a = r.direction().length_squared();
        float half_b = dot(origin_center, r.direction());
        float c = origin_center.length_squared() - radius * radius;
        float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) {
            return false;
        }

        float sqrt_discriminant = sqrt(discriminant);

        // Find nearest root in [t_min, t_max]
        float root = (-half_b - sqrt_discriminant) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_discriminant) / a;
            if (root < t_min || root > t_max) {
                return false;
            }
        }

        record.t = root;
        record.hit_point = r.at(record.t);
        calc::vecs outward_normal = (record.hit_point - center) / radius;
        record.face_normal(r, outward_normal);
        record.mat_ptr = mat_ptr;
        return true;
    }
}

#endif
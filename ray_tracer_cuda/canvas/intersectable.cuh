#ifndef INTERSECTABLE_H
#define INTERSECTABLE_H

#include "../emitters/ray.cuh"
#include "../calc/point.cuh"

namespace mat {
    class material;
}

namespace canvas {

    struct hit_record {
        calc::point hit_point;
        calc::vecs normal;
        float t = 0;
        bool front_face = false;
        mat::material* mat_ptr;

        __device__ void face_normal(const emit::ray& ray, const calc::vecs& outward_normal) {
            front_face = dot(ray.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
    };

    class intersectable {
    public:
        __device__ virtual bool hit(const emit::ray& ray, float t_min, float t_max, hit_record& rec) const = 0;
    };

}
#endif
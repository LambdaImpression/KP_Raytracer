#ifndef HITLIST_CUH
#define HITLIST_CUH

#include "cuda_runtime.h"
#include "./intersectable.cuh"
#include "../emitters/ray.cuh"

namespace canvas {

    class hitlist : public intersectable {

    public:
        __device__ hitlist() {};
        __device__ hitlist(intersectable** l, int n) { list = l; list_size = n; };
        __device__ virtual bool hit(const emit::ray& ray, float t_min, float t_max, hit_record& record) const;

        intersectable** list;
        int list_size;
    };

    __device__ inline bool hitlist::hit(const emit::ray& ray, float t_min, float t_max, hit_record& record) const {
        hit_record temp_record;
        bool hit_anything = false;
        float closest = t_max;

        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(ray, t_min, closest, temp_record)) {
                hit_anything = true;
                closest = temp_record.t;
                record = temp_record;
            }
        }
        return hit_anything;
    }
}
#endif
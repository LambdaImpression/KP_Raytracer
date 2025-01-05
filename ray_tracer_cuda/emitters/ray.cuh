#ifndef RAY_CUH
#define RAY_CUH

#include "../calc/vecs.cuh"
#include "../calc/point.cuh"

namespace emit {

    class ray {
    public:
        ray() = default;
        __device__ ray(const calc::point& origin, const calc::vecs& direction) : orig(origin), dir(direction) {}

        __device__ calc::point origin() const { return orig; };
        __device__ calc::vecs direction() const { return dir; };
        __device__ calc::point at(float t) const { return orig + t * dir; }

    public:
        calc::point orig;
        calc::vecs dir;

    };

}
#endif
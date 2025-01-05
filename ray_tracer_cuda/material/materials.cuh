#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "cuda_runtime.h"
#include "../emitters/ray.cuh"
#include "../canvas/color.cuh"
#include "../canvas/intersectable.cuh"

namespace mat{

    class material {
    public:
        __device__ virtual bool scatter(
            const emit::ray &r_in,
            canvas::hit_record &rec,
            canvas::color &attenuation,
            emit::ray &scattered,
            curandState *local_random_state) const = 0;

        __device__ bool refract(const calc::vecs &v, const calc::vecs &n, float ni_over_nt, calc::vecs &refracted) const {
            calc::vecs uv = calc::unit_vector(v);
            float dt = calc::dot(uv, n);
            float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
            if (discriminant > 0) {
                refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
                return true;
            } else {
                return false;
            }
        }
  
        __device__ float schlick(float cosine, float ref_idx) const {
            float r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1 - r0)*pow((1 - cosine), 5);
        }

        __device__ calc::vecs reflect(const calc::vecs& v, const calc::vecs &n) const {
            return v - 2 * dot(v, n) * n;
        }

    };

    class lambertian : public material {
    public:
        __device__ lambertian(const canvas::color&a) : albedo(a) {}
        __device__ virtual bool scatter(
            const emit::ray &r_in,
            canvas::hit_record &rec,
            canvas::color &attenuation,
            emit::ray &scattered,
            curandState *local_random_state) const override {
            calc::vecs scatter_direction = rec.normal + calc::random_unit_vector(local_random_state);

            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = emit::ray(rec.hit_point, scatter_direction);
            attenuation = albedo;
            return true;
        }

        canvas::color albedo;
    };

    class dielectric : public material {
    public:
        __device__ dielectric(float ri) : ref_idx(ri) {}
        __device__ virtual bool scatter(
            const emit::ray &r_in,
            canvas::hit_record &rec,
            canvas::color &attenuation,
            emit::ray &scattered,
            curandState *local_random_state) const override {
            calc::vecs outward_normal;
            calc::vecs reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = canvas::color(1.0, 1.0, 1.0);
            calc::vecs refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }
            else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }

            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
                reflect_prob = schlick(cosine, ref_idx);
            }
            // total internal reflection
            else {
                scattered = emit::ray(rec.hit_point, reflected);
                reflect_prob = 1.0;
            }

            // randomly choosing between reflection and refraction
            if (curand_uniform(local_random_state) < reflect_prob) {
                scattered = emit::ray(rec.hit_point, reflected);
            }
            else {
                scattered = emit::ray(rec.hit_point, refracted);
            }

            return true;
        }
        float ref_idx;
    };

    class metal : public material {
    public:
        __device__ metal(const calc::vecs &a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(
            const emit::ray &r_in,
            canvas::hit_record &rec,
            canvas::color &attenuation,
            emit::ray &scattered,
            curandState *local_random_state) const override{
            calc::vecs reflected = reflect(calc::unit_vector(r_in.direction()), rec.normal);
            scattered = emit::ray(
                rec.hit_point,
                reflected + fuzz* calc::random_unit_sphere(local_random_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

        canvas::color albedo;
        float fuzz;
    };
}

#endif //MATERIAL_CUH
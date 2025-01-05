#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "../calc/point.cuh"
#include "../calc/math_utils.cuh"
#include "../calc/vecs.cuh"
#include "ray.cuh"
#include <curand_kernel.h>

namespace emit {

	class camera {
	public:
		__host__ __device__ camera(
			calc::point lookfrom,
			calc::point lookat,
			calc::vecs vup,
			float vfov,
			float aspect_ratio,
			float aperture,
			float focus_dist
		) {
			float theta = calc::degrees_to_radians(vfov);
			float h = tan(theta / 2);
			float viewport_height = 2.0 * h;
			float viewport_width = aspect_ratio * viewport_height;

			w = unit_vector(lookfrom - lookat); // "z"
			u = unit_vector(cross(vup, w)); // "x"
			v = cross(w, u); // "y"

			origin = lookfrom;
			horizontal = focus_dist * viewport_width * u;
			vertical = focus_dist * viewport_height * v;
			lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist*w;

			lens_radius = aperture / 2;
		}

		__device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
			calc::vecs rd = lens_radius * calc::random_unit_disk(local_rand_state);
			calc::vecs offset = u * rd.x() + v * rd.y();
			return ray(origin + offset,  lower_left_corner + s * horizontal + t * vertical - origin - offset);
		}

	private:
		calc::point origin;
		calc::point lower_left_corner;
		calc::vecs horizontal;
		calc::vecs vertical;
		calc::vecs u, v, w;
		float lens_radius;
	};
}

#endif
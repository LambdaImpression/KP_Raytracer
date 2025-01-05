#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

namespace calc {

    __host__ __device__ inline float degrees_to_radians(float degrees) {
        return degrees * 3.1415926535897932385 / 180.0;
    }

    __host__ __device__ inline float clamp(float x, float min, float max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

	__host__ __device__ inline float random_float(uint32_t& state) {
			
		// PCG Hash
		state = state * 747796495u + 2891336453u;
		state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		state = (state >> 22u) ^ state;

		// scale the output between 0 and 1
		float scale = static_cast<float>(0xffffffffu);
		return static_cast<float>(state) / scale;
	}
}
#endif
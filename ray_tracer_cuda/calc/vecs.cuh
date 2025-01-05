#ifndef VECS_CUH
#define VECS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ostream>

namespace calc {
	class vecs {
	public:
		__host__ __device__ vecs() : values{ 0.0f, 0.0f, 0.0f } {}
		__host__ __device__ vecs(float x, float y, float z) : values{ x, y, z } {}

		__host__ __device__ float x() const { return values[0]; }
		__host__ __device__ float y() const { return values[1]; }
		__host__ __device__ float z() const { return values[2]; }
		__host__ __device__ float operator[](int i) const { return values[i]; }
		__host__ __device__ float& operator[](int i) { return values[i]; }

		__host__ __device__ vecs operator-() const
		{
			return vecs(-values[0], -values[1], -values[2]);
		}

		__host__ __device__ vecs& operator+=(const vecs& other)
		{
			values[0] += other.values[0];
			values[1] += other.values[1];
			values[2] += other.values[2];
			return *this;
		}

		__host__ __device__ vecs& operator-=(const vecs& other)
		{
			values[0] -= other.values[0];
			values[1] -= other.values[1];
			values[2] -= other.values[2];
			return *this;
		}

		__host__ __device__ vecs& operator*=(float t)
		{
			values[0] *= t;
			values[1] *= t;
			values[2] *= t;
			return *this;
		}

		__host__ __device__ vecs& operator/=(float t)
		{
			float d = 1.0f / t;
			values[0] *= d;
			values[1] *= d;
			values[2] *= d;
			return *this;
		}

		__host__ __device__ vecs& operator*=(const vecs& other)
		{
			values[0] *= other.values[0];
			values[1] *= other.values[1];
			values[2] *= other.values[2];
			return *this;
		}

		__host__ __device__ inline float length_squared() const
		{
			return (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
		}

		__host__ __device__ float length() const
		{
			return std::sqrt(length_squared());
		}

		__host__ __device__ vecs& normalize()
		{
			*this *= (1.0f / std::sqrt(length_squared()));
			return *this;
		}

		__device__ static vecs random(curandState *local_rand_state) {
			return vecs(
				curand_uniform(local_rand_state),
				curand_uniform(local_rand_state),
				curand_uniform(local_rand_state));
		}

		__device__ bool near_zero() const {
			const float eps = 1e-8;
			return (fabsf(values[0] < eps) && fabsf(values[1] < eps) && fabsf(values[2] < eps));
		}


		__host__ __device__ friend vecs operator-(const vecs& a, const vecs& b);

		__host__ __device__ friend float dot(const vecs& a, const vecs& b);

		__host__ __device__ friend vecs cross(const vecs& a, const vecs& b);

		__host__ __device__ friend vecs at(const vecs& a, const vecs& b, float t);

		__host__ __device__ float dot(vecs b) const {
			return (values[0] * b.values[0] + values[1] * b.values[1] + values[2] * b.values[2]);
		}

		float values[3];
	};

	inline std::ostream& operator<<(std::ostream& ostr, const vecs& v)
	{
		ostr << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
		return ostr;
	}

	__host__ __device__ inline vecs operator+(const vecs& a, const vecs& b)
	{
		vecs result{ a };
		return result += b;
	}

	__host__ __device__ inline vecs operator-(const vecs& a, const vecs& b)
	{
		vecs result{ a };
		return result -= b;

	}

	__host__ __device__  inline vecs operator*(const vecs &u, const vecs &v) {
		return vecs(u.values[0] * v.values[0], u.values[1] * v.values[1], u.values[2] * v.values[2]);
	}

	__host__ __device__  inline vecs operator*(float t, const vecs &v) {
		return vecs(t*v.values[0], t*v.values[1], t*v.values[2]);
	}

	__host__ __device__  inline vecs operator*(const vecs &v, float t) {
		return t * v;
	}

	__host__ __device__ inline vecs operator/(vecs v, float t) {
		return (1/t) * v;
	}

	__host__ __device__ inline vecs normalized(const vecs& v)
	{
		vecs vec{ v };
		return vec *= (1.0f / v.length());
	}

	__host__ __device__ inline float dot(const vecs& a, const vecs& b)
	{
		return (a.values[0] * b.values[0] + a.values[1] * b.values[1] + a.values[2] * b.values[2]);
	}

	__host__ __device__ inline vecs cross(const vecs& a, const vecs& b)
	{
		return vecs(
			a.values[1] * b.values[2] - a.values[2] * b.values[1],
			a.values[2] * b.values[0] - a.values[0] * b.values[2],
			a.values[0] * b.values[1] - a.values[1] * b.values[0]
		);
	}

	__host__ __device__ inline vecs unit_vector(vecs v) {
		return v / v.length();
	}

	__device__ inline vecs random_unit_sphere(curandState* local_rand_state) {
		vecs p;
		while (true) {
			p = 2.0f * vecs::random(local_rand_state) - vecs(1.0f, 1.0f, 1.0f);
			if (p.length_squared() >= 1.0f) continue;
			return p;
		};
	}

	__device__ inline vecs random_unit_vector(curandState* local_rand_state) {
		return unit_vector(random_unit_sphere(local_rand_state));
	}

	__device__ inline vecs random_unit_disk(curandState* local_rand_state) {
		while (true) {
			auto p = vecs(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0);
			if (p.length_squared() >= 1) continue;
			return p;
		}
	}
}

#endif
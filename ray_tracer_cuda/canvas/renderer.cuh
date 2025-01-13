#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <curand_kernel.h>
#include "intersectable.cuh"
#include "../material/materials.cuh"
#include "../objects/sphere.cuh"
#include "../canvas/hitlist.cuh"
#include "../include/json.hpp"
#include "../sphere_config.cuh"

namespace canvas {


    __device__ color ray_color(
        const emit::ray& r,
        intersectable **world,
        curandState *local_rand_state,
        int max_depth) {
        color white = color(1.0, 1.0, 1.0);
        color blue = color(0.5, 0.7, 1.0);
        color red = color(1.0, 0.0, 0.0);

        emit::ray cur_ray = r;
        color cur_attenuation = color(1.0, 1.0, 1.0);
        for(int i = 0; i < max_depth; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                emit::ray scattered;
                color attenuation;
                if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                } else {
                    return color(0.0,0.0,0.0);
                }

            } else {
                calc::vecs unit_direction = calc::unit_vector(r.direction());
                float t = 0.5f*(unit_direction.y() + 1.0f);
                calc::vecs c = (1.0f-t)*white + t*blue;
                return cur_attenuation *= c;
            }
        }
        return color(0.0,0.0,0.0); // recursion exceeded
    }

    __global__ void render_init(int max_column, int max_row, curandState *rand_state) {
        int column = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;

        if ((column >= max_column) || (row >= max_row)) {
            // Pixel outside image
            return;
        }

        int pixel_index = row * max_column + column;

        // Initialize random number generator for the current thread
        curand_init(1337, pixel_index, 0, &rand_state[pixel_index]);
    }

    __global__ void rand_init(curandState *rand_state) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            curand_init(1337, 0, 0, rand_state);
        }
    }

    __global__ void render(
        color *frame_buffer,
        int max_column,
        int max_row,
        emit::camera **cam,
        intersectable **world,
        curandState *rand_state,
        int samples_per_pixel,
        int max_depth) {

        int column = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;

        if ((column >= max_column) || (row >= max_row)) {
            return;
        }

        // Pixel index in the frame buffer (pixel = 3 floats)
        int pixel_index = row * max_column + column;

        curandState local_rand_state = rand_state[pixel_index];
        color pixel_color(0, 0, 0);
        for(int s = 0; s < samples_per_pixel; s++) {
            float u = (column + curand_uniform(&local_rand_state)) / static_cast<float>(max_column);
            float v = (row + curand_uniform(&local_rand_state)) / static_cast<float>(max_row);
            emit::ray r = (*cam)->get_ray(u,v, &local_rand_state);
            pixel_color += ray_color(r, world, &local_rand_state, max_depth);
        }
        rand_state[pixel_index] = local_rand_state;
        frame_buffer[pixel_index] = pixel_color;
    }

    __global__ void create_world(
        intersectable** objects_list,
        intersectable** world,
        emit::camera** camera,
        size_t num_intersectables,
        const sphere_config* spheres_config) {

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Create the ground sphere
            objects_list[0] = new objects::sphere(
                calc::vecs(0, -1000, 0),
                1000.0,
                new mat::lambertian(color(0.5, 0.5, 0.5))
            );

            // Create the rest of the spheres from the config
            for (size_t i = 1; i < num_intersectables; ++i) {
                const sphere_config& config = spheres_config[i - 1]; // Offset by 1 because ground sphere is already added
                calc::vecs position(config.x, config.y, config.z);

                switch (config.material_type) {
                    case 0: // Lambertian
                        objects_list[i] = new objects::sphere(position, config.radius,
                            new mat::lambertian(color(config.color_r, config.color_g, config.color_b)));
                    break;
                    case 1: // Metal
                        objects_list[i] = new objects::sphere(position, config.radius,
                            new mat::metal(color(config.color_r, config.color_g, config.color_b), config.fuzz));
                    break;
                    case 2: // Dielectric
                        objects_list[i] = new objects::sphere(position, config.radius,
                            new mat::dielectric(config.refractive_index));
                    break;
                }
            }

            *world = new hitlist(objects_list, num_intersectables);

            // Camera setup (unchanged)
            calc::point lookfrom(13, 2, 3);
            calc::point lookat(0, 0, 0);
            calc::vecs vup(0, 1, 0);
            float dist_to_focus = 10.0;
            float aperture = 0.1;
            *camera = new emit::camera(lookfrom, lookat, vup, 20, 3.0 / 2.0, aperture, dist_to_focus);
        }
    }

	__global__ static void free_world(
	    canvas::intersectable **objects_list,
	    canvas::intersectable **world,
	    emit::camera **camera,
	    size_t num_intersectables) {
    	for (int i = 0; i < num_intersectables; i++) {
    		delete static_cast<objects::sphere *>(objects_list[i])->mat_ptr;
    		delete objects_list[i];
    	}
    	delete *world;
    	delete *camera;
    }

    __global__ void render_2d_scene(
    canvas::color *frame_buffer,
    int width,
    int height,
    const sphere_config *spheres,
    size_t num_spheres
) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        // Normalize pixel coordinates to [0, 1]
        float u = float(x) / width;
        float v = float(y) / height;

        // Map pixel coordinates to a world-space range
        float world_x = (u - 0.5f) * 20.0f; // Adjust scale as needed
        float world_y = (v - 0.5f) * 20.0f;

        canvas::color pixel_color(1.0f, 1.0f, 1.0f); // Default white background

        for (size_t i = 0; i < num_spheres; ++i) {
            const sphere_config &sphere = spheres[i];

            // Check if the pixel is within the sphere's 2D projection
            float dx = world_x - sphere.x;
            float dy = world_y - sphere.y;
            float distance_squared = dx * dx + dy * dy;

            if (distance_squared <= sphere.radius * sphere.radius) {
                // If inside the sphere, use its material color
                pixel_color = canvas::color(sphere.color_r, sphere.color_g, sphere.color_b);
                break; // Stop once the closest sphere is found
            }
        }

        int pixel_index = y * width + x;
        frame_buffer[pixel_index] = pixel_color;
    }

}

#endif //RENDERER_CUH

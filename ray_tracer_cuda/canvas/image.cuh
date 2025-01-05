#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "cuda_runtime.h"
#include "../cuda_o/cuda_check.cuh"
#include "../canvas/color.cuh"
#include "../calc/math_utils.cuh"
#include <fstream>

namespace canvas {

	class image {
	public:
		__host__ static void write_ppm_image(
			std::ofstream &out,
			color *frame_buffer,
			int width,
			int height,
			int samples_per_pixel
			) {
			out << "P3\n" << width << " " << height << "\n255\n";
			for(int row = height - 1; row >= 0; row--) {
				for(int col = 0; col < width; col++) {
					size_t pixel_index = row*width + col;
					float r = frame_buffer[pixel_index].x();
					float g = frame_buffer[pixel_index].y();
					float b = frame_buffer[pixel_index].z();

					float scale = 1.0 / samples_per_pixel;
					r = sqrt(r * scale);
					g = sqrt(g * scale);
					b = sqrt(b * scale);

					int ir = int(255.99* calc::clamp(r, 0.0, 0.999));
					int ig = int(255.99* calc::clamp(g, 0.0, 0.999));
					int ib = int(255.99* calc::clamp(b, 0.0, 0.999));
					out << ir << " " << ig << " " << ib << "\n";
				}
			}
		}

	};
}

#endif
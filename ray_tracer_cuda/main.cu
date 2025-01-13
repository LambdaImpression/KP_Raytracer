#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./emitters/camera.cuh"
#include "./canvas/color.cuh"
#include "./cuda_o/cuda_check.cuh"
#include "./canvas/intersectable.cuh"
#include "./canvas/image.cuh"
#include "canvas/renderer.cuh"
#include "tests/run_all_tests.cu"
#include "include/json.hpp"
#include <curand_kernel.h>
#include <vector>

__host__ void load_spheres_from_json(const std::string& filename, std::vector<sphere_config>& spheres) {
    std::ifstream file(filename);
    nlohmann::json json_data;
    file >> json_data;

    for (const auto& sphere : json_data["spheres"]) {
        sphere_config config;

        // Check for required keys and set defaults if necessary
        if (!sphere.contains("position") || !sphere["position"].contains("x") ||
            !sphere["position"].contains("y") || !sphere["position"].contains("z")) {
            throw std::runtime_error("Missing position data in JSON.");
            }

        config.x = sphere["position"]["x"];
        config.y = sphere["position"]["y"];
        config.z = sphere["position"]["z"];

        if (!sphere.contains("radius")) {
            throw std::runtime_error("Missing radius data in JSON.");
        }
        config.radius = sphere["radius"];

        if (!sphere.contains("material") || !sphere["material"].contains("type")) {
            throw std::runtime_error("Missing material data in JSON.");
        }
        config.material_type = sphere["material"]["type"];

        // Handle optional material properties with default values
        config.color_r = sphere["material"].value("color", nlohmann::json::object()).value("r", 0.5f);
        config.color_g = sphere["material"].value("color", nlohmann::json::object()).value("g", 0.5f);
        config.color_b = sphere["material"].value("color", nlohmann::json::object()).value("b", 0.5f);
        config.fuzz = sphere["material"].value("fuzz", 0.0f);
        config.refractive_index = sphere["material"].value("refractive_index", 1.0f);

        spheres.push_back(config);
    }
}

// Adjust value to the nearest perfect square
__host__ int adjustToNearestSquare(int value) {
    return static_cast<int>(std::sqrt(value));
}

// Helper function to add an extension if missing
__host__ std::string ensureFileExtension(const std::string& filepath, const std::string& extension) {
    // Check if the filepath ends with the specified extension
    if (filepath.size() >= extension.size() &&
        filepath.compare(filepath.size() - extension.size(), extension.size(), extension) == 0) {
        return filepath; // Extension already present
        }
    return filepath + extension; // Append the extension
}

int main(int argc, char* argv[]) {
    // Default values
    int width = 1200;
    int height = 800;
    std::string filename = "created_images/default.ppm";
    std::string json_file_path = "scenes/default_scene.json";
    int num_threads = 8;
    int num_blocks = 8;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (arg == "--filename" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_file_path = argv[++i];
        } else if (arg == "--num_blocks" && i + 1 < argc) {
            num_blocks = std::atoi(argv[++i]);
        } else if (arg == "--num_threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--run_tests") {  // No additional argument needed
            run_all_tests();
            return 0;  // Exit after running tests
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " "
                                             "[--width WIDTH] "
                                             "[--height HEIGHT] "
                                             "[--filename FILENAME] "
                                             "[--json JSON_FILE] "
                                             "[--num_blocks THREADS_COL] "
                                             "[--num_threads THREADS_ROW] "
                                             "[--run_tests]" << std::endl;
            return 1;
        }
    }

    // Ensure extensions
    filename = ensureFileExtension(filename, ".ppm");
    json_file_path = ensureFileExtension(json_file_path, ".json");

    // Clamp user-input to a valid range
    width = static_cast<int>(calc::clamp(static_cast<float>(width), 128, 3840));
    height = static_cast<int>(calc::clamp(static_cast<float>(height), 128, 2160));
    num_threads = static_cast<int>(calc::clamp(static_cast<float>(num_threads),
        std::ceil(std::max(static_cast<float>(width) / static_cast<float>(num_blocks),
            static_cast<float>(height) / static_cast<float>(num_blocks))), 1024));
    num_blocks = static_cast<int>(calc::clamp(static_cast<float>(num_blocks),1 , 65535));

    // Adjust thread and block values to nearest square
    num_threads = adjustToNearestSquare(num_threads);
    num_blocks = adjustToNearestSquare(num_blocks);

    // Nb of blocks in the grid
    dim3 blocks(width/num_blocks + 1, height/num_blocks + 1);
    // Nb of threads in each block (one per pixel)
    dim3 threads(num_threads, num_threads);

    const int num_pixels = width * height;
    constexpr int samples_per_pixel = 100;
    constexpr int max_depth = 50;

    canvas::intersectable **objects_list, **world;
    emit::camera **camera;
    canvas::color *frame_buffer;
    curandState *random_state, *random_state2;

    // Allocate Frame Buffer
    cuda_o::checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&frame_buffer),
        width*height*sizeof(canvas::color)));

    // Allocate world
    std::vector<sphere_config> spheres;
    load_spheres_from_json(json_file_path, spheres);
    sphere_config* spheres_configs;
    size_t num_intersectables = spheres.size();

    cuda_o::checkCudaErrors(cudaMalloc(&spheres_configs,
        num_intersectables * sizeof(sphere_config)));
    cuda_o::checkCudaErrors(cudaMemcpy(spheres_configs,
        spheres.data(),
        num_intersectables * sizeof(sphere_config),
        cudaMemcpyHostToDevice));

    cuda_o::checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&objects_list),
        num_intersectables*sizeof(canvas::intersectable *)));
    cuda_o::checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&world),
        sizeof(canvas::intersectable *)));
    cuda_o::checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&camera),
        sizeof(emit::camera *)));

    // List of pixels random number generator states
    cuda_o::checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&random_state),
        num_pixels*sizeof(curandState)));
    cuda_o::checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&random_state2),
        1*sizeof(curandState)));

    canvas::rand_init<<<1,1>>>(random_state2);
    cuda_o::checkCudaErrors(cudaGetLastError());
    cuda_o::checkCudaErrors(cudaDeviceSynchronize());

    create_world<<<1,1>>>(objects_list, world, camera, num_intersectables, spheres_configs);
    cuda_o::checkCudaErrors(cudaGetLastError());
    cuda_o::checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Initialize the render random states
    canvas::render_init<<<blocks, threads>>>(width, height, random_state);
    cuda_o::checkCudaErrors(cudaGetLastError());
    cuda_o::checkCudaErrors(cudaDeviceSynchronize());

    // Render the scene
    render<<<blocks, threads>>>(
        frame_buffer,
        width,
        height,
        camera,
        world,
        random_state,
        samples_per_pixel,
        max_depth);
    cuda_o::checkCudaErrors(cudaGetLastError());
    cuda_o::checkCudaErrors(cudaDeviceSynchronize());

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert to seconds and display
    float seconds = milliseconds / 1000.0f;
    printf("CUDA Rendering Time: %.3f seconds\n", seconds);

    printf("CUDA Rendering Time: %.3f ms\n", milliseconds);

    // Write the output image to a file
    std::ofstream file(filename, std::ios::out);
    if (!file) {
        std::cerr << "Error: Unable to create or open the file!" << std::endl;
        return -1;
    }
    canvas::image::write_ppm_image(file, frame_buffer, width, height, samples_per_pixel);
    file.close();

    // Clean up
    cuda_o::checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(objects_list, world, camera, num_intersectables);
    cuda_o::checkCudaErrors(cudaGetLastError());
    cuda_o::checkCudaErrors(cudaFree(camera));
    cuda_o::checkCudaErrors(cudaFree(world));
    cuda_o::checkCudaErrors(cudaFree(objects_list));
    cuda_o::checkCudaErrors(cudaFree(random_state));
    cuda_o::checkCudaErrors(cudaFree(random_state2));
    cuda_o::checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();
}


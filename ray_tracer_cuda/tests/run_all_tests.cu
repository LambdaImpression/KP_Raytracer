#include <iostream>

// Declare test functions from each test file
extern void run_vecs_tests();
extern void run_math_util_tests();
extern void run_hitlist_tests();
extern void run_intersectable_tests();
extern void run_renderer_tests();
extern void run_camera_tests();
extern void run_ray_tests();
extern void run_material_tests();
extern void run_sphere_tests();

// Add more test declarations as needed for other test files
int run_all_tests() {
    std::cout << "Running all tests...\n";

    // Call test functions
    run_vecs_tests();
    run_math_util_tests();
    run_hitlist_tests();
    run_intersectable_tests();
    run_camera_tests();
    run_ray_tests();
    run_material_tests();
    run_sphere_tests();
    return 0;
}

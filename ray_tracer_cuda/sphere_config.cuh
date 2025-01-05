// SphereConfig.h
#ifndef SPHERECONFIG_CUH
#define SPHERECONFIG_CUH

struct sphere_config {
    float x, y, z;  // Sphere center position
    float radius;   // Sphere radius
    int material_type; // Material type (e.g., 0 = lambertian, 1 = metal, 2 = dielectric)
    float color_r, color_g, color_b; // Material color (if applicable)
    float fuzz; // Fuzziness for metal materials
    float refractive_index; // Refractive index for dielectric materials
};

#endif // SPHERECONFIG_CUH

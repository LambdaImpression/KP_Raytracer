# Ray Tracers: Golang and CUDA Versions

This repository contains two implementations of a ray tracer:
- A **Golang ray tracer** in the `ray_tracer_go_v2` folder.
- A **CUDA ray tracer** in the `ray_tracer_cuda` folder.

Both implementations support rendering scenes described in JSON files and provide command-line arguments for customization. Below, you’ll find the instructions to build, run, and test both ray tracers.

---

## Golang Ray Tracer

### Build Instructions
1. Navigate to the `ray_tracer_go_v2` folder:
   ```sh
   cd ray_tracer_go_v2
   ```
2. Build the project:
   ```sh
   go build
   ```

### Run Instructions
The Golang ray tracer accepts several command-line arguments:

```sh
./ray_tracer_go_v2 [flags]
```

#### Available Flags:
- `-fn`: Path to save the generated image (default: `.\created_images\default.ppm`).
- `-j`: Path to the JSON scene file (default: `.\scenes\default_scene.json`).
- `-p`: Number of processes for parallelism (default: number of CPUs).
- `-t`: Number of goroutines for parallelism (default: 4).
- `-wi`: Image width (default: 1200).
- `-hi`: Image height (default: 800).

#### Example:
Render an image with custom settings:
```sh
./ray_tracer_go_v2 -fn .\output\image.ppm -j .\scenes\custom_scene.json -wi 1920 -hi 1080
```

### Run Tests
1. Navigate to the `tests` folder:
   ```sh
   cd tests
   ```
2. Run the tests:
   ```sh
   go test
   ```

---

## CUDA Ray Tracer

### Build Instructions
1. Navigate to the `ray_tracer_cuda` folder:
   ```sh
   cd ray_tracer_cuda
   ```
2. Build the project using `nvcc`:
   ```sh
   nvcc main.cu tests/calc/test_math_utils.cu tests/calc/test_vecs.cu \
   tests/canvas/test_hitlist.cu tests/canvas/test_intersectable.cu \
   tests/emitters/test_camera.cu tests/emitters/test_ray.cu \
   tests/material/test_materials.cu tests/objects/test_sphere.cu \
   -o ray_tracer \
   -I./Calc -I./canvas -I./cuda_o -I./emitters -I./include -I./material -I./objects -I./tests
   ```

### Run Instructions
The CUDA ray tracer accepts several command-line arguments:

```sh
.\ray_tracer.exe [arguments]
```

#### Available Arguments:
- `--width`: Image width (default: varies).
- `--height`: Image height (default: varies).
- `--filename`: Path to save the generated image.
- `--json`: Path to the JSON scene file.
- `--num_blocks`: Number of thread blocks for CUDA parallelism.
- `--num_threads`: Number of threads per block for CUDA parallelism.
- `--run_tests`: Run all unit tests and exit.

#### Example:
Render an image with custom settings:
```sh
.\ray_tracer.exe --width 1920 --height 1080 --filename .\output\image.ppm --json .\scenes\custom_scene.json
```

To run tests:
```sh
.\ray_tracer.exe --run_tests
```

---

## Example JSON Scene
Both ray tracers can process scenes defined in JSON files. A default JSON scene is included in each ray tracer's folder, featuring all available material types. You can modify it as needed to create custom scenes.

---

## Notes
- Both ray tracers provide multithreading or parallelism support:
  - Golang: Controlled by `-p` (processes) and `-t` (goroutines).
  - CUDA: Controlled by `--num_blocks` and `--num_threads`.
- The default scenes demonstrate the capabilities of each ray tracer, including all material types for spheres.
- Ensure that the CUDA environment (e.g., NVIDIA CUDA Toolkit) is set up correctly before building the CUDA ray tracer.

Enjoy rendering your scenes!

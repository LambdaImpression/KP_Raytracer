package renderer

import (
	"math"
	"math/rand"
	"ray_tracer_go_v2/material"
	"ray_tracer_go_v2/objects"
	"sync"

	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/canvas"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
)

// rayColor computes the color for a ray.
func RayColor(r emitters.Ray, world canvas.Intersectable, maxDepth int) calc.Vec3 {
	white := calc.NewVec3(1.0, 1.0, 1.0)
	blue := calc.NewVec3(0.5, 0.7, 1.0)

	curRay := r
	curAttenuation := calc.NewVec3(1.0, 1.0, 1.0)

	for depth := 0; depth < maxDepth; depth++ {
		var rec common.HitRecord
		if world.Hit(curRay, 0.001, math.MaxFloat64, &rec) {
			var scattered emitters.Ray
			var attenuation calc.Vec3
			if rec.Material.Scatter(curRay, &rec, &attenuation, &scattered) {
				// Update attenuation and ray
				curAttenuation = curAttenuation.Mul(attenuation)
				curRay = scattered
			} else {
				// Absorption leads to black color
				return calc.NewVec3(0.0, 0.0, 0.0)
			}
		} else {

			// No hit: Return background gradient
			unitDirection := curRay.Direction.Normalize()
			t := 0.5 * (unitDirection.Y + 1.0)
			return curAttenuation.Mul(white.MulScalar(1.0 - t).Add(blue.MulScalar(t)))
		}
	}
	// Exceeded depth
	return calc.NewVec3(0.0, 0.0, 0.0)
}

// Render splits the rendering task into multiple processes and threads.
func Render(
	frameBuffer []calc.Vec3,
	width, height int,
	camera emitters.Camera,
	world canvas.Intersectable,
	samplesPerPixel, maxDepth, numProcesses, numThreads int) {

	var processWG sync.WaitGroup
	processHeight := height / numProcesses

	for p := 0; p < numProcesses; p++ {
		startRow := p * processHeight
		endRow := startRow + processHeight
		if p == numProcesses-1 {
			endRow = height // Ensure last block covers the remaining rows
		}

		processWG.Add(1)
		go func(startRow, endRow int) {
			defer processWG.Done()

			var threadWG sync.WaitGroup
			rowsPerThread := (endRow - startRow) / numThreads

			for t := 0; t < numThreads; t++ {
				threadStartRow := startRow + t*rowsPerThread
				threadEndRow := threadStartRow + rowsPerThread
				if t == numThreads-1 {
					threadEndRow = endRow // Ensure last thread covers remaining rows
				}

				threadWG.Add(1)
				go func(threadStartRow, threadEndRow int) {
					defer threadWG.Done()
					for y := threadStartRow; y < threadEndRow; y++ {
						for x := 0; x < width; x++ {
							pixelColor := calc.NewVec3(0, 0, 0)
							for s := 0; s < samplesPerPixel; s++ {
								u := (float64(x) + rand.Float64()) / float64(width)
								v := (float64(y) + rand.Float64()) / float64(height)
								ray := camera.GetRay(u, v)
								pixelColor = pixelColor.Add(RayColor(ray, world, maxDepth))
							}
							// Compute pixel index and store final color in frame buffer
							pixelIndex := y*width + x
							frameBuffer[pixelIndex] = pixelColor.DivScalar(float64(samplesPerPixel))
						}
					}
				}(threadStartRow, threadEndRow)
			}
			threadWG.Wait()
		}(startRow, endRow)
	}

	processWG.Wait()
}

func CreateWorld(sceneConfig *objects.SceneConfig) (canvas.Intersectable, *emitters.Camera) {

	ground := objects.NewSphere(
		calc.NewVec3(0, -1000, 0), // Center position of the ground sphere
		1000.0,                    // Large radius for the ground
		material.NewLambertian(calc.NewVec3(0.5, 0.5, 0.5)), // Gray Lambertian material
	)

	// Create spheres from the SceneConfig
	spheres := []canvas.Intersectable{ground}
	for _, sphere := range sceneConfig.Spheres {
		spheres = append(spheres, objects.NewSphere(sphere.Center, sphere.Radius, sphere.Material))
	}

	// Combine into a hit list
	world := canvas.NewHitList(spheres)

	// Set up the camera
	lookFrom := calc.NewVec3(13, 2, 3)
	lookAt := calc.NewVec3(0, 0, 0)
	vUp := calc.NewVec3(0, 1, 0)
	distToFocus := 10.0
	aperture := 0.1
	camera := emitters.NewCamera(lookFrom, lookAt, vUp, 20, 3.0/2.0, aperture, distToFocus)

	return world, camera
}

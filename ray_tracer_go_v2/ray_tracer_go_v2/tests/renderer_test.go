package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/canvas"
	"ray_tracer_go_v2/emitters"
	"ray_tracer_go_v2/material"
	"ray_tracer_go_v2/objects"
	"ray_tracer_go_v2/renderer"
	"testing"
)

func TestRayColor_NoHit(t *testing.T) {
	// Create a world with no objects
	world := canvas.NewHitList([]canvas.Intersectable{})

	// Create a ray that will not intersect
	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(0, 0, -1))

	color := renderer.RayColor(ray, world, 50)

	// Background color gradient expected
	unitDirection := ray.Direction.Normalize()
	tVal := 0.5 * (unitDirection.Y + 1.0)
	expectedColor := calc.NewVec3(1.0, 1.0, 1.0).MulScalar(1.0 - tVal).Add(calc.NewVec3(0.5, 0.7, 1.0).MulScalar(tVal))

	if math.Abs(color.X-expectedColor.X) > 1e-8 ||
		math.Abs(color.Y-expectedColor.Y) > 1e-8 ||
		math.Abs(color.Z-expectedColor.Z) > 1e-8 {
		t.Errorf("RayColor() = %v; want %v", color, expectedColor)
	}
}

func TestRayColor_Hit(t *testing.T) {
	// Create a sphere and a world
	sphere := objects.NewSphere(calc.NewVec3(0, 0, -1), 0.5, material.NewLambertian(calc.NewVec3(0.8, 0.6, 0.2)))
	world := canvas.NewHitList([]canvas.Intersectable{sphere})

	// Create a ray that intersects the sphere
	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(0, 0, -1))

	color := renderer.RayColor(ray, world, 50)

	// The exact color depends on the Lambertian scattering; ensure no black (missed color).
	if color.X <= 0 || color.Y <= 0 || color.Z <= 0 {
		t.Errorf("RayColor() = %v; expected non-zero color components", color)
	}
}

func TestRender(t *testing.T) {
	width := 10
	height := 10
	frameBuffer := make([]calc.Vec3, width*height)

	// Simple world with one sphere
	sphere := objects.NewSphere(calc.NewVec3(0, 0, -1), 0.5, material.NewLambertian(calc.NewVec3(0.8, 0.6, 0.2)))
	world := canvas.NewHitList([]canvas.Intersectable{sphere})

	// Camera setup
	lookFrom := calc.NewVec3(3, 3, 2)
	lookAt := calc.NewVec3(0, 0, -1)
	vUp := calc.NewVec3(0, 1, 0)
	camera := emitters.NewCamera(lookFrom, lookAt, vUp, 20, float64(width)/float64(height), 0.0, 1.0)

	renderer.Render(frameBuffer, width, height, *camera, world, 5, 10, 1, 1)

	// Check if the frame buffer contains non-zero values
	hasColor := false
	for _, pixel := range frameBuffer {
		if pixel.X > 0 || pixel.Y > 0 || pixel.Z > 0 {
			hasColor = true
			break
		}
	}

	if !hasColor {
		t.Errorf("Render failed; frameBuffer contains only zero values")
	}
}

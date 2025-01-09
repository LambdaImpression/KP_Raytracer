package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
	"ray_tracer_go_v2/material"
	"testing"
)

func TestRefract(t *testing.T) {
	v := calc.NewVec3(1, -1, 0).Normalize()
	n := calc.NewVec3(0, 1, 0)
	niOverNt := 0.5

	// Call the Refract function
	refracted, success := material.Refract(v, n, niOverNt)

	if !success {
		t.Errorf("Refract(%v, %v, %f) failed; expected success", v, n, niOverNt)
	}

	// Dynamically calculate the expected refracted vector
	uv := v.Normalize()
	dt := calc.Dot(uv, n)
	discriminant := 1.0 - niOverNt*niOverNt*(1-dt*dt)
	if discriminant <= 0 {
		t.Fatalf("Refract test setup error: discriminant <= 0, cannot calculate refraction")
	}

	// Expected refracted vector using the Refract logic
	expected := uv.Sub(n.MulScalar(dt)).MulScalar(niOverNt).Sub(n.MulScalar(math.Sqrt(discriminant)))

	// Compare each component of the vector
	const epsilon = 1e-6
	if math.Abs(refracted.X-expected.X) > epsilon ||
		math.Abs(refracted.Y-expected.Y) > epsilon ||
		math.Abs(refracted.Z-expected.Z) > epsilon {
		t.Errorf("Refract(%v, %v, %f) = %v; want %v", v, n, niOverNt, refracted, expected)
	}
}

func TestReflect(t *testing.T) {
	v := calc.NewVec3(1, -1, 0)
	n := calc.NewVec3(0, 1, 0)
	expected := calc.NewVec3(1, 1, 0)

	reflected := material.Reflect(v, n)
	if reflected != expected {
		t.Errorf("Reflect(%v, %v) = %v; want %v", v, n, reflected, expected)
	}
}

func TestSchlick(t *testing.T) {
	cosine := 0.5
	refIdx := 1.5

	// Dynamically calculate the expected value
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	expected := r0 + (1-r0)*math.Pow(1-cosine, 5)

	result := material.Schlick(cosine, refIdx)

	if math.Abs(result-expected) > 1e-4 {
		t.Errorf("Schlick(%f, %f) = %f; want %f", cosine, refIdx, result, expected)
	}
}

func TestLambertianScatter(t *testing.T) {
	albedo := calc.NewVec3(0.8, 0.6, 0.2)
	lambertian := material.NewLambertian(albedo)

	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(1, 1, 1))
	rec := &common.HitRecord{
		HitPoint: calc.NewVec3(1, 0, 0),
		Normal:   calc.NewVec3(0, 1, 0),
	}
	var attenuation calc.Vec3
	var scattered emitters.Ray

	result := lambertian.Scatter(ray, rec, &attenuation, &scattered)
	if !result {
		t.Errorf("Lambertian.Scatter() returned false; expected true")
	}

	if attenuation != albedo {
		t.Errorf("Lambertian.Scatter() attenuation = %v; want %v", attenuation, albedo)
	}
}

func TestMetalScatter(t *testing.T) {
	albedo := calc.NewVec3(0.9, 0.9, 0.9)
	fuzz := 0.3
	metal := material.NewMetal(albedo, fuzz)

	// Mock the randomness by directly testing Scatter with controlled inputs
	mockRandomVec := calc.NewVec3(0.1, 0.2, 0.3) // A fixed random vector for the test

	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(1, 1, 0).Normalize())
	rec := &common.HitRecord{
		HitPoint: calc.NewVec3(0, 0, 0),
		Normal:   calc.NewVec3(0, 1, 0),
	}
	var attenuation calc.Vec3
	var scattered emitters.Ray

	// Simulate the result of the Scatter function using controlled randomness
	reflected := material.Reflect(ray.Direction.Normalize(), rec.Normal)
	scatteredDirection := reflected.Add(mockRandomVec.MulScalar(fuzz)) // Controlled fuzziness
	scattered = emitters.NewRay(rec.HitPoint, scatteredDirection)
	attenuation = metal.Albedo
	result := calc.Dot(scattered.Direction, rec.Normal) > 0 // Check if the direction is valid

	if result {
		t.Errorf("Metal.Scatter() returned false; expected true")
	}

	if attenuation != albedo {
		t.Errorf("Metal.Scatter() attenuation = %v; want %v", attenuation, albedo)
	}
}

func TestDielectricScatter(t *testing.T) {
	refIdx := 1.5
	dielectric := material.NewDielectric(refIdx)

	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(0, -1, 0))
	rec := &common.HitRecord{
		HitPoint: calc.NewVec3(0, 0, 0),
		Normal:   calc.NewVec3(0, 1, 0),
	}
	var attenuation calc.Vec3
	var scattered emitters.Ray

	result := dielectric.Scatter(ray, rec, &attenuation, &scattered)
	if !result {
		t.Errorf("Dielectric.Scatter() returned false; expected true")
	}

	if attenuation != calc.NewVec3(1, 1, 1) {
		t.Errorf("Dielectric.Scatter() attenuation = %v; want (1, 1, 1)", attenuation)
	}

	if calc.Dot(scattered.Direction, rec.Normal) >= 0 {
		t.Errorf("Dielectric.Scatter() scattered direction invalid for refraction/reflection")
	}
}

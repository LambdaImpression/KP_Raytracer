package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
	"ray_tracer_go_v2/material"
	"ray_tracer_go_v2/objects"
	"testing"
)

func TestSphereHit(t *testing.T) {
	sphere := objects.NewSphere(calc.NewVec3(0, 0, -1), 0.5, material.NewLambertian(calc.NewVec3(0.8, 0.6, 0.2)))

	ray := emitters.NewRay(calc.NewVec3(0, 0, 0), calc.NewVec3(0, 0, -1))
	record := &common.HitRecord{}

	hit := sphere.Hit(ray, 0.1, math.Inf(1), record)
	if !hit {
		t.Fatalf("Sphere.Hit() = false; expected true")
	}

	expectedT := 0.5
	if math.Abs(record.T-expectedT) > 1e-8 {
		t.Errorf("Sphere.Hit() record.T = %f; want %f", record.T, expectedT)
	}

	expectedHitPoint := calc.NewVec3(0, 0, -0.5)
	if record.HitPoint != expectedHitPoint {
		t.Errorf("Sphere.Hit() HitPoint = %v; want %v", record.HitPoint, expectedHitPoint)
	}

	expectedNormal := calc.NewVec3(0, 0, 1)
	if record.Normal != expectedNormal {
		t.Errorf("Sphere.Hit() Normal = %v; want %v", record.Normal, expectedNormal)
	}
}

func TestSphereMiss(t *testing.T) {
	sphere := objects.NewSphere(calc.NewVec3(0, 0, -1), 0.5, material.NewLambertian(calc.NewVec3(0.8, 0.6, 0.2)))

	ray := emitters.NewRay(calc.NewVec3(0, 2, 0), calc.NewVec3(0, 0, -1))
	record := &common.HitRecord{}

	hit := sphere.Hit(ray, 0.1, math.Inf(1), record)
	if hit {
		t.Fatalf("Sphere.Hit() = true; expected false")
	}
}

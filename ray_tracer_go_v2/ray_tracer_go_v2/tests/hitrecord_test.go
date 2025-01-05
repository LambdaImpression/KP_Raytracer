package tests

import (
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
	"testing"
)

func TestSetFaceNormal_FrontFace(t *testing.T) {
	// Create a test ray and outward normal
	ray := emitters.Ray{
		Origin:    calc.NewVec3(0, 0, 0),
		Direction: calc.NewVec3(1, 0, 0),
	}
	outwardNormal := calc.NewVec3(-1, 0, 0)

	// Initialize HitRecord
	hr := &common.HitRecord{}
	hr.SetFaceNormal(ray, outwardNormal)

	// Test for front face
	if !hr.FrontFace {
		t.Errorf("SetFaceNormal failed; expected FrontFace to be true")
	}
	if hr.Normal != outwardNormal {
		t.Errorf("SetFaceNormal failed; expected Normal = %v, got %v", outwardNormal, hr.Normal)
	}
}

func TestSetFaceNormal_BackFace(t *testing.T) {
	// Create a test ray and outward normal
	ray := emitters.Ray{
		Origin:    calc.NewVec3(0, 0, 0),
		Direction: calc.NewVec3(1, 0, 0),
	}
	outwardNormal := calc.NewVec3(1, 0, 0)

	// Initialize HitRecord
	hr := &common.HitRecord{}
	hr.SetFaceNormal(ray, outwardNormal)

	// Test for back face
	if hr.FrontFace {
		t.Errorf("SetFaceNormal failed; expected FrontFace to be false")
	}
	if hr.Normal != outwardNormal.Negate() {
		t.Errorf("SetFaceNormal failed; expected Normal = %v, got %v", outwardNormal.Negate(), hr.Normal)
	}
}

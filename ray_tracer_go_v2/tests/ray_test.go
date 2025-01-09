package tests

import (
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/emitters"
	"testing"
)

func TestNewRay(t *testing.T) {
	origin := calc.NewVec3(1, 2, 3)
	direction := calc.NewVec3(4, 5, 6)

	ray := emitters.NewRay(origin, direction)

	if ray.Origin != origin {
		t.Errorf("NewRay failed: Origin = %v; want %v", ray.Origin, origin)
	}
	if ray.Direction != direction {
		t.Errorf("NewRay failed: Direction = %v; want %v", ray.Direction, direction)
	}
}

func TestAt(t *testing.T) {
	origin := calc.NewVec3(1, 2, 3)
	direction := calc.NewVec3(0, 1, 0)
	ray := emitters.NewRay(origin, direction)

	tCases := []struct {
		t        float64
		expected calc.Vec3
	}{
		{0, calc.NewVec3(1, 2, 3)},     // Point at the origin
		{1, calc.NewVec3(1, 3, 3)},     // One unit along the direction
		{2.5, calc.NewVec3(1, 4.5, 3)}, // 2.5 units along the direction
		{-1, calc.NewVec3(1, 1, 3)},    // Negative direction
	}

	for _, tc := range tCases {
		result := ray.At(tc.t)
		if result != tc.expected {
			t.Errorf("At(%f) = %v; want %v", tc.t, result, tc.expected)
		}
	}
}

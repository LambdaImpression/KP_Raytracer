package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"testing"
)

func TestDegreesToRadians(t *testing.T) {
	tests := []struct {
		degrees float64
		want    float64
	}{
		{0, 0},
		{90, math.Pi / 2},
		{180, math.Pi},
		{360, 2 * math.Pi},
	}

	for _, tt := range tests {
		t.Run(
			"DegreesToRadians",
			func(t *testing.T) {
				got := calc.DegreesToRadians(tt.degrees)
				if math.Abs(got-tt.want) > 1e-9 {
					t.Errorf("DegreesToRadians(%f) = %f; want %f", tt.degrees, got, tt.want)
				}
			},
		)
	}
}

func TestClamp(t *testing.T) {
	tests := []struct {
		x, min, max, want float64
	}{
		{5, 0, 10, 5},
		{-5, 0, 10, 0},
		{15, 0, 10, 10},
		{7.5, 5, 10, 7.5},
	}

	for _, tt := range tests {
		t.Run(
			"Clamp",
			func(t *testing.T) {
				got := calc.Clamp(tt.x, tt.min, tt.max)
				if got != tt.want {
					t.Errorf("Clamp(%f, %f, %f) = %f; want %f", tt.x, tt.min, tt.max, got, tt.want)
				}
			},
		)
	}
}

func TestRandomFloat(t *testing.T) {
	var state uint32 = 42

	// Run multiple iterations and verify the values are between 0 and 1
	for i := 0; i < 100; i++ {
		t.Run(
			"RandomFloat",
			func(t *testing.T) {
				got := calc.RandomFloat(&state)
				if got < 0 || got >= 1 {
					t.Errorf("RandomFloat() = %f; expected a value between 0 and 1", got)
				}
			},
		)
	}
}

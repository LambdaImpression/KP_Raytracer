package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/emitters"
	"testing"
)

func TestNewCamera(t *testing.T) {
	lookFrom := calc.NewVec3(0, 0, 0)
	lookAt := calc.NewVec3(0, 0, -1)
	vUp := calc.NewVec3(0, 1, 0)
	vFov := 90.0
	aspectRatio := 16.0 / 9.0
	aperture := 2.0
	focusDist := 1.0

	camera := emitters.NewCamera(lookFrom, lookAt, vUp, vFov, aspectRatio, aperture, focusDist)

	// Recalculate expected values dynamically
	theta := calc.DegreesToRadians(vFov)
	h := math.Tan(theta / 2)
	viewportHeight := 2.0 * h
	viewportWidth := aspectRatio * viewportHeight

	expectedHorizontal := calc.NewVec3(viewportWidth, 0, 0)
	expectedVertical := calc.NewVec3(0, viewportHeight, 0)
	expectedLowerLeftCorner := lookFrom.Sub(expectedHorizontal.DivScalar(2)).
		Sub(expectedVertical.DivScalar(2)).
		Sub(calc.NewVec3(0, 0, focusDist))

	const epsilon = 1e-6

	if math.Abs(camera.LowerLeftCorner.X-expectedLowerLeftCorner.X) > epsilon ||
		math.Abs(camera.LowerLeftCorner.Y-expectedLowerLeftCorner.Y) > epsilon ||
		math.Abs(camera.LowerLeftCorner.Z-expectedLowerLeftCorner.Z) > epsilon {
		t.Errorf("NewCamera failed: LowerLeftCorner = %v; want %v", camera.LowerLeftCorner, expectedLowerLeftCorner)
	}

	if math.Abs(camera.Horizontal.X-expectedHorizontal.X) > epsilon ||
		math.Abs(camera.Horizontal.Y-expectedHorizontal.Y) > epsilon ||
		math.Abs(camera.Horizontal.Z-expectedHorizontal.Z) > epsilon {
		t.Errorf("NewCamera failed: Horizontal = %v; want %v", camera.Horizontal, expectedHorizontal)
	}

	if math.Abs(camera.Vertical.X-expectedVertical.X) > epsilon ||
		math.Abs(camera.Vertical.Y-expectedVertical.Y) > epsilon ||
		math.Abs(camera.Vertical.Z-expectedVertical.Z) > epsilon {
		t.Errorf("NewCamera failed: Vertical = %v; want %v", camera.Vertical, expectedVertical)
	}
}

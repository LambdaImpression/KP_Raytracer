package calc

import (
	"math"
)

// DegreesToRadians converts degrees to radians.
func DegreesToRadians(degrees float64) float64 {
	return degrees * math.Pi / 180.0
}

// Clamp restricts a value to be within the specified range.
func Clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

// RandomFloat generates a pseudo-random float between 0 and 1 based on a mutable state.
func RandomFloat(state *uint32) float64 {
	// PCG Hash
	*state = *state*747796495 + 2891336453
	*state = ((*state >> ((*state >> 28) + 4)) ^ *state) * 277803737
	*state = (*state >> 22) ^ *state

	// Scale the output between 0 and 1
	scale := float64(0xffffffff)
	return float64(*state) / scale
}

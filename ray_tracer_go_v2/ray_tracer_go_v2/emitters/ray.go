package emitters

import "ray_tracer_go_v2/calc"

type Ray struct {
	Origin    calc.Point
	Direction calc.Vec3
}

// NewRay creates a new ray with the given origin and direction.
func NewRay(origin calc.Point, direction calc.Vec3) Ray {
	return Ray{
		Origin:    origin,
		Direction: direction,
	}
}

// At returns the point along the ray at distance t.
func (r Ray) At(t float64) calc.Point {
	return r.Origin.Add(r.Direction.MulScalar(t))
}

package common

import (
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/emitters"
)

type HitRecord struct {
	HitPoint  calc.Point
	Normal    calc.Vec3
	T         float64
	FrontFace bool
	Material  Material
}

func (hr *HitRecord) SetFaceNormal(ray emitters.Ray, outwardNormal calc.Vec3) {
	hr.FrontFace = calc.Dot(ray.Direction, outwardNormal) < 0
	if hr.FrontFace {
		hr.Normal = outwardNormal
	} else {
		hr.Normal = outwardNormal.Negate()
	}
}

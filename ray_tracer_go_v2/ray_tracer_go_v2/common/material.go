package common

import (
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/emitters"
)

// Material defines the behavior for materials in the ray tracer.
type Material interface {
	Scatter(rIn emitters.Ray, rec *HitRecord, attenuation *calc.Vec3, scattered *emitters.Ray) bool
}

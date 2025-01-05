package canvas

import (
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
)

type Intersectable interface {
	Hit(ray emitters.Ray, tMin, tMax float64, rec *common.HitRecord) bool
}

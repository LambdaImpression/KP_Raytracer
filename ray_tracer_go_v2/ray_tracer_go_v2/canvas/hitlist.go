// canvas/hitlist.go
package canvas

import (
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
)

type HitList struct {
	List []Intersectable
}

func NewHitList(list []Intersectable) *HitList {
	return &HitList{List: list}
}

func (hl *HitList) Hit(ray emitters.Ray, tMin, tMax float64, record *common.HitRecord) bool {
	var tempRecord common.HitRecord
	hitAnything := false
	closest := tMax

	for _, obj := range hl.List {
		if obj.Hit(ray, tMin, closest, &tempRecord) {
			hitAnything = true
			closest = tempRecord.T
			*record = tempRecord
		}
	}

	return hitAnything
}

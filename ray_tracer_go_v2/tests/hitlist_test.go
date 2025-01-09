package tests

import (
	"ray_tracer_go_v2/canvas"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
	"testing"
)

type MockIntersectable struct {
	hitReturn   bool
	hitRecord   common.HitRecord
	tMin, tMax  float64
	rayReceived emitters.Ray
}

func (mi *MockIntersectable) Hit(ray emitters.Ray, tMin, tMax float64, record *common.HitRecord) bool {
	mi.rayReceived = ray
	mi.tMin = tMin
	mi.tMax = tMax // Capture the updated tMax value
	if mi.hitReturn {
		*record = mi.hitRecord
	}
	return mi.hitReturn
}

func TestHitList_Hit_NoHits(t *testing.T) {
	hitList := canvas.NewHitList([]canvas.Intersectable{})
	ray := emitters.Ray{} // Assuming a default constructor for Ray
	record := common.HitRecord{}

	hit := hitList.Hit(ray, 0, 1, &record)
	if hit {
		t.Errorf("HitList.Hit() = true; want false when list is empty")
	}
}

func TestHitList_Hit_OneHit(t *testing.T) {
	mockObj := &MockIntersectable{
		hitReturn: true,
		hitRecord: common.HitRecord{
			T: 0.5,
		},
	}
	hitList := canvas.NewHitList([]canvas.Intersectable{mockObj})
	ray := emitters.Ray{} // Assuming a default constructor for Ray
	record := common.HitRecord{}

	hit := hitList.Hit(ray, 0, 1, &record)
	if !hit {
		t.Errorf("HitList.Hit() = false; want true when an object is hit")
	}
	if record.T != 0.5 {
		t.Errorf("HitList.Hit() record.T = %f; want 0.5", record.T)
	}
}

func TestHitList_Hit_ClosestHit(t *testing.T) {
	mockObj1 := &MockIntersectable{
		hitReturn: true,
		hitRecord: common.HitRecord{
			T: 0.7,
		},
	}
	mockObj2 := &MockIntersectable{
		hitReturn: true,
		hitRecord: common.HitRecord{
			T: 0.5,
		},
	}
	hitList := canvas.NewHitList([]canvas.Intersectable{mockObj1, mockObj2})
	ray := emitters.Ray{} // Assuming a default constructor for Ray
	record := common.HitRecord{}

	hit := hitList.Hit(ray, 0, 1, &record)
	if !hit {
		t.Errorf("HitList.Hit() = false; want true when at least one object is hit")
	}
	if record.T != 0.5 {
		t.Errorf("HitList.Hit() record.T = %f; want 0.5 for the closest hit", record.T)
	}
}

func TestHitList_Hit_NoIntersection(t *testing.T) {
	mockObj := &MockIntersectable{
		hitReturn: false,
	}
	hitList := canvas.NewHitList([]canvas.Intersectable{mockObj})
	ray := emitters.Ray{}
	record := common.HitRecord{}

	hit := hitList.Hit(ray, 0, 1, &record)
	if hit {
		t.Errorf("HitList.Hit() = true; want false when no object is hit")
	}
}

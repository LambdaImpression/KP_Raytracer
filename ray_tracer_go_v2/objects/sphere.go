package objects

import (
	"encoding/json"
	"math"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
	"ray_tracer_go_v2/material"
)

// Material is the concrete struct used for unmarshaling.
type Material struct {
	MaterialType    int        `json:"MaterialType"`
	Albedo          *calc.Vec3 `json:"Color,omitempty"`
	Fuzz            float64    `json:"Fuzz,omitempty"`
	RefractiveIndex float64    `json:"RefractiveIndex,omitempty"`
}

type Sphere struct {
	Center   calc.Point      `json:"Center"`
	Radius   float64         `json:"Radius"`
	Material common.Material `json:"-"`
}

func NewSphere(center calc.Vec3, radius float64, material common.Material) *Sphere {
	return &Sphere{
		Center:   center,
		Radius:   radius,
		Material: material,
	}
}

// Custom unmarshaler for Sphere to dynamically map materials.
func (s *Sphere) UnmarshalJSON(data []byte) error {
	type Alias Sphere
	aux := &struct {
		Material Material `json:"Material"`
		*Alias
	}{
		Alias: (*Alias)(s),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// Map Material to common.Material
	switch aux.Material.MaterialType {
	case 0: // Lambertian
		s.Material = material.NewLambertian(*aux.Material.Albedo)
	case 1: // Metal
		s.Material = material.NewMetal(*aux.Material.Albedo, aux.Material.Fuzz)
	case 2: // Dielectric
		s.Material = material.NewDielectric(aux.Material.RefractiveIndex)
	default:
		return nil // Default material
	}
	return nil
}

func (s *Sphere) Hit(r emitters.Ray, tMin, tMax float64, record *common.HitRecord) bool {
	originCenter := r.Origin.Sub(s.Center)
	a := r.Direction.LengthSquared()
	halfB := calc.Dot(originCenter, r.Direction)
	c := originCenter.LengthSquared() - s.Radius*s.Radius
	discriminant := halfB*halfB - a*c

	if discriminant < 0 {
		return false
	}

	sqrtDiscriminant := math.Sqrt(discriminant)

	// Find the nearest root that lies in the acceptable range.
	root := (-halfB - sqrtDiscriminant) / a
	if root < tMin || root > tMax {
		root = (-halfB + sqrtDiscriminant) / a
		if root < tMin || root > tMax {
			return false
		}
	}

	record.T = root
	record.HitPoint = r.At(record.T)
	outwardNormal := record.HitPoint.Sub(s.Center).DivScalar(s.Radius)
	record.SetFaceNormal(r, outwardNormal)
	record.Material = s.Material
	return true
}

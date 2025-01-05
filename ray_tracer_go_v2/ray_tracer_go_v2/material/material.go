package material

import (
	"math"
	"math/rand"

	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/common"
	"ray_tracer_go_v2/emitters"
)

// Reflect calculates the reflection of a vector.
func Reflect(v, n calc.Vec3) calc.Vec3 {
	return v.Sub(n.MulScalar(2 * calc.Dot(v, n)))
}

// Refract calculates the refraction of a vector.
func Refract(v, n calc.Vec3, niOverNt float64) (calc.Vec3, bool) {
	uv := v.Normalize()
	dt := calc.Dot(uv, n)
	discriminant := 1.0 - niOverNt*niOverNt*(1-dt*dt)
	if discriminant > 0 {
		refracted := uv.Sub(n.MulScalar(dt)).MulScalar(niOverNt).Sub(n.MulScalar(math.Sqrt(discriminant)))
		return refracted, true
	}
	return calc.Vec3{}, false
}

// Schlick calculates the reflection probability using Schlick's approximation.
func Schlick(cosine, refIdx float64) float64 {
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*math.Pow(1-cosine, 5)
}

// Lambertian Material
type Lambertian struct {
	Albedo calc.Vec3
}

func NewLambertian(albedo calc.Vec3) *Lambertian {
	return &Lambertian{Albedo: albedo}
}

func (l Lambertian) Scatter(rIn emitters.Ray, rec *common.HitRecord, attenuation *calc.Vec3, scattered *emitters.Ray) bool {

	scatterDirection := rec.Normal.Add(calc.RandomUnitVector())

	if scatterDirection.NearZero() {
		scatterDirection = rec.Normal
	}
	*scattered = emitters.NewRay(rec.HitPoint, scatterDirection)
	*attenuation = l.Albedo

	return true
}

// Metal Material
type Metal struct {
	Albedo calc.Vec3
	Fuzz   float64
}

func NewMetal(albedo calc.Vec3, fuzz float64) *Metal {
	if fuzz > 1 {
		fuzz = 1
	}
	return &Metal{Albedo: albedo, Fuzz: fuzz}
}

func (m Metal) Scatter(rIn emitters.Ray, rec *common.HitRecord, attenuation *calc.Vec3, scattered *emitters.Ray) bool {
	reflected := Reflect(rIn.Direction.Normalize(), rec.Normal)
	scatteredDirection := reflected.Add(calc.RandomInUnitSphere().MulScalar(m.Fuzz))
	*scattered = emitters.NewRay(rec.HitPoint, scatteredDirection)
	*attenuation = m.Albedo
	return calc.Dot(scattered.Direction, rec.Normal) > 0
}

// Dielectric Material
type Dielectric struct {
	RefIdx float64
}

func NewDielectric(refIdx float64) *Dielectric {
	return &Dielectric{RefIdx: refIdx}
}

func (d Dielectric) Scatter(rIn emitters.Ray, rec *common.HitRecord, attenuation *calc.Vec3, scattered *emitters.Ray) bool {
	var outwardNormal calc.Vec3
	var niOverNt float64
	var cosine float64
	*attenuation = calc.NewVec3(1.0, 1.0, 1.0)

	if calc.Dot(rIn.Direction, rec.Normal) > 0 {
		outwardNormal = rec.Normal.Negate()
		niOverNt = d.RefIdx
		cosine = d.RefIdx * calc.Dot(rIn.Direction, rec.Normal) / rIn.Direction.Length()
	} else {
		outwardNormal = rec.Normal
		niOverNt = 1.0 / d.RefIdx
		cosine = -calc.Dot(rIn.Direction, rec.Normal) / rIn.Direction.Length()
	}

	refracted, success := Refract(rIn.Direction, outwardNormal, niOverNt)

	reflectProb := 1.0
	if success {
		reflectProb = Schlick(cosine, d.RefIdx)
	}

	if rand.Float64() < reflectProb {
		reflected := Reflect(rIn.Direction, rec.Normal)
		*scattered = emitters.NewRay(rec.HitPoint, reflected)
	} else {
		*scattered = emitters.NewRay(rec.HitPoint, refracted)
	}

	return true
}

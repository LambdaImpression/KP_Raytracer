package emitters

import (
	"math"
	"ray_tracer_go_v2/calc"
)

type Camera struct {
	Origin          calc.Point
	LowerLeftCorner calc.Point
	Horizontal      calc.Vec3
	Vertical        calc.Vec3
	U, V, W         calc.Vec3
	LensRadius      float64
}

func NewCamera(lookFrom, lookAt calc.Point, vUp calc.Vec3, vFov, aspectRatio, aperture, focusDist float64) *Camera {
	theta := calc.DegreesToRadians(vFov)
	h := math.Tan(theta / 2)
	viewportHeight := 2.0 * h
	viewportWidth := aspectRatio * viewportHeight

	w := lookFrom.Sub(lookAt).Normalize()
	u := calc.Cross(vUp, w).Normalize()
	v := calc.Cross(w, u)

	origin := lookFrom
	horizontal := u.MulScalar(viewportWidth * focusDist)
	vertical := v.MulScalar(viewportHeight * focusDist)
	lowerLeftCorner := origin.Sub(horizontal.DivScalar(2)).Sub(vertical.DivScalar(2)).Sub(w.MulScalar(focusDist))

	return &Camera{
		Origin:          origin,
		LowerLeftCorner: lowerLeftCorner,
		Horizontal:      horizontal,
		Vertical:        vertical,
		U:               u,
		V:               v,
		W:               w,
		LensRadius:      aperture / 2,
	}
}

func (c *Camera) GetRay(u, v float64) Ray {
	rd := calc.RandomInUnitDisk().MulScalar(c.LensRadius)
	offset := c.U.MulScalar(rd.X).Add(c.V.MulScalar(rd.Y))
	return NewRay(
		c.Origin.Add(offset),
		c.LowerLeftCorner.Add(c.Horizontal.MulScalar(u)).Add(c.Vertical.MulScalar(v)).Sub(c.Origin).Sub(offset),
	)
}

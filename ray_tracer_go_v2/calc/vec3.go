package calc

import (
	"fmt"
	"math"
	"math/rand"
)

type Vec3 struct {
	X, Y, Z float64
}

// Constructors
func NewVec3(x, y, z float64) Vec3 {
	return Vec3{x, y, z}
}

func ZeroVec3() Vec3 {
	return Vec3{0, 0, 0}
}

// Accessors
func (v Vec3) GetX() float64 { return v.X }
func (v Vec3) GetY() float64 { return v.Y }
func (v Vec3) GetZ() float64 { return v.Z }

// Negation
func (v Vec3) Negate() Vec3 {
	return Vec3{-v.X, -v.Y, -v.Z}
}

// Arithmetic Operations
func (v Vec3) Add(other Vec3) Vec3 {
	return Vec3{
		X: v.X + other.X,
		Y: v.Y + other.Y,
		Z: v.Z + other.Z,
	}
}

func (v Vec3) Sub(other Vec3) Vec3 {
	return Vec3{
		X: v.X - other.X,
		Y: v.Y - other.Y,
		Z: v.Z - other.Z,
	}
}

func (v Vec3) Mul(other Vec3) Vec3 {
	return Vec3{
		X: v.X * other.X,
		Y: v.Y * other.Y,
		Z: v.Z * other.Z,
	}
}

func (v Vec3) MulScalar(t float64) Vec3 {
	return Vec3{
		X: v.X * t,
		Y: v.Y * t,
		Z: v.Z * t,
	}
}

// Vector Length and Normalization
func (v Vec3) LengthSquared() float64 {
	return v.X*v.X + v.Y*v.Y + v.Z*v.Z
}

// Normalized returns a new vector that is the unit vector of the current vector.
func (v Vec3) Normalize() Vec3 {
	length := v.Length()
	if length == 0 {
		return Vec3{0, 0, 0} // Avoid division by zero
	}
	return v.DivScalar(length)
}

// Length and other utility methods remain the same
func (v Vec3) Length() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y + v.Z*v.Z)
}

func (v Vec3) DivScalar(t float64) Vec3 {
	return Vec3{v.X / t, v.Y / t, v.Z / t}
}

// Utility Functions
func Dot(a, b Vec3) float64 {
	return a.X*b.X + a.Y*b.Y + a.Z*b.Z
}

func Cross(a, b Vec3) Vec3 {
	return Vec3{
		X: a.Y*b.Z - a.Z*b.Y,
		Y: a.Z*b.X - a.X*b.Z,
		Z: a.X*b.Y - a.Y*b.X,
	}
}

func (v Vec3) NearZero() bool {
	const eps = 1e-8
	return math.Abs(v.X) < eps && math.Abs(v.Y) < eps && math.Abs(v.Z) < eps
}

func (v Vec3) String() string {
	return fmt.Sprintf("(%f, %f, %f)", v.X, v.Y, v.Z)
}

func CurandUniformEquivalent() float64 {
	return 1.0 - rand.Float64() // Shift range from [0, 1) to (0, 1]
}

func RandomVec3() Vec3 {
	return Vec3{
		X: CurandUniformEquivalent(),
		Y: CurandUniformEquivalent(),
		Z: CurandUniformEquivalent(),
	}
}

func RandomInUnitSphere() Vec3 {
	for {
		p := RandomVec3().MulScalar(2.0).Sub(NewVec3(1.0, 1.0, 1.0))
		if p.LengthSquared() < 1.0 {
			return p
		}
	}
}

func RandomUnitVector() Vec3 {
	return RandomInUnitSphere().Normalize()
}

func RandomInUnitDisk() Vec3 {
	for {
		p := Vec3{
			X: CurandUniformEquivalent()*2.0 - 1.0,
			Y: CurandUniformEquivalent()*2.0 - 1.0,
			Z: 0.0,
		}
		if p.LengthSquared() < 1.0 {
			return p
		}
	}
}

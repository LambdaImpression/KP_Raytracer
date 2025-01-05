package tests

import (
	"math"
	"ray_tracer_go_v2/calc"
	"testing"
)

func TestNewVec3(t *testing.T) {
	v := calc.NewVec3(1, 2, 3)
	if v.X != 1 || v.Y != 2 || v.Z != 3 {
		t.Errorf("NewVec3(1, 2, 3) = %v; want Vec3{1, 2, 3}", v)
	}
}

func TestZeroVec3(t *testing.T) {
	v := calc.ZeroVec3()
	if v.X != 0 || v.Y != 0 || v.Z != 0 {
		t.Errorf("ZeroVec3() = %v; want Vec3{0, 0, 0}", v)
	}
}

func TestAccessors(t *testing.T) {
	v := calc.NewVec3(1, 2, 3)
	if v.GetX() != 1 || v.GetY() != 2 || v.GetZ() != 3 {
		t.Errorf("Accessors failed on Vec3{1, 2, 3}")
	}
}

func TestNegate(t *testing.T) {
	v := calc.NewVec3(1, -2, 3)
	neg := v.Negate()
	want := calc.Vec3{-1, 2, -3}
	if neg != want {
		t.Errorf("Negate() = %v; want %v", neg, want)
	}
}

func TestAdd(t *testing.T) {
	v1 := calc.NewVec3(1, 2, 3)
	v2 := calc.NewVec3(4, 5, 6)
	sum := v1.Add(v2)
	want := calc.Vec3{5, 7, 9}
	if sum != want {
		t.Errorf("Add() = %v; want %v", sum, want)
	}
}

func TestSub(t *testing.T) {
	v1 := calc.NewVec3(5, 7, 9)
	v2 := calc.NewVec3(1, 2, 3)
	diff := v1.Sub(v2)
	want := calc.Vec3{4, 5, 6}
	if diff != want {
		t.Errorf("Sub() = %v; want %v", diff, want)
	}
}

func TestMul(t *testing.T) {
	v1 := calc.NewVec3(1, 2, 3)
	v2 := calc.NewVec3(4, 5, 6)
	product := v1.Mul(v2)
	want := calc.Vec3{4, 10, 18}
	if product != want {
		t.Errorf("Mul() = %v; want %v", product, want)
	}
}

func TestMulScalar(t *testing.T) {
	v := calc.NewVec3(1, 2, 3)
	scalar := 2.5
	product := v.MulScalar(scalar)
	want := calc.Vec3{2.5, 5, 7.5}
	if product != want {
		t.Errorf("MulScalar() = %v; want %v", product, want)
	}
}

func TestLength(t *testing.T) {
	v := calc.NewVec3(3, 4, 0)
	if v.Length() != 5 {
		t.Errorf("Length() = %f; want %f", v.Length(), 5.0)
	}
}

func TestLengthSquared(t *testing.T) {
	v := calc.NewVec3(3, 4, 0)
	if v.LengthSquared() != 25 {
		t.Errorf("LengthSquared() = %f; want %f", v.LengthSquared(), 25.0)
	}
}

func TestNormalize(t *testing.T) {
	v := calc.NewVec3(3, 4, 0)
	unit := v.Normalize()
	if math.Abs(unit.Length()-1) > 1e-8 {
		t.Errorf("Normalize() resulted in a vector with length %f; want 1", unit.Length())
	}
}

func TestDivScalar(t *testing.T) {
	v := calc.NewVec3(3, 6, 9)
	div := v.DivScalar(3)
	want := calc.Vec3{1, 2, 3}
	if div != want {
		t.Errorf("DivScalar() = %v; want %v", div, want)
	}
}

func TestDot(t *testing.T) {
	v1 := calc.NewVec3(1, 2, 3)
	v2 := calc.NewVec3(4, -5, 6)
	dot := calc.Dot(v1, v2)
	want := 12.0
	if dot != want {
		t.Errorf("Dot() = %f; want %f", dot, want)
	}
}

func TestCross(t *testing.T) {
	v1 := calc.NewVec3(1, 2, 3)
	v2 := calc.NewVec3(4, 5, 6)
	cross := calc.Cross(v1, v2)
	want := calc.Vec3{-3, 6, -3}
	if cross != want {
		t.Errorf("Cross() = %v; want %v", cross, want)
	}
}

func TestNearZero(t *testing.T) {
	v := calc.NewVec3(1e-9, -1e-9, 0)
	if !v.NearZero() {
		t.Errorf("NearZero() = false; want true for %v", v)
	}
}

func TestRandomVec3(t *testing.T) {
	v := calc.RandomVec3()
	if v.X < 0 || v.X > 1 || v.Y < 0 || v.Y > 1 || v.Z < 0 || v.Z > 1 {
		t.Errorf("RandomVec3() = %v; components should be in range [0, 1)", v)
	}
}

func TestRandomInUnitSphere(t *testing.T) {
	v := calc.RandomInUnitSphere()
	if v.LengthSquared() >= 1 {
		t.Errorf("RandomInUnitSphere() = %v; LengthSquared() = %f; want < 1", v, v.LengthSquared())
	}
}

func TestRandomUnitVector(t *testing.T) {
	v := calc.RandomUnitVector()
	if math.Abs(v.Length()-1) > 1e-8 {
		t.Errorf("RandomUnitVector() = %v; Length() = %f; want 1", v, v.Length())
	}
}

func TestRandomInUnitDisk(t *testing.T) {
	v := calc.RandomInUnitDisk()
	if v.LengthSquared() >= 1 || v.Z != 0 {
		t.Errorf("RandomInUnitDisk() = %v; LengthSquared() = %f; want < 1 and Z = 0", v, v.LengthSquared())
	}
}

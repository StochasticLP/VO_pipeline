#pragma once
#include <array>
#include <iostream>

class Vector3 {
public:
    std::array<double, 3> data;

    Vector3(); // zero vector
    Vector3(double x, double y, double z);

    // TODO: implement these
    Vector3 operator+(const Vector3& other) const;
    Vector3 operator-(const Vector3& other) const;
    Vector3 operator*(double scalar) const;
    double dot(const Vector3& other) const;
    Vector3 cross(const Vector3& other) const;
    double norm() const;
    Vector3 normalized() const;

    // Accessor
    double& operator[](int i) { return data[i]; }
    double  operator[](int i) const { return data[i]; }

    friend std::ostream& operator<<(std::ostream& os, const Vector3& v);
};
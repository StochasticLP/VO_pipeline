#include "vector3.hpp"
#include <cmath>
#include <stdexcept>

Vector3::Vector3() : data{{0.0, 0.0, 0.0}} {}

Vector3::Vector3(double x, double y, double z) : data{x, y, z} {}

Vector3 Vector3::operator+(const Vector3& other) const {
    return Vector3(data[0] + other.data[0],
        data[1] + other.data[1],
        data[2] + other.data[2]);
}

Vector3 Vector3::operator-(const Vector3& other) const {
    return Vector3(data[0] - other.data[0],
        data[1] - other.data[1],
        data[2] - other.data[2]);
}

Vector3 Vector3::operator*(double scalar) const {
    return Vector3(scalar * data[0],
        scalar * data[1],
        scalar * data[2]);
}

double Vector3::dot(const Vector3& other) const {
    return data[0] * other.data[0] +
        data[1] * other.data[1] +
        data[2] * other.data[2];
}

Vector3 Vector3::cross(const Vector3& other) const {
    // TODO: the cross product formula
    Vector3 result;
    result[0] = data[1]*other[2] - data[2]*other[1];
    result[1] = data[2]*other[0] - data[0]*other[2];
    result[2] = data[0]*other[1] - data[1]*other[0];

    return result;
}

double Vector3::norm() const {
    return std::sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2]);
}

Vector3 Vector3::normalized() const {
    double n = norm();
    if (n < 1e-10) throw std::runtime_error("Cannot normalize zero vector");
    return *this * (1.0/n);
}

std::ostream& operator<<(std::ostream& os, const Vector3& v) {
    os << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return os;
}
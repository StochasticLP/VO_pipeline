#pragma once
#include "vector3.hpp"
#include <array>
#include <iostream>

class Matrix3 {
public:
    std::array<std::array<double, 3>, 3> data;  // row-major: data[row][col]

    Matrix3();                          // zero matrix
    Matrix3(std::array<std::array<double, 3>, 3> values);

    static Matrix3 identity();

    double& operator()(int row, int col) { return data[row][col]; }
    double  operator()(int row, int col) const { return data[row][col]; }

    // TODO: implement these
    Matrix3 operator+(const Matrix3& other) const;
    Matrix3 operator*(const Matrix3& other) const;
    Vector3 operator*(const Vector3& v) const;  // matrix-vector product
    Matrix3 transpose() const;
    double  determinant() const;

    friend std::ostream& operator<<(std::ostream& os, const Matrix3& m);
};
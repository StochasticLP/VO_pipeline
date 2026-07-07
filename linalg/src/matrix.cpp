#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include "vector3.hpp"

Matrix3::Matrix3() : data{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}} {}

Matrix3::Matrix3(std::array<std::array<double, 3>, 3> values) : data{values} {}

Matrix3 Matrix3::identity() {
	return Matrix3({{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}});
}


Matrix3 Matrix3::operator+(const Matrix3& other) const {
    Matrix3 sumMatrix;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sumMatrix(i, j) = data[i][j] + other(i, j);
        }
    }

    return sumMatrix;
}

Matrix3 Matrix3::operator*(const Matrix3& other) const {
	Matrix3 result;

	for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
        	result(i, j) = data[i][0] * other(0, j) +
        		data[i][1] * other(1, j) +
        		data[i][2] * other(2, j);
        }
    }

    return result;
}

Vector3 Matrix3::operator*(const Vector3& v) const{ // matrix-vector product
	Vector3 result;

	for (int i = 0; i < 3; i++) {
		result[i] = data[i][0]*v[i];
	}

	return result;
}

Matrix3 Matrix3::transpose() const {
	Matrix3 t;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; ++j) {
			t(i, j) = data[j][i];
		}
	}

	return t;
}


double Matrix3::determinant() const {
	return data[0][0] * (data[1][1]*data[2][2] - data[1][2]*data[2][1]) -
    	data[0][1] * (data[1][0]*data[2][2] - data[1][2]*data[2][0]) +
    	data[0][2] * (data[1][0]*data[2][1] - data[1][1]*data[2][0]);
}
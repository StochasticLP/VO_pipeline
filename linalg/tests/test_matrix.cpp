#include <gtest/gtest.h>
#include "matrix.hpp"

TEST(Matrix3Test, Identity) {
    Matrix3 I = Matrix3::identity();
    Vector3 v(1, 2, 3);
    Vector3 result = I * v;
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST(Matrix3Test, Transpose) {
    // TODO: build a non-symmetric matrix, transpose it, check result
}

TEST(Matrix3Test, MatMulIdentity) {
    // TODO: A * I should equal A
}
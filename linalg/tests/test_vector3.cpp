#include <gtest/gtest.h>
#include "vector3.hpp"
#include <cmath>

TEST(Vector3Test, Addition) {
    Vector3 a(1, 2, 3);
    Vector3 b(4, 5, 6);
    Vector3 c = a + b;
    EXPECT_DOUBLE_EQ(c[0], 5.0);
    EXPECT_DOUBLE_EQ(c[1], 7.0);
    EXPECT_DOUBLE_EQ(c[2], 9.0);
}

TEST(Vector3Test, DotProduct) {
    Vector3 a(1, 0, 0);
    Vector3 b(0, 1, 0);
    EXPECT_DOUBLE_EQ(a.dot(b), 0.0);  // perpendicular vectors
}

TEST(Vector3Test, CrossProduct) {
    Vector3 x(1, 0, 0);
    Vector3 y(0, 1, 0);
    Vector3 z = x.cross(y);
    EXPECT_DOUBLE_EQ(z[0], 0.0);
    EXPECT_DOUBLE_EQ(z[1], 0.0);
    EXPECT_DOUBLE_EQ(z[2], 1.0);  // should be z-hat
}

TEST(Vector3Test, Normalization) {
    Vector3 v(3, 0, 0);
    Vector3 n = v.normalized();
    EXPECT_NEAR(n.norm(), 1.0, 1e-10);
}

// TODO: add tests for subtraction, scalar multiply, edge cases
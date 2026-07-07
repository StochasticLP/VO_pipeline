#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include "matrix.hpp"
#include "vector3.hpp"

int main() {
    // --- Your implementation ---
    auto start = std::chrono::high_resolution_clock::now();

    Vector3 a(1.0, 2.0, 3.0);
    Vector3 b(4.0, 5.0, 6.0);
    double result_yours = 0;
    for (int i = 0; i < 1000000; i++) {
        result_yours += a.dot(b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms_yours = std::chrono::duration<double, std::milli>(end - start).count();

    // --- Eigen ---
    start = std::chrono::high_resolution_clock::now();

    Eigen::Vector3d ea(1.0, 2.0, 3.0);
    Eigen::Vector3d eb(4.0, 5.0, 6.0);
    double result_eigen = 0;
    for (int i = 0; i < 1000000; i++) {
        result_eigen += ea.dot(eb);
    }

    end = std::chrono::high_resolution_clock::now();
    double ms_eigen = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Yours:  " << ms_yours << " ms (result=" << result_yours << ")\n";
    std::cout << "Eigen:  " << ms_eigen << " ms (result=" << result_eigen << ")\n";
    std::cout << "Ratio:  " << ms_yours / ms_eigen << "x slower\n";

    return 0;
}
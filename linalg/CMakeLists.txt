cmake_minimum_required(VERSION 3.14)
project(linalg)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
include(GoogleTest)

# Your library
add_library(linalg_lib
    src/matrix.cpp
    src/vector3.cpp
)
target_include_directories(linalg_lib PUBLIC include)
target_link_libraries(linalg_lib Eigen3::Eigen)

# Tests
add_executable(run_tests
    tests/test_matrix.cpp
    tests/test_vector3.cpp
)
target_link_libraries(run_tests linalg_lib GTest::GTest GTest::Main)
gtest_discover_tests(run_tests)

# Benchmark
add_executable(bench benchmarks/bench.cpp)
target_link_libraries(bench linalg_lib)
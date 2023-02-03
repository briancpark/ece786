cmake_minimum_required(VERSION 3.11)
project(proj1a)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(quamsim quantumSimulation.cu)
target_compile_features(quamsim PRIVATE cxx_std_11)

# TODO: This is architecture specific, enable later
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}") # -arch=sm_60

file(COPY input DESTINATION .)
file(COPY test.sh DESTINATION .)
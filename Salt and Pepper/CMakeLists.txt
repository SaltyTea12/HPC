cmake_minimum_required(VERSION 3.10)
project(SaltAndPepper)

find_package(CUDA 11.8 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(EasyBMP)

# Принудительная установка архитектуры для GeForce 940MX
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_50)

CUDA_ADD_EXECUTABLE(SaltAndPepper 
    Salt_and_Pepper.cu
    EasyBMP/EasyBMP.cpp
)

target_link_libraries(SaltAndPepper ${CUDA_LIBRARIES})

set_target_properties(SaltAndPepper PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)
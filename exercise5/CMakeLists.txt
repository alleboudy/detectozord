cmake_minimum_required(VERSION 2.8.7)
project(Praktikum_PL)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

add_compile_options(-std=c++11)

###########################################################################################
# Find modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

## OpenCV
find_package(OpenCV 3.0 REQUIRED)
if(OpenCV_INCLUDE_DIRS)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
endif(OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_INCLUDE_DIR)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_INCLUDE_DIR)

## OpenNI2
find_package(openni2)
if(OPENNI2_FOUND)
	include_directories(${OPENNI2_INCLUDE_DIR})
endif(OPENNI2_FOUND)

add_definitions( -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"  )

###########################################################################################
# TARGETS
ADD_SUBDIRECTORY(Exercise05_1)
ADD_SUBDIRECTORY(Exercise05_2)

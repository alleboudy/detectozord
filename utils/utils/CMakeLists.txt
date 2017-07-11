
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)



set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type default to Release" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

###########################################################################################
# TARGETS

include_directories(${PROJECT_SOURCE_DIR}/TSDFLib)
link_directories(${LIBRARY_OUTPUT_PATH})

SET(SRCS main.cpp)

INSEG_ADD_EXECUTABLE(utils FILES ${SRCS} LINK_WITH 
${OpenCV_LIBS} 
${PCL_LIBRARIES} 
)



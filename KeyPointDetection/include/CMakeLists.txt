add_library(Pyramid src/GaussPyramid/GaussPyramid.hpp src/GaussPyramid/GaussPyramid.cpp)
target_include_directories(Pyramid PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(Rotation src/Rotation/rotation.h src/Rotation/rotation.cpp)
target_include_directories(Rotation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

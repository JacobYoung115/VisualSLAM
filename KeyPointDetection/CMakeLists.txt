cmake_minimum_required(VERSION 2.8)
project( CornerDetection )
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_subdirectory(include)
add_subdirectory(tests)


add_executable( DoG Diff_of_Gauss.cpp)
target_link_libraries( DoG ${OpenCV_LIBS} Rotation Pyramid)

add_executable( Harris Harris_corners.cpp)
target_link_libraries( Harris ${OpenCV_LIBS})

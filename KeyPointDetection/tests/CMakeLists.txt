add_executable( Pyramid_Test GaussPyramid_Test.cpp)
target_link_libraries( Pyramid_Test ${OpenCV_LIBS} Pyramid)

add_executable( RotateImgTest rotate_image_test.cpp)
target_link_libraries( RotateImgTest ${OpenCV_LIBS} Rotation)
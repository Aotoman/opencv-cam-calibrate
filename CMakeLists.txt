cmake_minimum_required(VERSION 3.14)
project(opencv_cam_calibrate)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV 3.1.0 REQUIRED )


add_executable(opencv_cam_calibrate main.cpp)

target_link_libraries(opencv_cam_calibrate ${OpenCV_LIBS} )


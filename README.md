# VisualSLAM
The purpose of this project is to demonstrate understanding of SLAM concepts and techniques by implementing them in from (mostly) scratch in C++.

SLAM or _**S**imultaneous **L**ocalization **A**nd **M**apping_ is a set of algorithms used in Robot Navigation, Robotic Mapping, Virtual and Augmented Reality. It works by constructing or updating a map of an unknown environment while simultaneously keeping track of the robot (or user's) location within the map (1).

The _Visual_ in VisualSLAM refers to using images, rather than laser scans as the basis for creation of the environment map and for understanding the robot's location within it.


## Breakdown
VisualSLAM is typically broken down into a few steps.
Each of these steps can be performed using different algorithms, I explore a few in this project:

### 1. Keypoint detection
..* Harris Corner Detection
..* Difference of Gaussian

### 2. Feature description / Image Matching
..* SIFT
..* ORB

### 3. 3D Reconstruction
..* Epipolar Geometry

Ultimately these steps are tied together to create a SLAM solution.


## Requirements
This project has the following depencies:
..* Linux - Ubuntu 22.04
..* g++ Compiler (version (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0)
..* OpenCV C++ (I use version 4.7.0-dev, but versions 3.5+ should work)
..* cmake (version 3.22.1)
..* make (GNU Make 4.3)
..* git (version 2.34.1)

Note that different versions of these depencies may work just fine.


## Building and running the project
Verify that you have all requirements satisfied, then start by cloning the project to a preferred location.
```
cd ~/Documents
git clone https://github.com/JacobYoung115/VisualSLAM.git
cd VisualSLAM/KeyPointDetection
```

Make a build directory and build the project:
```
mkdir build
cd build
cmake ..
make
```

Tests that the executables built properly and are working by running:
```
./DoG
./Harris
./tests/Pyramid_Test
./tests/RotateImgTest
```

## References
1. [Wikipedia SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping)
2. [Modern C++ for Computer Vision - 2021] (https://www.ipb.uni-bonn.de/teaching/cpp-2021/lectures/)
3. [Modern C++ for CV - 2018] (https://www.ipb.uni-bonn.de/teaching/modern-cpp/#slides)
4. [Mobile Sensing and Robotics 2], 2021 - University of Bonn (https://www.youtube.com/playlist?list=PLgnQpQtFTOGQh_J16IMwDlji18SWQ2PZ6)
5. SIFT - [Distinctive Image Features from Scale-Invariant Keypoints] 2004 - David G. Lowe (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
6. CMake - ["How to CMake Good"] (https://www.youtube.com/watch?v=_yFPO1ofyF0&list=PLK6MXr8gasrGmIiSuVQXpfFuE1uPT615s)

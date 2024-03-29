# VisualSLAM
The purpose of this project is to demonstrate understanding of SLAM concepts and techniques by implementing them in from (mostly) scratch in C++.

SLAM or _**S**imultaneous **L**ocalization **A**nd **M**apping_ is a set of algorithms used in Robot Navigation, Robotic Mapping, Virtual and Augmented Reality. It works by constructing or updating a map of an unknown environment while simultaneously keeping track of the robot (or user's) location within the map (1).

The _Visual_ in VisualSLAM refers to using images, rather than laser scans as the basis for creation of the environment map and for understanding the robot's location within it.


## Breakdown
VisualSLAM is typically broken down into a few steps.
Each of these steps can be performed using different algorithms, I explore a few in this project:

### 1. Keypoint detection
* [Harris Corner Detection](http://www.bmva.org/bmvc/1988/avc-88-023.pdf)
* [Difference of Gaussian](https://en.wikipedia.org/wiki/Difference_of_Gaussians)

### 2. Feature description / Image Matching
* [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)(5)
* [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF)

### 3. 3D Reconstruction
* [Epipolar Geometry](https://en.wikipedia.org/wiki/Epipolar_geometry)

Ultimately these steps are tied together to create a SLAM solution.


## Requirements
This project is written in C++ 17 and should be compatible with any C++ 17 compiler. It uses the most recently available versions of the following dependencies:
* [Linux](https://ubuntu.com/download/desktop) - Ubuntu 22.04
* [GCC](http://gcc.gnu.org/) (version (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0)
* [OpenCV C++](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) (I use version 4.7.0-dev, but versions 3.5+ should work)
* [CMake](https://cmake.org/) (version 3.22.1)
* make (GNU Make 4.3)
* [Git](https://git-scm.com/) (version 2.34.1)

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
2. [Modern C++ for Computer Vision - 2021](https://www.ipb.uni-bonn.de/teaching/cpp-2021/lectures/)
3. [Modern C++ for CV - 2018](https://www.ipb.uni-bonn.de/teaching/modern-cpp/#slides)
4. [Mobile Sensing and Robotics 2 - 2021](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQh_J16IMwDlji18SWQ2PZ6)
5. [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
6. [CMake tutorials](https://www.youtube.com/watch?v=_yFPO1ofyF0&list=PLK6MXr8gasrGmIiSuVQXpfFuE1uPT615s)

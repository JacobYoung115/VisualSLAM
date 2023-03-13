#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>

using namespace cv;

class GaussPyramid
{
    public:
        GaussPyramid(Mat& img, int numOctaves, float sigma) : numOctaves_{numOctaves}, sigma_{sigma} { this->createPyramid(img); } 
        const std::vector<Mat>& imagePyramid() { return this->img_pyramid; }
        const std::map<int, std::vector<Mat>>& gaussPyramid() { return this->gauss_pyramid; }
        const std::map<int, std::vector<Mat>>& diffPyramid() { return this->diff_pyramid; }
        const Mat& getOctave(int octave) { return this->img_pyramid.at(octave); }
        const std::vector<Mat>& getBlurOctave(int octave) { return this->gauss_pyramid.at(octave); }
        const std::vector<Mat>& getDiffOctave(int octave) { return this->diff_pyramid.at(octave); }
        static void displayPyramid(const std::map<int, std::vector<Mat>> pyramid);
        static void showOctave(const std::vector<Mat> images, const std::string window_name, const Point pos = Point(0,0));
        //lastly, we need the scale (i.e. sigma value) for every level of each octave
    private:
        void createPyramid(Mat& img);
        std::vector<Mat> GaussVector(Mat& img);
        std::vector<Mat> Diff_of_Gauss(std::vector<Mat> gaussians);
        std::vector<Mat> img_pyramid;
        std::map<int, std::vector<Mat>> gauss_pyramid;
        std::map<int, std::vector<Mat>> diff_pyramid;
        int numOctaves_ = 0;
        int numImages_ = numOctaves_ + 3;
        float sigma_ = 0.0f;
        float k_ = pow(2.0f, 1.0f/float(numOctaves_));
};
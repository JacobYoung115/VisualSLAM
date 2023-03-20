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
        GaussPyramid(Mat& img, int numOctaves, double sigma) : numOctaves_{numOctaves}, sigma_{sigma} { this->createPyramid(img); }
        GaussPyramid(Mat& img, double sigma) : sigma_{sigma} {
            this->calculateNumOctaves(img);
            this->createPyramid(img);
        } 
        const std::vector<Mat>& imagePyramid() { return this->img_pyramid; }
        const std::map<int, std::vector<Mat>>& gaussPyramid() { return this->gauss_pyramid; }
        const std::map<int, std::vector<Mat>>& diffPyramid() { return this->diff_pyramid; }
        const Mat& getOctave(int octave) { return this->img_pyramid.at(octave); }
        const int getNumOctaves();
        const int getNumLevels();
        const double getSigmaAt(int octave, int level);
        const std::vector<Mat>& getBlurOctave(int octave) { return this->gauss_pyramid.at(octave); }
        const std::vector<Mat>& getDiffOctave(int octave) { return this->diff_pyramid.at(octave); }
        const std::vector<double>& getSigmaValues() { return this->sigmas_; }
        static void displayPyramid(const std::map<int, std::vector<Mat>> pyramid);
        static void showOctave(const std::vector<Mat> images, const std::string window_name, const Point pos = Point(0,0));
        //lastly, we need the scale (i.e. sigma value) for every level of each octave
        //create a way to access the sigma value of the current image, given the octave & level.
    private:
        int calculateNumOctaves(Mat& img);
        double calculateSigma(int level);
        void createPyramid(Mat& img);
        std::vector<double> sigmas_;
        std::vector<Mat> GaussVector(Mat& img);
        std::vector<Mat> Diff_of_Gauss(std::vector<Mat> gaussians);
        std::vector<Mat> img_pyramid;
        std::map<int, std::vector<Mat>> gauss_pyramid;
        std::map<int, std::vector<Mat>> diff_pyramid;
        std::map<int, std::vector<double>> sigmas_pyramid;
        int numOctaves_ = 0;
        int scaleSamples_ = 3;
        int numLevels_ = scaleSamples_ + 3;
        int currentOctave_ = 0;
        double sigma_ = 0.0f;
        double k_ = pow(2.0f, 1.0f/double(scaleSamples_));
        //see: https://www.researchgate.net/profile/Michael-Veth/publication/216789729/figure/fig12/AS:394046162915328@1470959334135/Octaves-of-the-Difference-of-Gaussian-Functions-over-a-Scale-Space-Gaussian-blurred.png
};
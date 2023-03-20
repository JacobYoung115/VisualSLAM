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
        //member variable getters
        const int getNumOctaves() { return this->numOctaves_; }
        const int getNumLevels() { return this->numLevels_; }
        //octave getters.
        const double getSigmaAt(int octave, int level);
        const Mat& octaveImage(int octave) { return this->img_pyramid.at(octave); }
        const std::vector<double>& octaveSigma(int octave) { return this->sigmas_pyramid.at(octave); }
        const std::vector<Mat>& octaveBlur(int octave) { return this->gauss_pyramid.at(octave); }
        const std::vector<Mat>& octaveDiff(int octave) { return this->diff_pyramid.at(octave); }
        const std::vector<Mat>& octaveGradX(int octave) { return this->grad_x_pyramid.at(octave); }
        const std::vector<Mat>& octaveGradY(int octave) { return this->grad_y_pyramid.at(octave); }
        const std::vector<Mat>& octaveGradMag(int octave) { return this->grad_mag_pyramid.at(octave); }
        const std::vector<Mat>& octaveGradOrient(int octave) { return this->grad_orient_pyramid.at(octave); }
        //pyramid getters
        const std::vector<Mat>& imagePyramid() { return this->img_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidGauss() { return this->gauss_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidDiff() { return this->diff_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidGradX() { return this->grad_x_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidGradY() { return this->grad_y_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidGradMag() { return this->grad_mag_pyramid; }
        const std::map<int, std::vector<Mat>>& pyramidGradOrient() { return this->grad_orient_pyramid; }
        //display functions
        static void showOctave(const std::vector<Mat> images, const std::string window_name, const Point pos = Point(0,0));
        static void showPyramid(const std::map<int, std::vector<Mat>> pyramid);
    private:
        int calculateNumOctaves(Mat& img);
        double calculateSigma(int level);
        void createPyramid(Mat& img);
        void processGradients(std::vector<Mat> gaussians);
        std::vector<Mat> GaussVector(Mat& img);
        std::vector<Mat> Diff_of_Gauss(std::vector<Mat> gaussians);
        std::vector<Mat> img_pyramid;
        std::map<int, std::vector<double>> sigmas_pyramid;
        std::map<int, std::vector<Mat>> gauss_pyramid;
        std::map<int, std::vector<Mat>> diff_pyramid;
        std::map<int, std::vector<Mat>> grad_x_pyramid;
        std::map<int, std::vector<Mat>> grad_y_pyramid;
        std::map<int, std::vector<Mat>> grad_mag_pyramid;
        std::map<int, std::vector<Mat>> grad_orient_pyramid;
        int numOctaves_ = 0;
        int scaleSamples_ = 3;
        int numLevels_ = scaleSamples_ + 3;
        int currentOctave_ = 0;
        double sigma_ = 0.0f;
        double k_ = pow(2.0f, 1.0f/double(scaleSamples_));
        //see: https://www.researchgate.net/profile/Michael-Veth/publication/216789729/figure/fig12/AS:394046162915328@1470959334135/Octaves-of-the-Difference-of-Gaussian-Functions-over-a-Scale-Space-Gaussian-blurred.png
};
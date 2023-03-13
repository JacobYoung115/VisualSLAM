#include "GaussPyramid.hpp"

using namespace cv;
using std::cout, std::endl;

//SIFT paper considers the gaussian function G(x,y,sigma)
//it uses scales seperated by a constant k
//D (x,y,sigma) = ( G(x,y,k*sigma) - G(x,y,sigma) ) conv* I(x,y)

//after each octave, the guassian image is downsampled to half the size.
//i.e. octave 0 --> G(x,y,sigma), octave 1 --> G(x/2, y/2, sigma)

//they set k = 2^(1/s), where is s is an integer number of times we divide the image in scale space.
//ex 4 octaves, s = 4, k = 2^(1/4).

//per octave change, the sigma value is doubled while the image size is halved.
//so if we double sigma 4 times (i.e. 4 octave changes), we need 7 images in each octave?

//so basically, when computing the images of an octave, sigma stays the same
//what changes is the k factor

//does the s variable in k change depending on what octave you're currently in?
//ex: Octave 1:
//k = 2^(1/1) = 2

//Figure 1 says, for EACH octave, the intial image is repeatedly convolved w/ gaussians

//note, mathematically, due to the associative property of convolution, is it faster to
//convolve the gaussian kernels with each other, before convolving with the image.

//equally, you can blur an image with the same blur as two kernels by using
//new_sigma = sqrt( sig1**2 + sig2**2 )
//this will probably be the fastest method.
//https://math.stackexchange.com/questions/3159846/what-is-the-resulting-sigma-after-applying-successive-gaussian-blur
//for development sake though, it will just be fastest to blur each previous image.


void GaussPyramid::displayPyramid(const std::map<int, std::vector<Mat>> pyramid) {
    std::string window_name;
    for (const auto& kv: pyramid) {
        window_name = "Octave " + std::to_string(kv.first) + " blur";
        GaussPyramid::showOctave(kv.second, window_name, Point(kv.second.at(0).cols, 0));
    }
}

void GaussPyramid::showOctave(const std::vector<Mat> images, const std::string window_name, const Point pos) {
    Mat display;
    display = images.at(0);

    for (int i = 1; i < images.size(); ++i) {
        vconcat(display, images[i], display);
    }

    imshow(window_name, display);
    moveWindow(window_name, pos.x, pos.y);
}

void GaussPyramid::createPyramid(Mat& img) {
    //the sift paper states they double the size of the original image for the first level of the pyramid.
    //'double the size of the input image using linear interpolation prior to building the first level of the pyramid'
    Mat pyrBase;
    resize(img, pyrBase, Size(), 2, 2, INTER_LINEAR);

    for (int i = 0; i < this->numOctaves_; ++i) {
        //allocate 7 empty images to move the data over.
        std::vector<Mat> gaussians = GaussVector(pyrBase);
        std::vector<Mat> diffs = Diff_of_Gauss(gaussians);
        this->img_pyramid.emplace_back(pyrBase);                //TODO: Does the image pyramid need to contain unblurred images??

        //The SIFT paper states that, they take a gaussian image w/ twice the initial value of sigma, this corresponds to
        //the 3rd image from the top (in our case, the 5th image)
        Mat octaveBase = gaussians.at(this->numOctaves_); //only copies the header

        //now with the resized image, repeat the two for the other levels of the pyramid.
        resize(octaveBase, pyrBase, Size(), 0.5, 0.5, INTER_NEAREST);
        
        this->gauss_pyramid.emplace(i, gaussians);
        this->diff_pyramid.emplace(i, diffs);
    }
}

std::vector<Mat> GaussPyramid::GaussVector(Mat& img) {
    std::vector<Mat> gaussians{this->numImages_, Mat::zeros(img.size(), CV_32FC1)};
    Mat blurred;
    for (int i = 0; i < this->numImages_; ++i) {
        
        if (i == 0 ) {
            GaussianBlur(img, blurred, Size(0,0), this->sigma_*this->k_, 0, BORDER_DEFAULT);
        }
        else if(i > 0) {
            GaussianBlur(gaussians[i-1], blurred, Size(0,0), this->sigma_*this->k_, 0, BORDER_DEFAULT);
        }
        
        //cv::swap(blurred, gaussians[i]);     //note that swap is just slightly slower (by .1~1 miliseconds)
        gaussians[i] = std::move(blurred);
        //each iteration, take the previous blurred image & blur it again.
    }
    return gaussians;
}

std::vector<Mat> GaussPyramid::Diff_of_Gauss(std::vector<Mat> gaussians) {
    //now, get a vector of difference of gaussians.
    std::vector<Mat> diffs{gaussians.size()-1, Mat::zeros(gaussians[0].size(), CV_32FC1)};
    for (int i = 1; i < gaussians.size(); i++) {
        //start i =1, therefore we can always grab the previous gaussian.
         //moving it mighttt be the problem? Since it's an rvalue in the first place and we're moving to an lvalue?
        diffs[i-1] = std::move(gaussians[i] - gaussians[i-1]);      
    }
    return diffs;
}
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

//it may be useful to display the current octave number along with the sigma value of each image in the octave.
void GaussPyramid::showOctave(const std::vector<Mat> images, const std::string window_name, const Point pos) {
    Mat display;
    display = images.at(0);

    for (int i = 1; i < images.size(); ++i) {
        vconcat(display, images[i], display);
    }

    imshow(window_name, display);
    moveWindow(window_name, pos.x, pos.y);
}

void GaussPyramid::showPyramid(const std::map<int, std::vector<Mat>> pyramid) {
    std::string window_name;
    for (const auto& kv: pyramid) {
        window_name = "Octave " + std::to_string(kv.first) + " blur";
        GaussPyramid::showOctave(kv.second, window_name, Point(kv.second.at(0).cols, 0));
    }
}

void GaussPyramid::processGradients(std::vector<Mat> gaussians) {
    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    std::vector<Mat> xGradients{gaussians.size(), Mat::zeros(gaussians[0].size(), ddepth)};
    std::vector<Mat> yGradients{gaussians.size(), Mat::zeros(gaussians[0].size(), ddepth)};
    std::vector<Mat> gradMags{gaussians.size(), Mat::zeros(gaussians[0].size(), ddepth)};
    std::vector<Mat> gradOrients{gaussians.size(), Mat::zeros(gaussians[0].size(), ddepth)};

    for (int i = 0; i < gaussians.size(); ++i) {
        Sobel(gaussians[i], xGradients[i], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(gaussians[i], yGradients[i], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
        magnitude(xGradients[i], yGradients[i], gradMags[i]);
        phase(xGradients[i], yGradients[i], gradOrients[i], true);
    }
    this->grad_x_pyramid.emplace(this->currentOctave_, xGradients);
    this->grad_y_pyramid.emplace(this->currentOctave_, yGradients);
    this->grad_mag_pyramid.emplace(this->currentOctave_, gradMags);
    this->grad_orient_pyramid.emplace(this->currentOctave_, gradOrients);
}

void GaussPyramid::createPyramid(Mat& img) {
    //the sift paper states they double the size of the original image for the first level of the pyramid.
    //'double the size of the input image using linear interpolation prior to building the first level of the pyramid'
    Mat pyrBase;
    resize(img, pyrBase, Size(), 2, 2, INTER_LINEAR);

    for (int i = 0; i < this->numOctaves_; ++i) {
        this->currentOctave_ = i;   //this is set in order to calculate the sigma value of each level in each octave.

        //allocate 6 empty images to move the data over.
        std::vector<Mat> gaussians = GaussVector(pyrBase);
        std::vector<Mat> diffs = Diff_of_Gauss(gaussians);
        processGradients(gaussians);
        this->img_pyramid.emplace_back(pyrBase);                //TODO: Does the image pyramid need to contain unblurred images??

        //The SIFT paper states that, they take a gaussian image w/ twice the initial value of sigma, this corresponds to
        //the 3rd image from the top (in our case, the 4th image)
        Mat octaveBase = gaussians.at(this->scaleSamples_); //only copies the header

        //now with the resized image, repeat the two for the other levels of the pyramid.
        resize(octaveBase, pyrBase, Size(), 0.5, 0.5, INTER_NEAREST);
        
        this->gauss_pyramid.emplace(i, gaussians);
        this->diff_pyramid.emplace(i, diffs);
    }
}

std::vector<Mat> GaussPyramid::padOctave(int padding, const std::vector<Mat>& images) {
    std::vector<Mat> padded;
    for (const auto& image : images) {
        Mat paddedImg;
        copyMakeBorder(image, paddedImg, padding, padding, padding, padding, BORDER_REPLICATE);
        padded.emplace_back(std::move(paddedImg));        //not sure if this is adding the refence to the old image or copying a new image in.
    }
    return padded;
}

//The maximum number of octaves in an image is set by taking the smaller dimension of width&height
//then, taking log base 2 of that number, which gives the maximum number of times an image can be divided in half.
//dividing an image down further than 32x32 isn't necessarily useful though, so we subtract the max # of octaves by 4.
//ex: given an image 600x850, the maximum number of times it can be divided in half is log2(600) = ~9.2 = 9
//    remove the last 4 non-useful feature layers (1x1, 2x2, 4x4, 8x8)... 9 - 4 = 5 octaves over the images.
//see: https://stackoverflow.com/questions/30291398/vlfeat-computation-of-number-of-octaves-for-sift
//     https://www.researchgate.net/post/Can_any_one_help_me_understand_Deeply_SIFT
void GaussPyramid::calculateNumOctaves(Mat& img) {
    this->numOctaves_ = floor(log2(min(img.rows, img.cols))) - 4;
}

//the first portion pow(2, this->currentOctave) keeps track of the doubling value of each octave. At octave 0, it will be 1
//the second portion this->sigma_* pow(this->k_, level) tracks which level were on.
double GaussPyramid::calculateSigma(int level) {
    return pow(2,this->currentOctave_)*this->sigma_*pow( this->k_, level);
}

const double GaussPyramid::calculateSigma(int octave, int level) {
    return pow(2,octave)*this->sigma_*pow( this->k_, level);
}


//see: https://dsp.stackexchange.com/questions/10074/sift-why-s3-scales-per-octave
std::vector<Mat> GaussPyramid::GaussVector(Mat& img) {
    std::vector<Mat> gaussians{this->numLevels_, Mat::zeros(img.size(), CV_32FC1)};
    std::vector<double> sigmas(this->numLevels_, 0.0);
    Mat blurred;

    //instead of consecutively blurring the image, first calculate the sigma value for that blurred level..
    for (int i = 0; i < this->numLevels_; ++i) {
        sigmas[i] = calculateSigma(i);
        

        //edit this part to instead blur given the sigma of the current level.
        GaussianBlur(img, blurred, Size(0,0), sigmas[i], 0, BORDER_DEFAULT);
        
        //cv::swap(blurred, gaussians[i]);     //note that swap is just slightly slower (by .1~1 miliseconds)
        gaussians[i] = std::move(blurred);
        //each iteration, take the previous blurred image & blur it again.
    }
    this->sigmas_pyramid.emplace(this->currentOctave_, sigmas);
    return gaussians;
}

const double GaussPyramid::getSigmaAt(int octave, int level) {
    return this->sigmas_pyramid.at(octave).at(level);
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
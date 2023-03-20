#include <GaussPyramid/GaussPyramid.hpp>
using namespace cv;
using std::cout, std::endl;

double compareImages(Mat& img1, Mat& img2) {
    double total = 0;
    double divisor = img1.rows * img1.cols;
    int padding = 0;

    for (int i = padding; i < img1.rows-padding; ++i) {
        for (int j = padding; j < img1.cols-padding; ++j) {
            //note, can't divide by 0, but need to add the equivalent pixel value to keep ratios equal.
            if(img1.at<uchar>(i,j) > 0 && img2.at<uchar>(i,j) > 0) {
                total += img1.at<uchar>(i,j) / img2.at<uchar>(i,j);
            }
            else {
                divisor-=1.0;
            }
            
        }
    }

    total /= divisor;
    cout << "Average ratio between image pixels: " << total << endl;
    return total;
}

int roundToNextOdd(double value) {
    int up = ceil(value);
    if (up / 2 != 0) {  //if it isn't odd, return the value, otherwise, if it is even add 1.
        return up;
    }
    else {
        return up + 1;
    }
}

//returns a value 2*sigma, rounded to the nearest odd.
//instead, make it cieled to the next off number.
int getWindowSize(double sigma) {
    int window_size = 0;
    double window = 3.0*sigma;

    if (ceil(window) / 2 !=0) {
        window_size = ceil(window);
    }
    else {
        window_size = ceil(window)+1;
    }


    return window_size;
}

int main() {
    /*
        Goal: Create and display a gaussian pyramid (map)
        
    */

    //choosing a kernel size based upon the sigma value:
    //https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size

    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    std::string img_path = samples::findFile("building.jpg");
    Mat img_color = imread(img_path, IMREAD_COLOR);
    Mat img;
    cvtColor(img_color, img, COLOR_BGR2GRAY);

    int numOctaves = 4; //this represents the s variable in SIFT.
    double sigma = 1.6;                        //paper states a sigma of 1.6
    GaussPyramid pyramid{img, numOctaves, sigma};

    //Take a gaussian pyramid
    //manually calculate each level of the pyramid's sigma value.
    //see how similar the calculation is, to the successively blurred image sequence.
    for (const auto& kv: pyramid.gaussPyramid()) {
        cout << "Pyramid level: " << kv.first << endl;
        //cout << "size of data: " << kv.second.size() << endl;
        cout << "image size: width.height (" << kv.second.at(0).cols << ", " << kv.second.at(0).rows << ")" << endl;
        
        //calculate the sigma value for every level of every octave
    }

    std::vector<double> sigmas;
    std::vector<Mat> octave_1 = pyramid.getBlurOctave(0);
    int num_levels = octave_1.size();

    double mySigma = pyramid.getSigmaAt(0,0);
    cout << "First sigma value of pyramid : " << mySigma << endl;

    double mySecondSigma = pyramid.getSigmaAt(0,pyramid.getNumLevels()-1);
    cout << "Last sigma value of 1st octave : " << mySecondSigma << endl;

    double mySigma_doubled = pyramid.getSigmaAt(0,pyramid.getNumLevels()-3);
    cout << "double sigma value : " << mySigma_doubled << endl;

    
    //https://stackoverflow.com/questions/9905093/how-to-check-whether-two-matrices-are-identical-in-opencv
    //bool areIdentical = !cv::norm(downsized,downsized2,NORM_L1);
    //cout << "Downsized images equal?: " << areIdentical << endl;

    imshow("original image: ", img);        //erroring here for some reason..
    moveWindow("original image: ", 0,0);

    //imshow("downsampled image: ", downsized);
    //moveWindow("downsampled image: ", 0,img.rows + pyrBase.rows);

    //int octave = 3;

    //imshow("pyramid test", pyramid.getBlurOctave(octave).at(0));

    //std::string octave_window = "Octave 1 Blurs";
    //GaussPyramid::showOctave(pyramid.getBlurOctave(1), octave_window);

    std::string octave_window2 = "Octave 2 Blurs";
    //it may be useful for showOctave to print both the octave number and sigma value at that image.
    GaussPyramid::showOctave(pyramid.getBlurOctave(2), octave_window2);

    //std::string octave_window3 = "Octave 3 Blurs";
    //GaussPyramid::showOctave(pyramid.getBlurOctave(3), octave_window3);
    //GaussPyramid::displayPyramid(pyramid.gaussPyramid());

    /*
    imshow("blurred1 image: ", gaussians.at(0));
    moveWindow("blurred1 image: ", 0,img.rows);

    imshow("blurred2 image: ", gaussians.at(1));
    moveWindow("blurred2 image: ", 0,img.rows*2);
    

    //show the Difference of Gaussian images.
    imshow("dog1 image: ", diffs.at(0));
    moveWindow("dog1 image: ", img.cols,img.rows);

    imshow("dog2 image: ", diffs.at(1));
    moveWindow("dog2 image: ", img.cols,img.rows*2);
    */
    

    waitKey(0);
    return 0;
}

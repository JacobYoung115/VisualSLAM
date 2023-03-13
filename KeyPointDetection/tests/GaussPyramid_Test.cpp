#include <GaussPyramid/GaussPyramid.hpp>
using namespace cv;
using std::cout, std::endl;

void compareImages(Mat& img1, Mat& img2) {
    double total = 0;
    double divisor = img1.rows * img1.cols;
    int padding = 5;

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
} 

int main() {
    /*
        Goal: Create and display a gaussian pyramid (map)
        
    */

    std::string img_path = samples::findFile("building.jpg");
    Mat img_color = imread(img_path, IMREAD_COLOR);
    Mat img;
    cvtColor(img_color, img, COLOR_BGR2GRAY);

    int numOctaves = 4; //this represents the s variable in SIFT.
    float sigma = 1.6f;                        //paper states a sigma of 1.6
    GaussPyramid pyramid{img, numOctaves, sigma};


    for (const auto& kv: pyramid.gaussPyramid()) {
        cout << "Pyramid level: " << kv.first << endl;
        //cout << "size of data: " << kv.second.size() << endl;
        cout << "image size: width.height (" << kv.second.at(0).cols << ", " << kv.second.at(0).rows << ")" << endl;
    }


    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;


    //try using a double sigma.
    double sigmaTest = 1.6;
    //Write a function which keeps track of the sigma value of all the levels in each octave.
    //1. Compare manually a blurred image that is gaussian'd twice and an image which is blurred once (but w/ the correct gauss formula)
    //new_sigma = sqrt( sig1**2 + sig2**2 )
    Mat blur1;
    GaussianBlur(img, blur1, Size(), sigmaTest, 0, BORDER_DEFAULT);

    Mat blur2;
    GaussianBlur(blur1, blur2, Size(), sigmaTest, 0, BORDER_DEFAULT);

    minMaxLoc(blur2, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "min val of double blur: " << minVal << endl;
    cout << "max val of double blur: " << maxVal << endl;



    double newSigma = sqrt(pow(sigmaTest, 2.0) + pow(sigmaTest, 2.0));
    Mat blur3;
    GaussianBlur(img, blur3, Size(), newSigma, 0, BORDER_DEFAULT);

    minMaxLoc(blur3, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "min val of single blur: " << minVal << endl;
    cout << "max val of single blur: " << maxVal << endl;

    bool areIdentical = !cv::norm(blur2,blur3,NORM_L1);
    cout << "blurred images equal?: " << areIdentical << endl;      //says they're not equal.
    //this NEEDS to be figured out.

    //95.28% equivalent. Perhaps the remaining percentages could be due to the edge pixels.
    //try sampling only internal pixels (which wouldn't be affected by edge pixels.)
    //note: excluding edge pixels actually decreases the accuracy ._. How??
    compareImages(blur2, blur3);

    imshow("double blurred", blur2);
    imshow("single blurred with new sigma", blur3);

    Mat diff = blur2 - blur3;
    imshow("diff between blurs", diff);


    minMaxLoc(diff, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "min val: " << minVal << endl;
    cout << "max val: " << maxVal << endl;


    //Test if OpenCV is consistent when switching between sigma calculation and kernel size
    Mat blurKernel;
    GaussianBlur(img, blurKernel, Size(5,5), 0, 0, BORDER_DEFAULT);

    Mat blurSigma;
    GaussianBlur(img, blurSigma, Size(), 1.1, 1.1, BORDER_DEFAULT);

    areIdentical = !cv::norm(blurKernel,blurSigma,NORM_L1);
    cout << "opencv sigma / kernelsize conversion equal?: " << areIdentical << endl;      //says they're not equal.

    //results comparing ratios show that these two are not very equal (83%).
    compareImages(blurKernel, blurSigma);
    //apparently they are NOT equal, so OpenCV is not inversable in the way it calculates size vs sigma.
    //Although, ensure the tests are accurate by checking with the different comparisons.


    //Other ways to compare the similarity between the images?
    //1. Histogram comparison cv::compareHist()

    //2. templateMatching cv::matchTemplate()

    //3. take an average ratio between pixel intensities? This could give up to 3 decimal places of accuracy

    //or. create my own gaussian kernel w/ my own sigma value to ensure the kernel and sigma are always aligned.
    //--> this may obviously be the most accurate solution.




    
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

    //std::string octave_window2 = "Octave 2 Blurs";
    //GaussPyramid::showOctave(pyramid.getBlurOctave(2), octave_window2);

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

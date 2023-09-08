#include <iostream>
#include <opencv2/highgui.hpp>
#include <Rotation/rotation.h>

using namespace std;

/*
    Test the following:
    obtain a 100x100 window rotated at 45 degrees of the original image
    1. my custom method of getting the positions, then rotating those and accessing them
    2. the OpenCV method of cropping the minimum values before rotation, then rotating and cropping again.
    Then compare the performance of these two methods.

*/

const char* window_name = "Rotate Image";
const char* trackbar_rotation = "Rotation";

const char* window_name2 = "Rotate Point";
const char* trackbar_pt_rotation = "Rotation";

//read in an image from OpenCV samples.
//std::string img_string = samples::findFile("blox.jpg");
//Mat img = imread(img_string, IMREAD_GRAYSCALE);
Mat img = imread("../images/blox.jpg", IMREAD_GRAYSCALE);
int rotation = 0;
int pt_rotation = 0;
int maxRotation = 360;

//float angle_deg = 180.0f;       //rotating 180 degrees looks fine, but 90 absolutely destroys the image.
int rows = img.rows;
int cols = img.cols;
const Point2i origin(0, 0);
const Point2i center(cols/2, rows/2);
Point2f angles = SLAM::Rotation::cos_sin_of_angle(rotation);
Point2i pt(cols/2, 0);  //should be mid top.      //x = cols, y = rows
int windowSize = 16;
Mat tmp;


static void Rotation_Demo(int, void*) {
    angles = SLAM::Rotation::cos_sin_of_angle(rotation);
    Mat rotated = SLAM::Rotation::rotate_mat_CCW(img, center, angles);
    imshow(window_name, rotated);
}


static void Rotation_Point_Demo(int, void*) {
    img.copyTo(tmp);
    Point2i rot = SLAM::Rotation::rotate_pt_CCW(pt, center, pt_rotation);
    circle(tmp, rot, 3, Scalar(0), 3);
    imshow(window_name2, tmp);
}

int main() {

    //will break if all 3 are run at the same time (too much memory allocated on stack)
    const int times = 100;
    double t = (double)getTickCount();

    /*
    for (int i = 0; i < times; i++) {
        //rotate matrix here
        rotate_mat_CCW(img, center, angles);
    }
    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;
    cout << "Time rotating image (averaged for " << times << " runs): " << t << " milliseconds." << endl;
    */


    //1. Directly computes new rotated positions from previous positions
    Mat ROI = SLAM::Rotation::getRotatedWindow(img, center, 100, 45.0f);
    t = (double)getTickCount();
    for (int i = 0; i < times; i++) {
        //rotate matrix here
        SLAM::Rotation::getRotatedWindow(img, center, windowSize, 45.0f);
    }
    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;
    cout << "Time rotating 16x16 ROI (averaged for " << times << " runs): " << t << " milliseconds." << endl;

   

   //2. DoubleCrop - Crop a window of maximum radius to fit an inner window of the expected size.
   Mat ROI2 = SLAM::Rotation::doubleCrop(img, center, 100, 45.0);
   t = (double)getTickCount();
   for (int i = 0; i < times; i++) {
        //rotate matrix here
        SLAM::Rotation::doubleCrop(img, center, windowSize, 45.0);
    }
    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;
    cout << "Time rotating 16x16 ROI with the double crop method (averaged for " << times << " runs): " << t << " milliseconds." << endl;


    namedWindow(window_name, WINDOW_AUTOSIZE);
    namedWindow(window_name2, WINDOW_AUTOSIZE);

    //create a trackbar PER variable you want to input into the function.
    createTrackbar(trackbar_rotation, window_name, &rotation, maxRotation, Rotation_Demo);
    Rotation_Demo(0,0); //call function to initialize


    createTrackbar(trackbar_pt_rotation, window_name2, &pt_rotation, maxRotation, Rotation_Point_Demo);
    Rotation_Point_Demo(0,0); //call function to initialize

    imshow("ROI", ROI);
    imshow("ROI2", ROI2);
    
    waitKey();
    return 0;
}

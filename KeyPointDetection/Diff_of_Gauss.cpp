#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <iterator>
#include "rotation.h"



using namespace cv;

/*
Notes: first just do it in a way that I understand
then, go back and look at the imgproc tutorials and see ways I can optimize this.
--> eventually need to add scale/octave to the point struct.
*/
namespace SLAM {
    struct point {
        point(int row_, int col_, int value_, int padding_) : row(row_), col(col_), value(value_), padding(padding_) {}
        int row;
        int col;
        int value;
        int padding;
    };
}

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

int main() {

    std::string img_path = samples::findFile("blox.jpg");
    std::string img_path2 = samples::findFile("chessboard.png");

    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    //now that image is loaded, blur it.
    Mat blurred1;
    Mat blurred2;
    Mat blurred3;
    Mat blurred4;
    Mat blurred5;

    //octave 1 (original image size)
    GaussianBlur(img, blurred1, Size(3,3), 0, 0, BORDER_DEFAULT);       //consider adjusting the sigmaX and sigmaY values.
    GaussianBlur(img, blurred2, Size(5,5), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred3, Size(7,7), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred4, Size(9,9), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred5, Size(11,11), 0, 0, BORDER_DEFAULT);

    //subtract blurred images.
    Mat DoG1 = blurred1 - blurred2;
    Mat DoG2 = blurred2 - blurred3;
    Mat DoG3 = blurred3 - blurred4;
    Mat DoG4 = blurred4 - blurred5;

    
    std::vector<Mat> dogs = { DoG1, DoG2, DoG3, DoG4};
    std::vector<Mat> dogs_padded;
    int windowSize = 3;
    int padding = (windowSize-1) / 2;
    for (const auto& dog : dogs) {
        Mat paddedDoG;
        copyMakeBorder(dog, paddedDoG, padding, padding, padding, padding, BORDER_REPLICATE);
        dogs_padded.emplace_back(std::move(paddedDoG));        //not sure if this is adding the refence to the old image or copying a new image in.
    }

    
    std::vector<SLAM::point> keypoints;
    //now compare adjacent DoGs to obtain 26 neighbors dogs[i-1], dogs[i], dogs[i+1]
    //add padding of size 'kernelsize' to the image
    for (int i = padding; i < DoG1.rows; i += windowSize) {
        for (int j = padding; j < DoG1.cols; j += windowSize) {
            std::vector<int> neighbors;
            int this_pixel = (int)dogs_padded[1].at<uchar>(i,j);
            
            //now take a window of the surrounding 26 neighbors 3x3 neighbors in adjacent smoothing directions.
            for (int u = i-padding; u < i+padding; u++) {
                for (int v = j - padding; v < j+padding; v++) {
                    //this is the window. add all neighbbors to
                    neighbors.emplace_back((int)dogs_padded[0].at<uchar>(u,v));
                    neighbors.emplace_back((int)dogs_padded[1].at<uchar>(u,v));
                    neighbors.emplace_back((int)dogs_padded[2].at<uchar>(u,v));
                }
            }

            int min = *min_element(neighbors.begin(), neighbors.end());
            int max = *max_element(neighbors.begin(), neighbors.end());

            
            if (this_pixel == min || this_pixel == max) {
                //Feature Point Localization:
                //https://www.youtube.com/watch?v=LXk4A24V8mQ
                //It filters out additional points by interpolating between positions and scale space to approximate a 3D surface.
                //We do this with the derivative of the pixel w/ respect to x,y and scale directions.
                //effectively this means taking the difference of points in:
                //  x-1,        x+1 direction
                //  y-1,        y+1 direction
                //  adjacent-1, adjacent+1 direction
                int d_x = (int)dogs_padded[1].at<uchar>(i,j-1) - (int)dogs_padded[1].at<uchar>(i,j+1);
                int d_y = (int)dogs_padded[1].at<uchar>(i-1,j) - (int)dogs_padded[1].at<uchar>(i + 1,j);
                int d_scale = (int)dogs_padded[0].at<uchar>(i,j) - (int)dogs_padded[2].at<uchar>(i,j);


                //Now, Perform 'Feature Point Localization'
                //Note that in the SIFT paper, they consider intensity values less than 0.03 in the scale of [0-1] to be low contrast
                //0.03 * 255 = 7.65
                cv::Vec3f A_vec((float)d_x / 255.0f, (float)d_y / 255.0f, (float)d_scale / 255.0f);
                cv::Mat A = Mat(A_vec);     //3x1
                cv::Mat A_T;                                                            
                transpose(A, A_T);          //1x3
                
                Mat B = A*A_T;              //A (3x1) * A^T (1x3) = 3x3 matrix.
                Mat B_inverse = -(B.inv());
                
                Mat z_hat = B_inverse * A;        //3x3 * 3x1 = 3x1 vector
                //std::cout << "z_hat" << std::endl << z_hat << std::endl;
                
                Mat dog_zhat_mat = ((float)this_pixel / 255.0f) + 0.5f * A_T * z_hat;
                //std::cout << "dog_zhat_mat" << std::endl << dog_zhat_mat << std::endl;
                float dog_zhat = dog_zhat_mat.at<float>(0,0);
                
                if (dog_zhat > 0.03f) {     //if the value is above 0.03, then it is a high enough contrast point.
                    keypoints.emplace_back( SLAM::point{i,j,(int)(dog_zhat * 255), padding});
                    //DoG1.at<uchar>(i,j) = (uchar)255;
                }

            }
        }
    }
    
    std::cout << "Number of keypoints (new method): " << keypoints.size() << std::endl;

    //For DoG we also need the deriv in x&y directions, to additionally calculate the gradient direction
    //derivs
    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;        //16bit signed (short)
    Mat grad_x, grad_y, mag, orient;
    Sobel(blurred1, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(blurred1, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    magnitude(grad_x, grad_y, mag);
    phase(grad_x, grad_y, orient, true);
    
    //Now, further filter the keypoints using the Hessian
    Mat Hessian = Mat::zeros(2,2, CV_32F);                 //2,2 matrix filled w/ 0.
    int extrema = 0;
    int windowSize2 = 16;
    int padding2 = windowSize2/2;
    int sigma = 1.5*3;                  //later, update this to by 1.5x the scale of the keypoint.
    std::vector<SLAM::point> reducedKeypoints;

    Mat paddedImg2;
    cv::copyMakeBorder(img, paddedImg2, padding2, padding2, padding2, padding2, BORDER_REPLICATE);
    Mat paddedMag;
    cv::copyMakeBorder(mag, paddedMag, padding2, padding2, padding2, padding2, BORDER_REPLICATE);
    Mat paddedOrient;
    cv::copyMakeBorder(orient, paddedOrient, padding2, padding2, padding2, padding2, BORDER_REPLICATE);

    for (const auto& keypoint : keypoints) {
        //at each keypoint, accumulate the derivatives
        float Ix2 = 0;
        float Iy2 = 0;
        float IxIy = 0;
        int x = keypoint.col;
        int y = keypoint.row;

        //iterate in a window around the keypoint
        for (int u = y - keypoint.padding; u < y + keypoint.padding; u++) {
            for (int v = x - keypoint.padding; v < x + keypoint.padding; v++) {
                Ix2 += grad_x.at<float>(u,v) * grad_x.at<float>(u,v);
                Iy2 += grad_y.at<float>(u,v) * grad_y.at<float>(u,v);
                IxIy += grad_x.at<float>(u,v) * grad_y.at<float>(u,v);
            }
        }
        
        Hessian.at<float>(0,0) = Ix2;
        Hessian.at<float>(0,1) = IxIy;
        Hessian.at<float>(1,0) = IxIy;
        Hessian.at<float>(1,1) = Iy2;
        float det = determinant(Hessian);
        float tr = trace(Hessian)[0];        //trace returns a scalar, which is 4 doubles.
        float response = (tr*tr) / det;
        float r = 10.0f;
        float threshold = ((r + 1.0f) * (r + 1.0f)) / r;

        //only keep values that satify the condition
        if (response < threshold) {
            extrema++;
            //are these calculated for each keypoint or for all neighbors of the keypoint?
            //the paper seems to indicate calculating these values for all pixels
            //the best way to do it would be to take a subsection of the image around that pixel and then perform magnitude, orientation, etc on it.
            //create a 16x16 window around the current pixel.
            Mat window (paddedImg2, Rect(x, y, windowSize2, windowSize2));
            Mat windowMag (paddedMag, Rect(x, y, windowSize2, windowSize2));
            Mat windowOrient (paddedOrient, Rect(x, y, windowSize2, windowSize2));

            Mat magWeighted;
            //note, it might be faster to calculate the orientation & magnitude using just the img window
            GaussianBlur(windowMag, magWeighted, Size(0, 0), sigma, 0, BORDER_DEFAULT);        //note: the scale used here will need to adapt to the scale of the DoG Octave.

            /*
                Now, create a histrogram with this data
                36 bins for orientation
                weight each point w/ a Gaussian window of 1.5 sigma
                --> this could be done by taking a gaussian function the size of the image region (16x16) and multiplying the values by the gaussian value
                    at that point, then dividing the values by the highest value of the gaussian (1).
                
                //How are the values of the bins decided? Well, the magnitude tells the strength at the point of the angle
                //so the magnitude image will need to be weighted
                //But, once we have the magnitudes weighted, how do they add to the histogram?

                //so it seems the histogram doesn't just count the number of angles which appear at that angle
                //it also takes into account the total magnitude at that angle.
            */
            //adding {} is refered to as 'value-initialization' and sets all values to 0.

            /*
            */
            std::vector<float> histo(36, 0.0f);
            //float histo[36]{};
            for (int i = 0; i < window.rows; ++i) {
                for (int j = 0; j < window.cols; ++j) {
                    int index = (int)(windowOrient.at<float>(i,j) * 0.1f);      //angle between [0-360] / 10, int
                    histo.at(index) += magWeighted.at<float>(i,j);
                }
            }

            std::vector<float>::iterator result;
            result = max_element(histo.begin(), histo.end());
            float maxPeak = *result;
            float peakThreshold = maxPeak * 0.8f;
            int index = std::distance(histo.begin(), result);

            //std::cout << "highest orientation peak at angle: " << index*10 << std::endl;
            //std::cout << "value of max peak:  " << maxPeak << std::endl;
            

            //now, read through the histrogram and find peaks.
            for (int k = 0; k < histo.size(); ++k) {
                if (histo.at(k) > peakThreshold) {
                    //std::cout << "peak found with value: " << histo.at(k) << std::endl;
                    //another keypoint should be added if an additional peak is found.
                    //note, verify this!!

                    //also, add the dominant orientation to the keypoint info
                    //here, k*10 gives the angle at that point. Additional peaks at different angles will generate new keypoints.
                    reducedKeypoints.emplace_back( SLAM::point{y,x,k*10,0});
                    //DoG1.at<uchar>(y,x) = (uchar)255;       //note, y is row, x is col.
                }
            }
            


            //DoG1.at<uchar>(keypoint.rows, keypoint.cols) = (uchar)255;
            //break;
        }
    }

    std::cout << "Number of keypoints after edge orientation: " << reducedKeypoints.size() << std::endl;


    //Now that we have the appropriately filtered keypoints, which have a dominant orientation..
    //iterate through each keypoint, take it's primary orientation and rotate a box
    //sample the neighborhood included in the rotated box.

    //so... how do we rotate a portion of an image and iterate over that?
    int windowSize3 = 16;
    int maxPadding = ceil(float(windowSize) * 1.414f / 2.0f);
    std::vector<Point2i> rotatedPoints;
    Point2i currentPoint;
    Mat paddedImg3;
    Mat paddedMag3;
    Mat paddedOrient3;
    copyMakeBorder(img, paddedImg3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
    copyMakeBorder(mag, paddedMag3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
    copyMakeBorder(orient, paddedOrient3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
    std::vector<std::vector<float>> featureDescriptors_vec;

    for (const auto& keypoint : reducedKeypoints) {
        std::vector<float> featureDescriptor;      //maybe should just make this an array of floats.
        featureDescriptor.reserve(128);
        //if we need to use these coordinates (of the real image) multiple times, then it might be best to get those instead.
        rotatedPoints = SLAM::Rotation::getRotatedWindowPoints(paddedImg3, Point2i(keypoint.col + maxPadding, keypoint.row + maxPadding), windowSize3, keypoint.value);
        //assume given region is correct. Now get the img, gradient magnitude & orientation of this window.
        Mat imgROI = Mat::zeros(Size(windowSize3, windowSize3), paddedImg3.type());
        Mat magROI = Mat::zeros(Size(windowSize3, windowSize3), paddedMag3.type());
        Mat orientROI = Mat::zeros(Size(windowSize3, windowSize3), paddedOrient3.type());


        
        for (int i = 0; i < imgROI.rows; ++i) {
            for (int j = 0; j < imgROI.cols; ++j) {
                currentPoint = rotatedPoints[i * imgROI.rows + j];
                imgROI.at<uchar>(i,j) = paddedImg3.at<uchar>(currentPoint.y, currentPoint.x);
                magROI.at<uchar>(i,j) = paddedMag3.at<uchar>(currentPoint.y, currentPoint.x);
                orientROI.at<uchar>(i,j) = paddedOrient3.at<uchar>(currentPoint.y, currentPoint.x);
            }
        }

        Mat magWeighted;
        //TODO:
        // The magnitudes are further weighted by a Gaussian function with sigma  equal to one half the width of the descriptor window.
        // The descriptor then becomes a vector of all the values of these histograms
        GaussianBlur(magROI, magWeighted, Size(0, 0), sigma, 0, BORDER_DEFAULT); 

        imshow("rotated img ROI", imgROI);
        imshow("ROI Magnitude", mat2gray(magROI));
        imshow("ROI Orientation",mat2gray(orientROI));


        //then, create a histrogram of gradients for each 4x4 section of the ROI
        //the histogram can be performed after assignment.
        int histoSize = 8;
        int angleThresh = 360 / histoSize;
        
        for (int h = 1; h < 5; h++) {
            std::vector<float> histo(8, 0.0f);
            histo.reserve(16);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    //this finding of the index uses nearest int indexing, but the SIFT paper uses tri-linear interpolation (p.15).
                    //"to avoid all boundary affects..."
                    int index = int(orientROI.at<float>(i*h,j*h)) / angleThresh;      //angle between [0-360], need it grouped by 45 degrees
                    histo.at(index) += magROI.at<float>(i*h,j*h);
                }
            }
            for (const auto& val : histo) {
                featureDescriptor.emplace_back(val);
            }
            
            //each time a block is done, add that histrogram of 8 values to another vector
        }
        //Now that the 128x1 featureDescriptor is complete, normalize it by the max value.
        float maxPeak = *max_element(featureDescriptor.begin(), featureDescriptor.end());
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [maxPeak](float& c){ return c/maxPeak; });

        //once normalized, reduce the influence of large gradient magnitudes, by thresholding the values in the unit feature vector
        //we do this to reduce the influence of large magnitudes, while emphasizing the distribution of orientations.
        float threshold = 0.2f;
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [threshold](float& c)
        { 
            c = min(c, threshold);
        });

        //then, renormalize again.
        maxPeak = *max_element(featureDescriptor.begin(), featureDescriptor.end());
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [maxPeak](float& c){ return c/maxPeak; });

        featureDescriptors_vec.emplace_back(featureDescriptor);
        



        //append the gradients together.
        break;
    }



    //imshow("SobelX", grad_x);
    //imshow("SobelY", grad_y);
    //imshow("Magnitude", mat2gray(mag));
    //imshow("Orientation",mat2gray(orient));
    imshow("DoG1", DoG1);
    int k = waitKey(0);

    if (k == 's') {
        imwrite("DoG1.jpg", DoG1);
    }

    return 0;

}
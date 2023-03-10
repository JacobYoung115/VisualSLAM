#include <Rotation/rotation.h>
#include <GaussPyramid/GaussPyramid.hpp>
/*
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
*/
#include <algorithm>
#include <iterator>

using namespace cv;
using std::cout, std::endl;

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

    struct ImgRegion {
        ImgRegion(int minRows_, int minCols_, int maxRows_, int maxCols_) : minRows(minRows_), minCols(minCols_), maxRows(maxRows_), maxCols(maxCols_) {}
        int minRows;
        int minCols;
        int maxRows;
        int maxCols;
    };

    //Mat = operation only creates a new header, but automatically refences the data.
    struct ProcessedImage {
        Mat grey;
        Mat color;
        Mat blurred;
        Mat grad_x;
        Mat grad_y;
        Mat magnitude;
        Mat orientation;
    };

}   //namespace SLAM

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

//Use the Hessian structure matrix to filter out keypoints which correspond only to edges.
float computeEdgeResponse(const SLAM::point& keypoint, Mat& grad_x, Mat& grad_y) {
    //at each keypoint, accumulate the derivatives
    float Ix2 = 0;
    float Iy2 = 0;
    float IxIy = 0;
    int x = keypoint.col;
    int y = keypoint.row;
    Mat Hessian = Mat::zeros(2,2, CV_32F);                 //2,2 matrix filled w/ 0.

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
    return response;
}

//region argument is for using an additional loop for multiple image patches.
std::vector<float> orientationHistogram(int size, SLAM::ImgRegion region, Mat& magWeighted, Mat& windowOrient) {
    std::vector<float> histo(size, 0.0f);
    float reductionCoeff = float(size) / 360.0f;

    histo.reserve(size);
    //this finding of the index uses nearest int indexing, but the SIFT paper uses tri-linear interpolation (p.15).
    //"to avoid all boundary affects..."
    for (int i = region.minRows; i < region.maxRows; ++i) {
        for (int j = region.minCols; j < region.maxCols; ++j) {
            float orientation = windowOrient.at<float>(i,j);
            int index = (int)(orientation * reductionCoeff);      //angle between [0-360] / 10, int
            histo.at(index) += magWeighted.at<float>(i,j);
        }
    }

    return histo;
}

void DrawBoundingBox(Mat& image, const std::vector<Point>& positions, Scalar color = Scalar(255)) {
    std::vector<int> x_positions;
    std::vector<int> y_positions;
    x_positions.reserve(positions.size());
    y_positions.reserve(positions.size());
    for (const auto& rPoint : positions) {
        x_positions.emplace_back(rPoint.x);
        y_positions.emplace_back(rPoint.y);
    }

    std::vector<int>::iterator result;
    int index = 0;

    //Drawing the bounding boxes is working.
    //min X
    result = min_element(x_positions.begin(), x_positions.end());
    index = std::distance(x_positions.begin(), result);
    Point p1 = Point(x_positions.at(index), y_positions.at(index));

    //max X
    result = max_element(x_positions.begin(), x_positions.end());
    index = std::distance(x_positions.begin(), result);
    Point p2 = Point(x_positions.at(index), y_positions.at(index));

    //min Y
    result = min_element(y_positions.begin(), y_positions.end());
    index = std::distance(y_positions.begin(), result);
    Point p3 = Point(x_positions.at(index), y_positions.at(index));

    //max Y
    result = max_element(y_positions.begin(), y_positions.end());
    index = std::distance(y_positions.begin(), result);
    Point p4 = Point(x_positions.at(index), y_positions.at(index));

    //now draw lines between these 4 points.
    int thickness = 2;
    int lineType = LINE_8;

    line(image, p1, p3, color, thickness, lineType);
    line(image, p3, p2, color, thickness, lineType);
    line(image, p2, p4, color, thickness, lineType);
    line(image, p4, p1, color, thickness, lineType);
}

void DrawKeypoint(Mat& img, Point center, int radius=3, Scalar color = Scalar(0,255,255), int angle=0, bool drawAngle=true) {
    circle(img, center, radius, color);

    if (drawAngle) {
        Point rotationEdge(center.x+radius,center.y+radius);        //position, then rotate.
        rotationEdge = SLAM::Rotation::rotate_pt_CCW(rotationEdge, center, angle);     //need to ensure this aligns with the later functions.
        
        //it would also be good to draw the rotation direction as a vector.
        line(img, center, rotationEdge, color);
    }
}

//It filters out additional points by interpolating between positions and scale space to approximate a 3D surface.
//We do this with the derivative of the pixel w/ respect to x,y and scale directions.
//effectively this means taking the difference of points in:
//  x-1,        x+1 direction
//  y-1,        y+1 direction
//  adjacent-1, adjacent+1 direction
void FeaturePointLocalization(std::vector<Mat>& dogs_padded, std::vector<SLAM::point>& keypoints, int i, int j, int this_pixel, int padding) {
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
    Mat dog_zhat_mat = ((float)this_pixel / 255.0f) + 0.5f * A_T * z_hat;

    float dog_zhat = dog_zhat_mat.at<float>(0,0);
    
    if (dog_zhat > 0.03f) {     //if the value is above 0.03, then it is a high enough contrast point.
        keypoints.emplace_back( SLAM::point{i,j,(int)(dog_zhat * 255), padding});
    }
}

//aka Scale Space Extrema Detection
std::vector<SLAM::point> initialKeypointDetection(const std::vector<Mat>& dogs, int windowSize) {
    std::vector<SLAM::point> keypoints;

    int padding = (windowSize-1) / 2;
    std::vector<Mat> dogs_padded;
    for (const auto& dog : dogs) {
        Mat paddedDoG;
        copyMakeBorder(dog, paddedDoG, padding, padding, padding, padding, BORDER_REPLICATE);
        dogs_padded.emplace_back(std::move(paddedDoG));        //not sure if this is adding the refence to the old image or copying a new image in.
    }

    //now compare adjacent DoGs to obtain 26 neighbors dogs[i-1], dogs[i], dogs[i+1]
    //add padding of size 'kernelsize' to the image
    for (int i = padding; i < dogs[0].rows; i += windowSize) {
        for (int j = padding; j < dogs[0].cols; j += windowSize) {
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

            
            //if pixel is a local max or minima, intepolate the extreme in x,y & scale space.
            if (this_pixel == min || this_pixel == max) {
                //https://www.youtube.com/watch?v=LXk4A24V8mQ
                FeaturePointLocalization(dogs_padded, keypoints, i, j, this_pixel, padding);
            }
        }
    }

    return keypoints;
}


//keypoints are first filtered by removing any points which correspond to edges or flat regions.
std::vector<SLAM::point> filterKeypoints(SLAM::ProcessedImage myImages, std::vector<SLAM::point>& keypoints) {
    //Now, further filter the keypoints using the Hessian
    int extrema = 0;
    int windowSize2 = 16;
    int padding2 = windowSize2/2;
    int sigma = 1.5*3;                  //later, update this to by 1.5x the scale of the keypoint.
    std::vector<SLAM::point> reducedKeypoints;
    

    Mat paddedImg2;
    cv::copyMakeBorder(myImages.grey, paddedImg2, padding2, padding2, padding2, padding2, BORDER_REPLICATE);
    Mat paddedMag;
    cv::copyMakeBorder(myImages.magnitude, paddedMag, padding2, padding2, padding2, padding2, BORDER_REPLICATE);
    Mat paddedOrient;
    cv::copyMakeBorder(myImages.orientation, paddedOrient, padding2, padding2, padding2, padding2, BORDER_REPLICATE);

    for (const auto& keypoint : keypoints) {
        int x = keypoint.col;
        int y = keypoint.row;
        float response = computeEdgeResponse(keypoint, myImages.grad_x, myImages.grad_y);
        float r = 10.0f;
        float threshold = ((r + 1.0f) * (r + 1.0f)) / r;

        //only keep values that satify the condition
        if (response < threshold) {
            extrema++;

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

            //Calculates a histogram of directions. Size controls the amount of degrees between each bin.
            //ex: size = 36 --> 360 / 36  --> 36 bins each of 10 degrees.
            int size = 36;
            SLAM::ImgRegion region = {0, 0, window.rows, window.cols};
            std::vector<float> histo = orientationHistogram(size, region, magWeighted, windowOrient);

            std::vector<float>::iterator result;
            result = max_element(histo.begin(), histo.end());
            float maxPeak = *result;
            float peakThreshold = maxPeak * 0.8f;
            
            //now, read through the histrogram and find peaks.
            for (int k = 0; k < histo.size(); ++k) {
                if (histo.at(k) > peakThreshold) {
                    int angle = k * 10;
                    //here, k*10 gives the angle at that point. Additional peaks at different angles will generate new keypoints.
                    reducedKeypoints.emplace_back( SLAM::point{y,x,angle,0});
                    //DoG1.at<uchar>(y,x) = (uchar)255;       //note, y is row, x is col.

                    //draw a circle around each reduced keypoint                  
                    DrawKeypoint(myImages.color, Point(x,y), 9, Scalar(0,255,255), angle, true);
                }
            }
        }
    }

    return reducedKeypoints;
}


//need a function to handle the difference of gaussian pyramids.


//all image preprocessing should be done in a single function.
void PostProcessing(Mat& color, Mat& blurred, SLAM::ProcessedImage& myImages) {
    myImages.color = color;
    //cvtColor(myImages.color, myImages.grey, COLOR_BGR2GRAY);
    myImages.blurred = blurred;
    //myImages.grey = img;
    
    //For DoG we also need the deriv in x&y directions, to additionally calculate the gradient direction
    //derivs
    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;        //16bit signed (short)

    Sobel(myImages.blurred, myImages.grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(myImages.blurred, myImages.grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    magnitude(myImages.grad_x, myImages.grad_y, myImages.magnitude);
    phase(myImages.grad_x, myImages.grad_y, myImages.orientation, true);
}

int main() {

    std::string img_path = samples::findFile("building.jpg");
    std::string img_path2 = samples::findFile("chessboard.png");

    Mat img_color = imread(img_path, IMREAD_COLOR);
    Mat img;
    cvtColor(img_color, img, COLOR_BGR2GRAY);
    // = imread(img_path, IMREAD_GRAYSCALE);

    int numOctaves = 4; //this represents the s variable in SIFT.
    float pyr_sigma = 1.6f;                        //paper states a pyr_sigma of 1.6
    std::string octave_window = "Octave 3 Blurs";
    GaussPyramid pyramid{img, numOctaves, pyr_sigma};

    
    for (const auto& kv: pyramid.gaussPyramid()) {
        cout << "Pyramid level: " << kv.first << endl;
        //cout << "size of data: " << kv.second.size() << endl;
        cout << "image size: width.height (" << kv.second.at(0).cols << ", " << kv.second.at(0).rows << ")" << endl;
    }
    GaussPyramid::showOctave(pyramid.getBlurOctave(2), octave_window);
    

   //TODO: Now, replace the code which uses individual images, to use the full pyramid.

    //now that image is loaded, blur it.
    Mat blurred1;
    Mat blurred2;
    Mat blurred3;
    Mat blurred4;
    Mat blurred5;

    SLAM::ProcessedImage myImages;
    myImages.grey = img;
    

    //octave 1 (original image size)
    GaussianBlur(img, blurred1, Size(3,3), 0, 0, BORDER_DEFAULT);       //consider adjusting the sigmaX and sigmaY values.
    PostProcessing(img_color, blurred1, myImages);                      //need to adjust the post-processing struct to work w/ pyramid.
    
    GaussianBlur(img, blurred2, Size(5,5), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred3, Size(7,7), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred4, Size(9,9), 0, 0, BORDER_DEFAULT);
    GaussianBlur(img, blurred5, Size(11,11), 0, 0, BORDER_DEFAULT);

    //subtract blurred images.
    Mat DoG1 = blurred1 - blurred2;
    Mat DoG2 = blurred2 - blurred3;
    Mat DoG3 = blurred3 - blurred4;
    Mat DoG4 = blurred4 - blurred5;


    //The Gaussian Pyramid can be a map (dictionary), where the key is the octave
    //while the value is the list of blurred images.

    
    
    //const std::vector<Mat> dogs = { DoG1, DoG2, DoG3, DoG4};

    int windowSize = 3;
    //note, it should actually take in the dogs list, rather than the padded.
    //std::vector<SLAM::point> keypoints = initialKeypointDetection(dogs, windowSize);
    
    std::vector<SLAM::point> keypoints = initialKeypointDetection(pyramid.getDiffOctave(0), windowSize);


    std::cout << "Number of keypoints (new method): " << keypoints.size() << std::endl;

    std::vector<SLAM::point> reducedKeypoints = filterKeypoints(myImages, keypoints);
    std::cout << "Number of keypoints after edge orientation: " << reducedKeypoints.size() << std::endl;


    //Now that we have the appropriately filtered keypoints, which have a dominant orientation..
    //iterate through each keypoint, take it's primary orientation and rotate a box
    //sample the neighborhood included in the rotated box.
    int windowSize3 = 16;
    //1.414f is sqrt(2), which is used to pad an image according to the 
    //hypotaneuse of a square window, rather than by width & height.
    int maxPadding = ceil(float(windowSize) * 1.414f / 2.0f);
    int sigma = 1.5*3;                  //Update this later!!       
    std::vector<Point2i> rotatedPoints;
    Point2i currentPoint;
    Mat paddedImg3;
    Mat paddedMag3;
    Mat paddedOrient3;
    copyMakeBorder(myImages.grey, paddedImg3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
    copyMakeBorder(myImages.magnitude, paddedMag3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
    copyMakeBorder(myImages.orientation, paddedOrient3, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);
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

        DrawBoundingBox(paddedImg3, rotatedPoints);
        
        for (int i = 0; i < imgROI.rows; ++i) {
            for (int j = 0; j < imgROI.cols; ++j) {

                //TODO: Be careful here.. there was a segmentation fault, I swaped the x & y's & that seemed to
                //made the error go away..
                currentPoint = rotatedPoints[i * imgROI.rows + j];
                imgROI.at<uchar>(i,j) = paddedImg3.at<uchar>(currentPoint.x, currentPoint.y);
                magROI.at<float>(i,j) = paddedMag3.at<float>(currentPoint.x, currentPoint.y);
                orientROI.at<float>(i,j) = paddedOrient3.at<float>(currentPoint.x, currentPoint.y);
            }
        }

        Mat magWeighted;
        //TODO:
        // The magnitudes are further weighted by a Gaussian function with sigma  equal to one half the width of the descriptor window.
        // The descriptor then becomes a vector of all the values of these histograms
        GaussianBlur(magROI, magWeighted, Size(0, 0), sigma, 0, BORDER_DEFAULT); 

        /*
        imshow("rotated img ROI", imgROI);
        imshow("ROI Magnitude", mat2gray(magROI));
        imshow("ROI Orientation",mat2gray(orientROI));
        */


        //then, create a histrogram of gradients for each 4x4 section of the ROI
        //the histogram can be performed after assignment.
        int histoSize = 8;
        int angleThresh = 360 / histoSize;
        int subregion = 4;
        SLAM::ImgRegion region = {0,0,subregion,subregion};

        //TODO: convert this into a function "iterate subregion", which takes as arg a function & other stuff.
        while (region.minRows < windowSize3) {
            std::vector<float> histo = orientationHistogram(histoSize, region, magWeighted, orientROI);
            for (const auto& val : histo) {
                featureDescriptor.emplace_back(val);
            }

            region.maxCols += subregion;
            region.minCols += subregion;

            if (region.minCols == windowSize3) {
                region.maxRows += subregion;
                region.minRows += subregion;
                region.minCols = 0;
                region.maxCols = subregion;
            }         
            //each time a block is done, add that histrogram of 8 values to another vector
        }
        //Now that the 128x1 featureDescriptor is complete, normalize it by the max value.
        std::cout << "Feature descriptor size: " << featureDescriptor.size() << std::endl;


        
        float maxPeak = *max_element(featureDescriptor.begin(), featureDescriptor.end());
        
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [maxPeak](float& c){ return c/maxPeak; });
        
        //once normalized, reduce the influence of large gradient magnitudes, by thresholding the values in the unit feature vector
        //we do this to reduce the influence of large magnitudes, while emphasizing the distribution of orientations.
        float threshold = 0.2f;
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [threshold](float& c)
        { 
            return min(c, threshold);
        });
        

        //then, renormalize again.
        maxPeak = *max_element(featureDescriptor.begin(), featureDescriptor.end());
        transform(featureDescriptor.begin(), featureDescriptor.end(), featureDescriptor.begin(), [maxPeak](float& c){ return c/maxPeak; });

        //append the gradients together.
        featureDescriptors_vec.emplace_back(featureDescriptor);

        
        //final step is graphing and comparing two images with each other.
        break;
    }



    //imshow("SobelX", grad_x);
    //imshow("SobelY", grad_y);
    //imshow("Magnitude", mat2gray(mag));
    //imshow("Orientation",mat2gray(orient));
    //imshow("DoG1", DoG1);
    imshow("Img", myImages.color);
    moveWindow("Img", 0, 0);
    imshow("Padded img w/ lines", paddedImg3);
    moveWindow("Padded img w/ lines", 0, myImages.color.rows);
    int k = waitKey(0);

    if (k == 's') {
        imwrite("DoG1.jpg", DoG1);
    }

    return 0;

}
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
#include <fstream>

using namespace cv;
using std::cout, std::endl;

/*
Notes: first just do it in a way that I understand
then, go back and look at the imgproc tutorials and see ways I can optimize this.
--> eventually need to add scale/octave to the point struct.
*/
namespace SLAM {
    struct point {
        point(int row_, int col_, int value_, int padding_, int octave_, int level_) : row(row_), col(col_), value(value_), padding(padding_), octave(octave_), level(level_) {}
        int row;
        int col;
        int value;
        int padding;
        int octave;
        int level;
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

void MinMax(Mat img, std::string const& output="image") {
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;

    minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc );

    cout << "min val of " << output << ": " << minVal << endl;
    cout << "max val of " << output << ": " << maxVal << endl;
}

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

//Use the Hessian structure matrix to filter out keypoints which correspond only to edges.
float computeEdgeResponse(const SLAM::point& keypoint, const Mat& grad_x, const Mat& grad_y) {
    //at each keypoint, accumulate the derivatives
    float Ix2 = 0;
    float Iy2 = 0;
    float IxIy = 0;
    int x = keypoint.col;
    int y = keypoint.row;
    Mat Hessian = Mat::zeros(2,2, CV_32F);                 //2,2 matrix filled w/ 0.

    //imshow("cer gradX", mat2gray(grad_x));
    //imshow("cer gradY", mat2gray(grad_y));
    //waitKey(0);

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
            //cout << "Orientation: " << orientation << endl; //suddenly the last orientation becomes HUGE
            //why?
            //cout << "reductionCoeff: " << reductionCoeff << endl;
            int index = (int)(orientation * reductionCoeff);      //angle between [0-360] / 10, int
            //for some reason the value is returning the max negative int value.
            //cout << "Index of histo: " << index << " max size: " << size << endl;
            histo.at(index) += magWeighted.at<float>(i,j);
        }
    }

    return histo;
}

void DrawBoundingBox(Mat& image, const std::vector<Point>& positions, int octave, Scalar color = Scalar(255)) {
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
    int radius = 8;
    float factor = pow(2.0, octave) / 2.0f;
    radius = int(float(radius) * factor);


    //min X
    result = min_element(x_positions.begin(), x_positions.end());
    index = std::distance(x_positions.begin(), result);
    Point p1 = Point(x_positions.at(index) - radius, y_positions.at(index));

    //max X
    result = max_element(x_positions.begin(), x_positions.end());
    index = std::distance(x_positions.begin(), result);
    Point p2 = Point(x_positions.at(index) + radius, y_positions.at(index));

    //min Y
    result = min_element(y_positions.begin(), y_positions.end());
    index = std::distance(y_positions.begin(), result);
    Point p3 = Point(x_positions.at(index), y_positions.at(index) - radius);

    //max Y
    result = max_element(y_positions.begin(), y_positions.end());
    index = std::distance(y_positions.begin(), result);
    Point p4 = Point(x_positions.at(index), y_positions.at(index) + radius);

    //now draw lines between these 4 points.
    int thickness = octave+1;
    int lineType = LINE_8;

    line(image, p1, p3, color, thickness, lineType);
    line(image, p3, p2, color, thickness, lineType);
    line(image, p2, p4, color, thickness, lineType);
    line(image, p4, p1, color, thickness, lineType);
}

void DrawKeypoint(Mat& img, Point center, int octave, int radius=6, Scalar color = Scalar(0,255,255), int angle=0, bool drawAngle=true) {
    //change the circle size dependent on the octave size.
    radius = int(radius * (pow(2.0, octave) / 2.0f));
    //also, change the color depending on the radius.

    
    if (octave == 0) {
        color = Scalar(0,255,255);
        circle(img, center, radius, color);
    } else if (octave == 1) {
        color = Scalar(255,0,0);
        circle(img, center, radius, color);
    } else if (octave == 2) {
        color = Scalar(0,0,255);
        circle(img, center, radius, color);
    } else if (octave == 3) {
        color = Scalar(255,0,255);
        circle(img, center, radius, color);
    }
    

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
bool FeaturePointLocalization(std::vector<Mat>& dogs_padded, std::vector<SLAM::point>& keypoints, int level, SLAM::point& point) {
    int i = point.row;
    int j = point.col;
    int d_x = (int)dogs_padded[level].at<uchar>(i,j-1) - (int)dogs_padded[level].at<uchar>(i,j+1);
    int d_y = (int)dogs_padded[level].at<uchar>(i-1,j) - (int)dogs_padded[level].at<uchar>(i + 1,j);
    int d_scale = (int)dogs_padded[level-1].at<uchar>(i,j) - (int)dogs_padded[level+1].at<uchar>(i,j);

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
    Mat dog_zhat_mat = ((float)point.value / 255.0f) + 0.5f * A_T * z_hat;

    float dog_zhat = dog_zhat_mat.at<float>(0,0);
    
    if (dog_zhat > 0.03f) {     //if the value is above 0.03, then it is a high enough contrast point.
        point.value = (int)(dog_zhat * 255.0f);
        keypoints.emplace_back(point);
        return true;  
    }
    return false;
}

//aka Scale Space Extrema Detection
void initialKeypointDetection(std::vector<SLAM::point>& keypoints, GaussPyramid& pyramid, int octave, int windowSize) {
    const std::vector<Mat>& dogs = pyramid.octaveDiff(octave);
    
    //cout << "size of dogs: " << dogs.size() << endl;

    int padding = (windowSize-1) / 2;
    std::vector<Mat> dogs_padded = GaussPyramid::padOctave(padding, dogs);

    //adjust it to work with all dogs padded images, not just the first 3
    //from our 5 dogs images, we will create 3 scale samples
    for (int level=1; level < dogs.size()-1; level++) {
        //now compare adjacent DoGs to obtain 26 neighbors dogs[i-1], dogs[i], dogs[i+1]
        //add padding of size 'kernelsize' to the image
        for (int i = padding; i < dogs[0].rows; i += windowSize) {
            for (int j = padding; j < dogs[0].cols; j += windowSize) {
                std::vector<int> neighbors;
                int this_pixel = (int)dogs_padded[level].at<uchar>(i,j);
                
                //now take a window of the surrounding 26 neighbors 3x3 neighbors in adjacent smoothing directions.
                for (int u = i-padding; u < i+padding; u++) {
                    for (int v = j - padding; v < j+padding; v++) {
                        //this is the window. add all neighbbors to
                        neighbors.emplace_back((int)dogs_padded[level-1].at<uchar>(u,v));
                        neighbors.emplace_back((int)dogs_padded[level].at<uchar>(u,v));
                        neighbors.emplace_back((int)dogs_padded[level+1].at<uchar>(u,v));
                    }
                }

                int min = *min_element(neighbors.begin(), neighbors.end());
                int max = *max_element(neighbors.begin(), neighbors.end());

                
                //if pixel is a local max or minima, intepolate the extreme in x,y & scale space.
                if (this_pixel == min || this_pixel == max) {
                    //https://www.youtube.com/watch?v=LXk4A24V8mQ
                    SLAM::point local_extrema(i,j,this_pixel, padding, octave, level);
                    bool added = FeaturePointLocalization(dogs_padded, keypoints, level, local_extrema);
                }
            }
        }


    }
}


//keypoints are first filtered by removing any points which correspond to edges or flat regions.
void filterKeypoints(GaussPyramid& pyramid, int octave, std::vector<SLAM::point>& keypoints, std::vector<SLAM::point>& reducedKeypoints) {
    //Now, further filter the keypoints using the Hessian
    int extrema = 0;
    int windowSize = 16;
    int padding = windowSize/2;

    //the SIFT paper states
    /*
        The scale of the keypoint is used to select the Gaussian smoothed image, L, with the closest scale, 
        so that all computations are performed in a scale-invariant manner. For each image sample, L(x, y), at this
        scale, the gradient magnitude, m(x, y), and orientation, θ(x, y), is precomputed using pixel differences:

        In my case, this means that using the octave and level is the correct approach to selecting a ROI 
        of the calculated gradient mag & orientation
    */

   //How can I avoid padding the images? Do I just need to precompute it in the gaussian pyramid?
    std::vector<Mat> paddedBlurs = GaussPyramid::padOctave(padding, pyramid.octaveBlur(octave));
    std::vector<Mat> paddedMags = GaussPyramid::padOctave(padding, pyramid.octaveGradMag(octave));
    std::vector<Mat> paddedOrients = GaussPyramid::padOctave(padding, pyramid.octaveGradOrient(octave));

    //cout << "number of keypoints: " << keypoints.size() << endl;
    //cout << "size of padded blurs: " << paddedBlurs.size() << endl;
    //cout << "size of img in padded blurs: " << paddedBlurs[0].size() << endl;

    for (const auto& keypoint : keypoints) {
        int x = keypoint.col;
        int y = keypoint.row;
        float response = computeEdgeResponse(keypoint, 
                                            pyramid.octaveGradX(octave).at(keypoint.level), 
                                            pyramid.octaveGradY(octave).at(keypoint.level));
        float r = 10.0f;
        float threshold = ((r + 1.0f) * (r + 1.0f)) / r;

        //only keep values that satify the condition
        if (response < threshold) {
            //cout << "image response: " << response << endl;
            extrema++;

            //create a 16x16 window around the current pixel.
            Mat window (paddedBlurs.at(keypoint.level), Rect(x, y, windowSize, windowSize));
            Mat windowMag (paddedMags.at(keypoint.level), Rect(x, y, windowSize, windowSize));
            Mat windowOrient (paddedOrients.at(keypoint.level), Rect(x, y, windowSize, windowSize));

            Mat magWeighted;
            double sigma = 1.5 * pyramid.getSigmaAt(keypoint.octave, keypoint.level);                  //later, update this to by 1.5x the scale of the keypoint.
            //note, it might be faster to calculate the orientation & magnitude using just the img window
            GaussianBlur(windowMag, magWeighted, Size(0, 0), sigma, 0, BORDER_DEFAULT);        //note: the scale used here will need to adapt to the scale of the DoG Octave.

            //Calculates a histogram of directions. Size controls the amount of degrees between each bin.
            //ex: size = 36 --> 360 / 10  --> 36 bins each of 10 degrees.
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
                    reducedKeypoints.emplace_back( SLAM::point{y, x, angle, 0, keypoint.octave, keypoint.level});
                    //DoG1.at<uchar>(y,x) = (uchar)255;       //note, y is row, x is col.
                }
            }
        }
    }
}

void debugOrientation(const SLAM::point& keypoint, Point2i& rotatedPoint, float orientation, Mat& levelImg, Mat& levelOrient, Mat& levelMag, int i, int j) {
    //error occurs as soon as orientation is used.
    if (orientation > 360.0f || orientation < 0.0f) {

        if (rotatedPoint.x == 324 && rotatedPoint.y == 22) {
            cout << "Orientation out of range: " << orientation << endl;
            cout << "Occured at (octave, level): " << keypoint.octave << ":" << keypoint.level << endl;
            cout << "Occured at point (x, y): " << rotatedPoint.x << "x" << rotatedPoint.y << endl;
            //orientation = 0.0f; //set it to a correct value.
            //notice that the error is only occuring at the same point 324x22 on octave 2 : all levels.
            //why is this point broken??


            //check 
            //0. check what i & j are.
            cout << "index of non-rotated points: (" << i << ", " << j << ")" << endl;
            //1. print the values immediately left and right of the pixel. They MUST be between 0 - 1.0
            cout << "orient values adjacent to bad point (left): " << levelOrient.at<float>(rotatedPoint.x-1, rotatedPoint.y) << endl;
            cout << "orient values adjacent to bad point (right): " << levelOrient.at<float>(rotatedPoint.x+1, rotatedPoint.y) << endl;
            cout << "orient values adjacent to bad point (up): " << levelOrient.at<float>(rotatedPoint.x, rotatedPoint.y+1) << endl;
            cout << "orient values adjacent to bad point (down): " << levelOrient.at<float>(rotatedPoint.x, rotatedPoint.y-1) << endl;

            //2. check the type
            cout << "Image type: " << levelOrient.type() << endl;

            //3. check the image values of other images
            cout << "orient val at bad point (orient): " << orientation << endl;
            cout << "orient val at bad point (levelOrient): " << levelOrient.at<float>(rotatedPoint.x, rotatedPoint.y) << endl;
            cout << "mag val at bad point: " << levelMag.at<float>(rotatedPoint.x, rotatedPoint.y) << endl;
            cout << "img val at bad point: " << (int)levelImg.at<uchar>(rotatedPoint.x, rotatedPoint.y) << endl;

        }
        
    }
}

void SIFTDebugging() {

    /*
    if (octave == 2 && keypoint.level == 1) {
        imshow("orientation img", levelOrient);
        //break;
    }
    */

   //DEBUGGING. need to be ABSOLUTELY CERTAIN that these rotated points are within the padded bounds!
    //Continue this later..
    /*
    int width = levelOrient.cols;
    int height = levelOrient.rows;
    for (const auto& rPoint : rotatedPoints) {
        if (rPoint.x > width || rPoint.x < 0) {
            cout << "point x is out of range: " << rPoint.x  << " max range[" << maxPadding << ", " <<  width+maxPadding  << "]" << endl;
        }
        if (rPoint.y > height || rPoint.y < 0) {
            cout << "point y is out of range: " << rPoint.y  << " max range[" << maxPadding << ", " <<  height+maxPadding  << "]" << endl;
        }
    }
    */


    /*  DEBUGGING - print the values of every image, try to find a bad value.
    
    //go through the images here.
    int count = 0;
    for (Mat image: paddedOrients) {
        //first, print the minmax of the image.
        MinMax(image, "tor_padded_" + std::to_string(count));

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                //check if values are out of range.
                float orientation = image.at<float>(i, j);
                if (orientation > 360.0f || orientation < 0.0f) {
                    cout << "bad value {" << orientation << "} at col: " << j << " row: " << i << endl;
                }
            }
        }
        count++;
    }

    count = 0;
    for (Mat image: pyramid.octaveGradOrient(octave)) {
        //first, print the minmax of the image.
        MinMax(image, "tor_" + std::to_string(count));

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                //check if values are out of range.
                float orientation = image.at<float>(i, j);
                if (orientation > 360.0f || orientation < 0.0f) {
                    cout << "bad value {" << orientation << "} at col: " << j << " row: " << i << endl;
                }
            }
        }
        count++;
    }
    */



    /*  DEBUGGING
    //TODO: Move this into the keypoint, then check only on that keypoint.
    // this will check the value at point 324x22 in the non-padded image
    
    if (octave == 2) {
        Mat tor_padded = paddedOrients.at(1);
        Mat tor = pyramid.octaveGradOrient(octave).at(1);

        //now, manually pad the image, and see if it is an issue.
        Mat tor_padded_manual;
        copyMakeBorder(tor, tor_padded_manual, maxPadding, maxPadding, maxPadding, maxPadding, BORDER_REPLICATE);

        
        cout << "Max padding: " << maxPadding << endl;
        cout << "width: " << tor.cols << " x height: " << tor.rows << endl;
        cout << "(padded) width: " << tor_padded.cols << " x height: " << tor_padded.rows << endl;
        cout << "(padded manual) width: " << tor_padded_manual.cols << " x height: " << tor_padded_manual.rows << endl;

        cout << "Image type: " << tor.type() << endl;
        cout << "(padded) Image type: " << tor_padded.type() << endl;
        cout << "(padded manual) Image type: " << tor_padded_manual.type() << endl;

        //print the min and max values.
        MinMax(tor, "tor");
        MinMax(tor_padded, "tor_padded");
        MinMax(tor_padded_manual, "tor_padded_manual");


        //I need to be absolutely sure this is the correct position.
        //print the size of the padded image & the size of the non-padded image.
        //I suspect that it may not be the correct position, since
        cout << "gradient value in non-padded image: " 
             << pyramid.octaveGradOrient(octave).at(1).at<float>(324-maxPadding, 22-maxPadding) 
             << endl;

        cout << "gradient value in (padded) image: " 
             << tor_padded.at<float>(324, 22) 
             << endl;
        
        cout << "gradient value in (padded manual) image: " 
             << tor_padded_manual.at<float>(324, 22) 
             << endl;
        
        //is it possible to show the image and draw a circle around where the error is occuring?
        //circle(tor_padded_manual, Point(324,22), 15, Scalar(0,0,0), 5);
        //circle(tor_padded_manual, Point(324,22), 5, Scalar(1,1,1), 5);
        //imshow("error", mat2gray(tor_padded_manual));
        //waitKey(0);
    }
    */
}

//sets the image section to it's rotated counterpart.
void rotateImageSection(Mat& levelImg, Mat& levelMag, Mat& levelOrient, Mat& imgROI, Mat& magROI, Mat& orientROI, std::vector<cv::Point>& rotatedPoints) {
    Point2i rotatedPoint;
    for (int i = 0; i < imgROI.rows; ++i) {
        for (int j = 0; j < imgROI.cols; ++j) {
            //Ideas on the error:
            //could it be related to the padding? Maybe at a certain point, invalid memory is accessed?
            //the point seems to be occurring well within the boundaries.
            //or.. do we need to check that the creation of the magnitude is somehow messing up?
            //Perhaps it's a race condition??
            //in fact, sometimes the program does not have an issue with the rotated value..

            //TODO: Be careful here.. there was a segmentation fault, I swaped the x & y's & that seemed to
            //made the error go away..
            
            

            //notice 1D access of 2D array. i is rows (height) and j is cols (width)
            rotatedPoint = rotatedPoints[i * imgROI.rows + j];

            //Why am I swapping the x's & y's here?? The point rotationed are calculated with x as cols
            //and y as rows.
            imgROI.at<uchar>(i,j) = levelImg.at<uchar>(rotatedPoint.x, rotatedPoint.y);
            magROI.at<float>(i,j) = levelMag.at<float>(rotatedPoint.x, rotatedPoint.y);

            
            //see here?
            float orientation = levelOrient.at<float>(rotatedPoint.x, rotatedPoint.y);
            orientROI.at<float>(i,j) = orientation;
            //debugOrientation(keypoint, rotatedPoint, orientation, levelImg, levelOrient, levelMag, i, j);
        }
    }
}

 void SIFT(std::vector<SLAM::point>& reducedKeypoints, std::vector<std::vector<float>>& featureDescriptors_vec, GaussPyramid& pyramid, int octave) {
    //Now that we have the appropriately filtered keypoints, which have a dominant orientation..
    //iterate through each keypoint, take it's primary orientation and rotate a box
    //sample the neighborhood included in the rotated box.
    int windowSize = 16;
    //1.414f is sqrt(2), which is used to pad an image according to the 
    //hypotaneuse of a square window, rather than by width & height.
    //increasing the maxPadding is resolving the issue, clearly the window is going out of bounds.

    //what if, the pixel ratio is not {1, 1, sqrt(2)} but {1, 2, sqrt(5)}?
    int maxPadding = 20;    //20 seems to be the minimum padding that prevents errors.
    //ceil(float(windowSize) * 1.414f / 2.0f);
    //22;        
    //ceil(float(windowSize) * 1.414f / 2.0f);
    //cout << "number of reduced keypoints: " << reducedKeypoints.size() << endl;

    
    std::vector<Mat> paddedBlurs = GaussPyramid::padOctave(maxPadding, pyramid.octaveBlur(octave));
    std::vector<Mat> paddedMags = GaussPyramid::padOctave(maxPadding, pyramid.octaveGradMag(octave));
    std::vector<Mat> paddedOrients = GaussPyramid::padOctave(maxPadding, pyramid.octaveGradOrient(octave));
    
    //std::vector<std::vector<float>> featureDescriptors_vec;
    for (const auto& keypoint : reducedKeypoints) {
        int level = keypoint.level;
        Mat levelImg = paddedBlurs.at(keypoint.level);
        Mat levelMag = paddedMags.at(keypoint.level);
        Mat levelOrient = paddedOrients.at(keypoint.level);
        std::vector<Point2i> rotatedPoints;
        
        //if we need to use these coordinates (of the real image) multiple times, then it might be best to get those instead.
        rotatedPoints = SLAM::Rotation::getRotatedWindowPoints(
            levelImg, 
            Point2i(keypoint.col + maxPadding, keypoint.row + maxPadding), 
            windowSize, 
            keypoint.value      //keypoint.value holds the amount to rotate (in degrees)
        );

        //draw the bounding box surrounding each keypoint. It would be nice to do this on a color image.
        //note: only draws a single bounding box.
        //DrawBoundingBox(levelImg, rotatedPoints);
        //imshow("Drawn image", levelImg);
        //waitKey(0);

        //assume given region is correct. Now get the img, gradient magnitude & orientation of this window.
        Mat imgROI = Mat::zeros(Size(windowSize, windowSize), levelImg.type());
        Mat magROI = Mat::zeros(Size(windowSize, windowSize), levelMag.type());
        Mat orientROI = Mat::zeros(Size(windowSize, windowSize), levelOrient.type());

        //this sets the image sections {imgROI, magROI, orientROI} to the rotated section of {levelImg, mag, orient}.
        rotateImageSection(levelImg, levelMag, levelOrient, imgROI, magROI, orientROI, rotatedPoints);
        
        //TODO:
        // The magnitudes are further weighted by a Gaussian function with sigma  equal to one half the width of the descriptor window.
        // The descriptor then becomes a vector of all the values of these histograms
        //TODO: just blurring the magwindow is not correct. It must only be blurred with a single convolution
        //centered on the window.
        //!!! The paper uses sigma interchangably with radius
        //"A gaussian weighting function w/ sigma = (1/2)*window_width is used to assign a weight to the mag
        //of each sample point"
        //Use this to blur the magnitude window. Note it will be 1.5x the sigma of the scale of the keypoint.
        double sigma = 1.5*pyramid.getSigmaAt(keypoint.octave, level); 
        Mat magWeighted;
        GaussianBlur(magROI, magWeighted, Size(0, 0), sigma, 0, BORDER_DEFAULT);

        //this **might** cause it to break if the reference gets lost?
        //imshow("ROI Magnitude", mat2gray(magROI));
        //imshow("rotated img ROI", imgROI);
        //imshow("ROI Orientation",mat2gray(orientROI));


        //then, create a histrogram of gradients for each 4x4 section of the ROI
        //the histogram can be performed after assignment.
        int histoSize = 8;
        int angleThresh = 360 / histoSize;
        int subregion = 4;
        SLAM::ImgRegion region = {0,0,subregion,subregion};
        std::vector<float> featureDescriptor;      //maybe should just make this an array of floats.
        featureDescriptor.reserve(128);

        //TODO: convert this into a function "iterate subregion", which takes as arg a function & other stuff.
        while (region.minRows < windowSize) {
            std::vector<float> histo = orientationHistogram(histoSize, region, magWeighted, orientROI);
            for (const auto& val : histo) {
                featureDescriptor.emplace_back(val);
            }

            region.maxCols += subregion;
            region.minCols += subregion;

            if (region.minCols == windowSize) {
                region.maxRows += subregion;
                region.minRows += subregion;
                region.minCols = 0;
                region.maxCols = subregion;
            }         
            //each time a block is done, add that histrogram of 8 values to another vector
        }
        //Now that the 128x1 featureDescriptor is complete, normalize it by the max value.
        //std::cout << "Feature descriptor size: " << featureDescriptor.size() << std::endl;


        
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


        //each featureDescriptors_vec will need to be accompanied by it's keypoint, this should allow drawing the bounding box
        //of ALL feature descriptors onto a single image.
        
        //final step is graphing and comparing two images with each other.
        
        //break;
    }
    

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

    //TODO: SIFT paper calculates these for every level of the pyramid 
    Sobel(myImages.blurred, myImages.grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(myImages.blurred, myImages.grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    magnitude(myImages.grad_x, myImages.grad_y, myImages.magnitude);
    phase(myImages.grad_x, myImages.grad_y, myImages.orientation, true);
}

void recalculateSize(SLAM::point& point) {
    float factor = pow(2.0, point.octave) / 2.0f;
    point.row = int(float(point.row) * factor);
    point.col = int(float(point.col) * factor);
}


int main() {

    std::string img_path = samples::findFile("building.jpg");
    std::string img_path2 = samples::findFile("chessboard.png");
    std::string img_path3 = samples::findFile("home.jpg");

    Mat img_color = imread(img_path3, IMREAD_COLOR);
    Mat img;
    cvtColor(img_color, img, COLOR_BGR2GRAY);
    // = imread(img_path, IMREAD_GRAYSCALE);

    int numOctaves = 4; //this represents the s variable in SIFT.
    double pyr_sigma = 1.6;                        //paper states a pyr_sigma of 1.6
    
    //the pyramid now contains images (resized), blurs, diffs, magnitude values and sigmas per level.
    GaussPyramid pyramid{img, numOctaves, pyr_sigma};
    
    //std::string octave_window = "Octave 3 Blurs";
    //GaussPyramid::showOctave(pyramid.octaveBlur(2), octave_window);

    /*  DEBUGGING - Ensure all processed images are present on each level
    std::string octave_window_gradx = "Octave 4 gradX";
    GaussPyramid::showOctave(pyramid.octaveGradX(3), octave_window_gradx);

    std::string octave_window_grady = "Octave 4 gradY";
    GaussPyramid::showOctave(pyramid.octaveGradX(3), octave_window_grady);

    std::string octave_window_mag = "Octave 4 mag";
    GaussPyramid::showOctave(pyramid.octaveGradMag(3), octave_window_mag);

    std::string octave_window_orient = "Octave 4 orient";
    GaussPyramid::showOctave(pyramid.octaveGradOrient(3), octave_window_orient);
    */
    
    

   //TODO: Now, replace the code which uses individual images, to use the full pyramid.
    SLAM::ProcessedImage myImages;
    myImages.grey = img;


    int windowSize = 3;    
    int mPadding = 20;
    Mat img_padded;
    copyMakeBorder(img, img_padded, mPadding, mPadding, mPadding, mPadding, BORDER_REPLICATE);
    //now that it is working for all levels of a single octave, extend it to work on the entire pyramid.
    std::vector<SLAM::point> reducedKeypoints_AllScales;
    
    std::vector<std::vector<float>> featureDescriptors_vec;
    for (const auto& kv: pyramid.pyramidGauss()) {
        std::vector<SLAM::point> keypoints;
        std::vector<SLAM::point> reducedKeypoints;
        int octave = kv.first;
        cout << "Pyramid octave: " << kv.first << endl;
        initialKeypointDetection(keypoints, pyramid, octave, windowSize);
        cout << "image size: width.height (" << kv.second.at(0).cols << ", " << kv.second.at(0).rows << ")" << endl;
        filterKeypoints(pyramid, octave, keypoints, reducedKeypoints);

        //So... SIFT it should have a descriptor (128 vec) for each feature.
        //this will be the complete feature descriptor vec.
        SIFT(reducedKeypoints, featureDescriptors_vec, pyramid, octave);
        //break;

        
        reducedKeypoints_AllScales.reserve(reducedKeypoints.size());
        for (auto& point : reducedKeypoints) {

            recalculateSize(point);

            std::vector<Point2i> rotatedPoints;
            
            //if we need to use these coordinates (of the real image) multiple times, then it might be best to get those instead.
            rotatedPoints = SLAM::Rotation::getRotatedWindowPoints(
                img_padded, 
                Point2i(point.col + mPadding, point.row + mPadding), 
                16, 
                point.value      //keypoint.value holds the amount to rotate (in degrees)
            );

            DrawBoundingBox(img_padded, rotatedPoints, point.octave);
            reducedKeypoints_AllScales.emplace_back(point);
        }
    }

    cout << "Number of features found in image: " << featureDescriptors_vec.size() << endl;

    
    int wSize = 16;
    
    //draw the keypoints.
    //It looks like the keypoint's rotation values are not correct anymore..
    //SLAM::point{y, x, angle, 0, keypoint.octave, keypoint.level}
    //Note, need to convert the positions of the slam point to correspond to the correct image size.
    //also note that the y and x values match row & col respectively.
    for (auto& keypoint : reducedKeypoints_AllScales) {
        //cout << "Octave: " << keypoint.octave << " Level: " << keypoint.level << " col: " << keypoint.col << " row: " << keypoint.row << endl;
        //cout << "Octave: " << keypoint.octave << " Level: " << keypoint.level << " col: " << keypoint.col << " row: " << keypoint.row <<  << endl;
        
        //draw a circle around each reduced keypoint                  
        DrawKeypoint(img_color, Point(keypoint.col,keypoint.row), keypoint.octave, 10, Scalar(0,255,255), keypoint.value, true);
        //break;
    }
 
    std::cout << "Number of keypoints after edge orientation: " << reducedKeypoints_AllScales.size() << std::endl;


    //Now, write out featureDescriptors_vec & reducedKeypoints_AllScales into binary files.
    std::string file_name = "featureDescriptors.dat";
    std::ofstream file(file_name, std::ios_base::out | std::ios_base::binary);
    if(!file)  { return EXIT_FAILURE; }

    int vecSize = featureDescriptors_vec.size();
    int histoLength = 128;
    int frontSize = sizeof(featureDescriptors_vec.front());

    file.write(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));
    file.write(reinterpret_cast<char*>(&histoLength), sizeof(histoLength));
    file.write(reinterpret_cast<char*>(&frontSize), sizeof(frontSize));

    //It is recommended instead to write out each vector one at a time.
    /*
    for (std::vector<float>& fd : featureDescriptors_vec) {
        file.write(reinterpret_cast<char*>( &fd.front() ), fd.size() * sizeof(fd.front()));
    }
    */

    //First, check for sure that data exists.
    //Test to ensure the data is infact normalized, and not just getting cut off at 1. It's working as intended.

    for (int i = 0; i < vecSize; i++) {
        //cout << "Called!" << endl;
        file.write(reinterpret_cast<char*>( &featureDescriptors_vec.at(i).front() ), featureDescriptors_vec.at(i).size() * sizeof(featureDescriptors_vec.at(i).front()));
    }

    
    

    imshow("Img", img_color);
    imshow("Drawn image", img_padded);
    moveWindow("Img", 0, 0);
    //imshow("Padded img w/ lines", paddedImg3);
    //moveWindow("Padded img w/ lines", 0, myImages.color.rows);
    int k = waitKey(0);

    return 0;

}
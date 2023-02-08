#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

//we're at the position of a single pixel, but want to compute the sum of the region around the pixel.
void StructureMatrix(Mat& M, Mat& Ix, Mat& Iy, int padding, int i, int j) {
    float Ix2 = 0.0f;
    float Iy2 = 0.0f;
    float IxIy = 0.0f;

    //we want to go from
    for (int u = i - padding; u <= i + padding; u++) {
        for (int v = j - padding; v <= j + padding; v++) {
            Ix2 += Ix.at<float>(u,v) * Ix.at<float>(u,v);
            Iy2 += Iy.at<float>(u,v) * Iy.at<float>(u,v);
            IxIy += Ix.at<float>(u,v) * Iy.at<float>(u,v);
        }
    }

    //the structure tensor will always be 2x2. It is calculated PER pixel (or region around a pixel)
    M.at<float>(0,0) = Ix2;
    M.at<float>(0,1) = IxIy;
    M.at<float>(1,0) = IxIy;
    M.at<float>(1,1) = Iy2;
}

Mat HarrisCorner(Mat& Ix, Mat& Iy) {
    int width = Ix.cols;
    int height = Ix.rows;
    int windowSize = 3;     //note, if the value is 1, then there isn't a response above 0 to detect corners.
    int padding = (windowSize-1) / 2;
    float k = 0.04; //float is CV_32F

    //Datatype is necessary for determinant
    Mat M = Mat::zeros(2,2, CV_32F);                 //2,2 matrix filled w/ 0.
    Mat image = Mat::zeros(width, height, CV_32F);
    Mat paddedIx, paddedIy;
    copyMakeBorder(Ix, paddedIx, padding, padding, padding, padding, BORDER_REPLICATE);
    copyMakeBorder(Iy, paddedIy, padding, padding, padding, padding, BORDER_REPLICATE);
    
    //1. iterate over padded matrix
    //left to right, top to bottom.
    for (int i = padding; i < paddedIx.cols - padding; i++) {
        for (int j = padding; j < paddedIx.rows - padding; j++) {
            //what if I added padding to i, j here, then subtract it in the function?
            StructureMatrix(M, paddedIx, paddedIy, padding, i, j);
            //calculate the determinant and trace of the matrix.
            //Determinant: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#determinant
            //Trace: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#Scalar%20trace(InputArray%20mtx)
            float det = determinant(M);
            float tr = trace(M)[0];        //trace returns a scalar, which is 4 doubles.

            float response = det - k * (tr*tr);

            //now, remove padding for appropriate setting.
            if (response > 0) {
                image.at<float>(i-padding,j-padding) = response;
            }
        }
    }

    //if response is negative, it's an edge, if it's 0 it's a flat region, if it's positive it's a corner.
    return image;
}

Mat NonMaximumSuppression(Mat& response, int windowSize) {
    int windowCenter = (windowSize - 1) / 2;

    Mat kernelLocalMax = Mat::ones(windowSize, windowSize, CV_8U);
    kernelLocalMax.at<uchar>(windowCenter,windowCenter) = 0;
    Mat imageLM;
    dilate(response, imageLM, kernelLocalMax);


    Mat localMax = (response > imageLM);
    return localMax;
}

Mat NMS2(Mat& response, int windowSize) {
    //now modify the data types.
    //Note height/width corresponds to rows/cols NOT cols/rows.
    Mat nms = Mat::zeros(response.rows, response.cols, CV_8U);

    std::cout << "response mat size: " << response.rows << "x" << response.cols << std::endl;
    std::cout << "nms mat size: " << nms.rows << "x" << nms.cols << std::endl;

    int padding = (windowSize - 1) / 2;
    float true_max = 0;
    
    for (int i = padding; i < response.rows - padding; i++) {
        for (int j = padding; j < response.cols - padding; j++) {
            //Check out all pixels in the neighborhood, find the max.
            float max = 0;
            
            //this might break it if we're overaccessing.
            for (int u = i - padding; u < i + padding; u++) {
                for (int v = j - padding; v < j + padding; v++) {
                    
                    if (response.at<float>(u,v) > max) {
                        //std::cout << (int)response.at<float>(u,v) << std::endl;
                        max = response.at<float>(u,v);
                        //pixels are being found.
                        if (max > true_max) {
                            true_max = max;
                        }
                    }

                }
            }

            //once we have max, see if the current pixel is the max, if so, add it to the nms mat.
            if (response.at<float>(i,j) >= max) {
                //std::cout << "Max value in local neighborhood: " << max << std::endl;
                nms.at<float>(i,j) = max;
            }
        }
    }

    //The function is working as intended now, but the next issue is that it doesn't handle cases where there are two of the same max
    //value in the area. Should this be fixed by handling the image before converting?
    //or should it handle plateau cases?

    std::cout << "True max of NMS: " << true_max << std::endl;      //now it's correctly saying 255!
    return nms;
}


void draw_them_circles(Mat& img) {
    int radius = 3;
    int line_color = 255;
    int thickness = 2;
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {

            if (img.at<float>(i,j) > 253) {
                circle(img, Point(i,j), radius, line_color, thickness);
            }
        }
    }
}

int main() {
    //try building.jpg or blox.jpg
    std::string img_path = samples::findFile("chessboard.png");
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;        //16bit signed (short)

    //now that image is loaded, take derivs.
    Mat blurred;
    GaussianBlur(img, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);

    //derivs
    Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

    Sobel(blurred, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(blurred, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    //Once we have the gradients in both directions, create the structure tensor:
    //M = sum [ Ix^2 IxIy ]
    //        [ IxIy Iy^2 ]

    //so, for each point we get a 2x2 matrix informing us if it's a corner or not.
    //iterate over the image size, set a patch size
    Mat HResponse = HarrisCorner(grad_x, grad_y);
    Mat abs_HResponse;
    
    //convert back to CV_8U
    convertScaleAbs(HResponse, abs_HResponse);

    Mat nms = NonMaximumSuppression(abs_HResponse, 3);          
    Mat nms2 = NMS2(HResponse, 5); //try using the non abs scaled for this. Working
    Mat abs_NMS;
    convertScaleAbs(nms2, abs_NMS);
    draw_them_circles(abs_NMS);

    imshow("Harris", abs_NMS);
    int k = waitKey(0);

    if (k == 's') {
        imwrite("Harris.jpg", abs_NMS);
    }

    return 0;

}
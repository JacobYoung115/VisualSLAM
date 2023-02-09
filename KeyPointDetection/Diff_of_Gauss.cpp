#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>

using namespace cv;

/*
Notes: first just do it in a way that I understand
then, go back and look at the imgproc tutorials and see ways I can optimize this.

*/
namespace SLAM {
    struct point {
        point(int pos_x_, int pos_y_, int value_) : pos_x(pos_x_), pos_y(pos_y_), value(value_) {}
        int pos_x;
        int pos_y;
        int value;
    };
}

int main() {

    std::string img_path = samples::findFile("blox.jpg");
    std::string img_path2 = samples::findFile("building.jpg");

    Mat img = imread(img_path2, IMREAD_GRAYSCALE);

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
    for (const auto& image : dogs) {
        Mat paddedImg;
        copyMakeBorder(image, paddedImg, padding, padding, padding, padding, BORDER_REPLICATE);
        dogs_padded.emplace_back(std::move(paddedImg));        //not sure if this is adding the refence to the old image or copying a new image in.
    }

    int extrema = 0;
    
    

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

            //Feature Point Localization:
            //https://www.youtube.com/watch?v=LXk4A24V8mQ
            //It filters out additional points by interpolating between positions and scale space to approximate a 3D surface.
            //effectively this means taking the difference of points in:
            //  x-1,        x+1 direction
            //  y-1,        y+1 direction
            //  adjacent-1, adjacent+1 direction
            int d_x = (int)dogs_padded[1].at<uchar>(i,j-1) - (int)dogs_padded[1].at<uchar>(i,j+1);
            int d_y = (int)dogs_padded[1].at<uchar>(i-1,j) - (int)dogs_padded[1].at<uchar>(i + 1,j);
            int d_scale = (int)dogs_padded[0].at<uchar>(i,j) - (int)dogs_padded[2].at<uchar>(i,j);
            

            
            /*
            std::cout << "d_x: " << d_x << std::endl;
            std::cout << "d_y: " << d_y << std::endl; 
            std::cout << "d_scale: " << d_scale << std::endl;
            */
            
            

            //int fpl = this_pixel + 0.5 * 

            
            if (this_pixel == min || this_pixel == max) {
                //Now, Perform 'Feature Point Localization'
                //Note that in the SIFT paper, they consider intensity values less than 0.03 in the scale of [0-1] to be low contrast
                //0.03 * 255 = 7.65
                cv::Vec3f A_vec((float)d_x / 255.0f, (float)d_y / 255.0f, (float)d_scale / 255.0f);
                cv::Mat A = Mat(A_vec);     //3x1
                cv::Mat A_T;                                                            
                transpose(A, A_T);          //1x3
                
                Mat B = A*A_T;              //A (3x1) * A^T (1x3) = 3x3 matrix.
                Mat B_inverse = -(B.inv());
                
                Mat z_hat = B_inverse * A;        //3x3 * 3x1 = 3x1
                //std::cout << "z_hat" << std::endl << z_hat << std::endl;
                
                Mat dog_zhat_mat = ((float)this_pixel / 255.0f) + 0.5f * A_T * z_hat;
                //std::cout << "dog_zhat_mat" << std::endl << dog_zhat_mat << std::endl;
                float dog_zhat = dog_zhat_mat.at<float>(0,0);
                
                if (dog_zhat > 0.03f) {     //if the value is above 0.03, then it is a high enough contrast point.
                    keypoints.emplace_back( SLAM::point{i,j,(int)(dog_zhat * 255)} );
                    DoG1.at<uchar>(i,j) = (uchar)255;
                }
                
                
                if (this_pixel > 0) {
                    extrema++;
                    //
                    
                }
                //
            }

            /*
            std::cout << "Neighbor pixels: [";
            for (const auto& item: neighbors) {
                std::cout << item << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Min value: " << min << std::endl;
            std::cout << "Max value: " << max << std::endl;
            std::cout << "Current pixel's value: " << this_pixel << std::endl;
            */
            //break;
        }
        //break;
    }
    
    
    std::cout << "Number of extrema detected in DoG2 (old method): " << extrema << std::endl;
    std::cout << "Number of keypoints (new method): " << keypoints.size() << std::endl;

    //Now, further filter the keypoints (Feature Point Localization)
    

    //now, look for extrema which are different in the more&less smoothed images and also in the x&y directions.


    //For DoG we also need the deriv in x&y directions, to additionally calculate the gradient direction
    //derivs
    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;        //16bit signed (short)
    Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
    Sobel(blurred1, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(blurred1, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);


    imshow("DoG1", DoG1);
    int k = waitKey(0);

    if (k == 's') {
        imwrite("DoG1.jpg", DoG1);
    }

    return 0;

}
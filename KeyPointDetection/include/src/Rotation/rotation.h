#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace cv;

namespace SLAM {
    class Transform {};

    class Rotation : public Transform {
        public:
        static float convertToRadians(float theta);
        static const Point2f cos_sin_of_angle(float theta, bool degrees=true);
        static Point2i rotate_pt_CW(const Point2i& pt, const Point2i& center, const Point2f& angles);
        static Point2i rotate_pt_CCW(const Point2i& pt, const Point2i& center, const Point2f& angles);
        static Point2i rotate_pt_CCW(const Point2i& pt, const Point2i& center, float theta, bool degrees=true);
        static Point2i rotate_pt_CW(const Point2i& pt, const Point2i& center, float theta, bool degrees=true);
        static Mat rotate_mat_CCW(Mat& I, const Point2i& center, const Point2f& angles);
        static Mat getRotatedWindow(Mat& I, const Point2i& center, int windowSize, float theta, bool degrees=true);
        static std::vector<Point2i> getRotatedWindowPoints(Mat& I, const Point2i& center, int windowSize, float theta, bool degrees=true);
        static Point drawRotated(Point& pt, Mat& src, Mat& roi_rotation_mat, bool draw=true);
        static Mat doubleCrop(Mat& src, const Point2i& center, int windowSize, double angle);
    };
} //namespace SLAM
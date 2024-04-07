#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void drawShapesWithLabels(const vector<vector<Point>>& contours, Mat& contour_image) 
{
    for (size_t i = 0; i < contours.size(); i++) 
    {
        vector<Point> approxCurve;
        approxPolyDP(contours[i], approxCurve, 0.03 * arcLength(contours[i], true), true);
        drawContours(contour_image, contours, static_cast<int>(i), Scalar(255, 0, 0), 2);
        int verticesCount = approxCurve.size();
        string Name;
        if (verticesCount == 3)
            Name = "triangle";
        else if (verticesCount == 4)
            Name = "square";
        else
            Name = "circle";
        Moments mu = moments(contours[i]);
        Point centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);
        putText(contour_image, Name, centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }
}

int main() 
{
    Mat image = imread("image1.jpg");
    Mat blurred, edges;
    GaussianBlur(image, blurred, Size(5, 5), 0);
    Canny(blurred, edges, 30, 150);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contour_image = Mat::zeros(image.size(), CV_8UC3);
    drawShapesWithLabels(contours, contour_image);
    imshow("Contours", contour_image);
    waitKey(0);
    return 0;
}

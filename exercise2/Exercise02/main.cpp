
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

/**
 * @brief Draw rectangle of estimated homography in the scene image
 * @param [in]    imgObjSize : image size of object image
 * @param [in]    homography : homography matrix (3x3)
 * @param [inout] imgMatches : scene image
 */
void drawHomographyRect(Size imgObjSize, Mat &homography, Mat &imgMatches)
{
    if (homography.empty() || imgMatches.cols == 0)
    {
        return;
    }

    // define object corners in the object image
    vector<Point2d> cornersObj(4);
    cornersObj[0] = Point2d(0, 0);
    cornersObj[1] = Point2d(imgObjSize.width, 0);
    cornersObj[2] = Point2d(imgObjSize.width, imgObjSize.height);
    cornersObj[3] = Point2d(0, imgObjSize.height);

    // compute object corners in the scene image using homography
    vector<Point2d> cornersScene(4);

    // draw rectangle of detected object
}

int main(int argc, char *argv[])
{
    string projectSrcDir = PROJECT_SOURCE_DIR;

    // Load images
    Mat imgObj = imread(projectSrcDir + "/Data/chocolate1.png");
    Mat imgScene = imread(projectSrcDir + "/Data/chocolate_scene.png");

    // Step a)1: Detect keypoints and extract descriptors
    Mat outObj;
    Mat outScene;
    vector<KeyPoint> keypointsObj;
    vector<KeyPoint> keypointsScene;
    Ptr<ORB> detector = ORB::create();
    detector->detect(imgObj, keypointsObj);
    detector->detect(imgScene, keypointsScene);

    // Step a)2: Draw keypoints
    drawKeypoints(imgObj, keypointsObj, outObj);
    drawKeypoints(imgScene, keypointsScene, outScene);

    // Step b)1: Match descriptors
    Mat descriptorsObj;
    Mat descriptorsScene;
    detector->compute(imgObj, keypointsObj, descriptorsObj);
    detector->compute(imgScene, keypointsScene, descriptorsScene);

    // Step b)2: Display corresponding pair
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptorsObj, descriptorsScene, matches,2);
    Mat img_matches1;
    drawMatches(imgObj, keypointsObj, imgScene, keypointsScene, matches, img_matches1);

    imshow("result",img_matches1);
    waitKey(0);

    // Step c)1: Compute homography with RANSAC

    // Step c)2: Display inlier matches

    // Step c)3: Display object rectangle in the scene using drawHomographyRect()

    // Step d: Overlay another object on the detected object in the scene for augmented reality
    Mat arObj = imread(projectSrcDir + "/Data/chocolate2.png");

#if 0
    // Step e: Implement live demo for the object detector, using web-camera
    VideoCapture cap;
    cap.open(0);
    while (true)
    {
        cap >> imgScene;

        // Detect planer object, using the code from step a) to d).
        // show the detection result and overlay AR image
        imshow( "scene image", imgScene);
 
        
        
        
        
        
        int key = waitKey(1);
        if (key=='q' || key==27){
            break;
        }
    }
#endif

    return 0;
}

<<<<<<< HEAD

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
=======
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
#include <thread>
using namespace std;
using namespace cv;


/**
* @brief Draw rectangle of estimated homography in the scene image
* @param [in]    imgObjSize : image size of object image
* @param [in]    homography : homography matrix (3x3)
* @param [inout] imgMatches : scene image
*/
void drawHomographyRect(Size imgObjSize, Mat& homography, Mat& imgMatches)
{
	if (homography.empty() || imgMatches.cols == 0) {
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
	perspectiveTransform(cornersObj, cornersScene, homography);



	// draw rectangle of detected object

	line(imgMatches, cornersScene[0] + Point2d(imgObjSize.width, 0), cornersScene[1] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[1] + Point2d(imgObjSize.width, 0), cornersScene[2] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[2] + Point2d(imgObjSize.width, 0), cornersScene[3] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[3] + Point2d(imgObjSize.width, 0), cornersScene[0] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);

	/*
	imshow("imgScene", imgMatches);
	waitKey();
	*/
}

Mat DoTheThing(Mat imgObj, Mat imgScene, Mat arObj) {

	Mat outObj;
	Mat outScene;

	// Step a)1: Detect keypoints and extract descriptors
	vector<KeyPoint> keypointsObj, keypointsScene;
	Ptr<ORB> detector = ORB::create();
	detector->detect(imgObj, keypointsObj);
	detector->detect(imgScene, keypointsScene);

	// Step a)2: Draw keypoints
	drawKeypoints(imgObj, keypointsObj, outObj);
	/*	imshow("outObj", outObj);
	waitKey();
	*/
	drawKeypoints(imgScene, keypointsScene, outScene);

	/*	imshow("outScene", outScene);
	waitKey();
	*/

	// Step b)1: Match descriptors
	Mat descriptorsObj, descriptorsScene;
	detector->compute(imgObj, keypointsObj, descriptorsObj);
	detector->compute(imgScene, keypointsScene, descriptorsScene);


	BFMatcher matcher(NORM_L2);
	vector< DMatch > matches;
	matcher.match(descriptorsObj, descriptorsScene, matches);

	Mat imgMatches;
	drawMatches(imgObj, keypointsObj, imgScene, keypointsScene, matches, imgMatches);

	// Step b)2: Display corresponding pair
	/*	imshow("imgMatches", imgMatches);
	waitKey();*/
	// Step c)1: Compute homography with RANSAC
	Mat masks;
	//-- Localize the object
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (size_t i = 0; i < keypointsObj.size(); i++)
	{
		obj.push_back(keypointsObj[matches[i].queryIdx].pt);
		scene.push_back(keypointsScene[matches[i].trainIdx].pt);

	}

	Mat homography = findHomography(obj, scene, RANSAC, 3.0, masks);
	cout << determinant(homography);

	vector<KeyPoint>filteredKPObj, filteredKPScene;
	vector<DMatch>  matchesFiltered;
	vector<Point2f> filteredObj;
	vector<Point2f> filteredScene;
	for (int i = 0; i < masks.rows; i++)
	{
		if ((unsigned int)masks.at<uchar>(i)) {
			filteredKPObj.push_back(keypointsObj[i]);
			filteredKPScene.push_back(keypointsScene[i]);
			matchesFiltered.push_back(matches[i]);
			filteredObj.push_back(keypointsObj[matches[i].queryIdx].pt);
			filteredScene.push_back(keypointsScene[matches[i].trainIdx].pt);
		}
	}
	if (filteredObj.size()<30)
	{
		return imgScene;
	}
	//	homography = findHomography(filteredObj, filteredScene, LMEDS);

	cout << determinant(homography);


	// Step c)2: Display inlier matches
	Mat imgMatchesFiltered;
	drawMatches(imgObj, keypointsObj, imgScene, keypointsScene, matchesFiltered, imgMatchesFiltered);
	/*imshow("imgMatchesFiltered", imgMatchesFiltered);
	waitKey();*/
	// Step c)3: Display object rectangle in the scene using drawHomographyRect()
	drawHomographyRect(Size(imgObj.cols, imgObj.rows), homography, imgMatchesFiltered);

	// Step d: Overlay another object on the detected object in the scene for augmented reality
	//Mat arObj = imread(projectSrcDir + "/Data/chocolate3.jpg");
	Mat imgSceneCopy = imgScene.clone();
	Mat resultImg(imgScene.rows, imgScene.cols, CV_8UC3);

	warpPerspective(arObj, resultImg, homography, Size(imgScene.cols, imgScene.rows));



	//	resultImg.copyTo(imgScene, resultImg);
	//	fillPoly(imgScene, points,4,0);
	/*
	imshow("scene", imgScene);
	waitKey();*/
	return resultImg | imgScene;
}

int main(int argc, char* argv[])
{
	string projectSrcDir = PROJECT_DIR; //"C:/Users/ahmad/Downloads/exercise2";
	string videoStreamAddress = "http://131.159.208.121:8080/shot.jpg";
	// Load images
	Mat imgObj = imread(projectSrcDir + "/Data/chocolate1.png");
	Mat arObj = imread(projectSrcDir + "/Data/chocolate4.jpg");
	//Mat imgScene = imread(projectSrcDir + "/Data/chocolate_scene.png");


	// Step e: Implement live demo for the object detector, using web-camera
	VideoCapture cap;
	// give camera some extra time to get ready:
	//std::this_thread::sleep_for(std::chrono::milliseconds(200));
	if (!cap.open(videoStreamAddress)) // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	while (true)
	{
		//Mat imgScene = imread(projectSrcDir + "/Data/chocolate_scene.png");;

		//
		Mat imgScene;//; = Mat();
		cap.open(videoStreamAddress);
		//cap >> imgScene;
		cap.read(imgScene);
		/*while (countNonZero(imgScene) == 0)
		{
		cap.read(imgScene);

		}
		*/
		// Detect planer object, using the code from step a) to d).
		// show the detection result and overlay AR image
		//imshow("scene image", imgScene);

		imshow("scene image", DoTheThing(imgObj, imgScene, arObj));






		int key = waitKey(1);
		if (key == 'q' || key == 27) {
			break;
		}
	}


	return 0;
}





>>>>>>> 1c27d51ff045689166a82aafab1f18b84ec7fbd2

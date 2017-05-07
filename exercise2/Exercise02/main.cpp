#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <thread>
#include <chrono>
int THRESHOLD = 20;

using namespace std;
using namespace cv;

vector<Point2d> getHomographicTransformedPoints(Size imgObjSize, Mat &homography)
{
	// define object corners in the object image
	vector<Point2d> cornersObj(4);
	cornersObj[0] = Point2d(0, 0);
	cornersObj[1] = Point2d(imgObjSize.width, 0);
	cornersObj[2] = Point2d(imgObjSize.width, imgObjSize.height);
	cornersObj[3] = Point2d(0, imgObjSize.height);

	// compute object corners in the scene image using homography
	vector<Point2d> cornersScene(4);
	perspectiveTransform(cornersObj, cornersScene, homography);

	return cornersScene;
}

/**
* @brief Draw rectangle of estimated homography in the scene image
* @param [in]    imgObjSize : image size of object image
* @param [in]    homography : homography matrix (3x3)
* @param [inout] imgMatches : scene image
*/
void drawHomographyRect(Size imgObjSize, Mat &homography, Mat &imgMatches, bool showSteps = false)
{
	if (homography.empty() || imgMatches.cols == 0)
	{
		return;
	}

	auto cornersScene = getHomographicTransformedPoints(imgObjSize, homography);

	// draw rectangle of detected object
	// define color BGR (current RED)
	line(imgMatches, cornersScene[0] + Point2d(imgObjSize.width, 0), cornersScene[1] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[1] + Point2d(imgObjSize.width, 0), cornersScene[2] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[2] + Point2d(imgObjSize.width, 0), cornersScene[3] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);
	line(imgMatches, cornersScene[3] + Point2d(imgObjSize.width, 0), cornersScene[0] + Point2d(imgObjSize.width, 0), Scalar(0, 0, 255), 4);

	if (showSteps)
	{
		imshow("imgScene", imgMatches);
		waitKey();
		cvDestroyWindow("imgScene");
	}
}

/**
* @brief given a template image, a scene and a replacement image, this function matches the
*        template in the scene and replaces it with the replacement
* @param [in]    imgObj: A template to match for
* @param [in]    imgScene: image where we match for the template
* @param [in]    showSteps: showing all steps in the process (for debugging)
  @param [out]   imgScene: The result of the match process and the replacement
*/
void detectChocolate(Mat imgObj, Mat &imgScene, Mat arObj, bool showSteps = false)
{

	Mat outObj;
	Mat outScene;

	// Step a)1: Detect keypoints and extract descriptors
	vector<KeyPoint> keypointsObj, keypointsScene;
	Ptr<ORB> detector = ORB::create();
	detector->detect(imgObj, keypointsObj);
	detector->detect(imgScene, keypointsScene);

	// Step a)2: Draw keypoints
	if (showSteps)
	{
		drawKeypoints(imgObj, keypointsObj, outObj);
		imshow("outObj", outObj);
		waitKey();
		cvDestroyWindow("outObj");

		drawKeypoints(imgScene, keypointsScene, outScene);

		imshow("outScene", outScene);
		waitKey();
		cvDestroyWindow("outScene");

	}

	// Step b)1: Match descriptors
	Mat descriptorsObj, descriptorsScene;
	detector->compute(imgObj, keypointsObj, descriptorsObj);
	detector->compute(imgScene, keypointsScene, descriptorsScene);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptorsObj, descriptorsScene, matches);

	Mat imgMatches;
	// Step b)2: Display corresponding pair
	if (showSteps)
	{
		drawMatches(imgObj, keypointsObj, imgScene, keypointsScene, matches, imgMatches);
		imshow("imgMatches", imgMatches);
		waitKey();
		cvDestroyWindow("imgMatches");

	}
	// Step c)1: Compute homography with RANSAC
	Mat masks;
	//-- Localize the object
	vector<Point2f> objPoints;
	vector<Point2f> scenePoints;
	for (size_t i = 0; i < keypointsObj.size(); i++)
	{
		objPoints.push_back(keypointsObj[matches[i].queryIdx].pt);
		scenePoints.push_back(keypointsScene[matches[i].trainIdx].pt);
	}

	Mat homography = findHomography(objPoints, scenePoints, RANSAC, 3.0, masks);
	double homodet = determinant(homography);
	cout << "determinant of the homography: " << homodet << endl;
	if (homodet<0)
	{
		
		return;

	}
	// Step c)2: Display inlier matches
	//vector<KeyPoint> filteredKPObj, filteredKPScene;
	vector<DMatch> matchesFiltered;
	for (int i = 0; i < masks.rows; i++)
	{
		if ((unsigned int)masks.at<uchar>(i))
		{
			//filteredKPObj.push_back(keypointsObj[i]);
			//filteredKPScene.push_back(keypointsScene[i]);
			matchesFiltered.push_back(matches[i]);
		}
	}

	// changing of threshold is possible
	// if the amount of inlier keypoints is less than the treshold
	// then quit and return unchanged image
	// otherwise we continue with the replacement process
	cout << "numberof matches afterfiltering: " << matchesFiltered.size() << endl;
	if (matchesFiltered.size() < THRESHOLD)
	{
		return;
	}

	if (showSteps)
	{
		Mat imgMatchesFiltered;
		drawMatches(imgObj, keypointsObj, imgScene, keypointsScene, matchesFiltered, imgMatchesFiltered);
		imshow("imgMatchesFiltered", imgMatchesFiltered);
		waitKey();
		cvDestroyWindow("imgMatchesFiltered");

		// Step c)3: Display object rectangle in the scene using drawHomographyRect()
		drawHomographyRect(Size(imgObj.cols, imgObj.rows), homography, imgMatchesFiltered, showSteps);
	}

	// Step d: Overlay another object on the detected object in the scene for augmented reality
	Mat resultImg(imgScene.rows, imgScene.cols, CV_8UC3);

	warpPerspective(arObj, resultImg, homography, Size(imgScene.cols, imgScene.rows));

	auto cornersScene = getHomographicTransformedPoints(Size(arObj.cols, arObj.rows), homography);

	vector<Point> cornersPoint;
	for (int i = 0; i < cornersScene.size(); i++)
	{
		cornersPoint.push_back(Point((int)cornersScene[i].x, (int)cornersScene[i].y));
	}
	
	fillConvexPoly(imgScene, cornersPoint, 4, 0);
	imgScene = imgScene | resultImg;
	//return resultImg | imgScene;
}

int main(int argc, char *argv[])
{
	string projectSrcDir = PROJECT_SOURCE_DIR;
	// Load images
	Mat imgObj = imread(projectSrcDir + "/Data/chocolate1.png"); // template we are searching for.
	Mat arObj = imread(projectSrcDir + "/Data/chocolate2.png"); // image that will be in the place of the template.
	Mat imgScene = imread(projectSrcDir + "/Data/chocolate_scene.png");
	detectChocolate(imgObj, imgScene, arObj, true);
	imshow("scene image", imgScene);
	waitKey();
	cvDestroyWindow("scene image");


	// Step e: Implement live demo for the object detector, using web-camera
	VideoCapture cap;

	if (!cap.open(0)) // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	do
	{
		imgScene.release(); // will free memory

		cap.read(imgScene);
		// alternative:
		//cap >> imgScene;

		// Detect planer object, using the code from step a) to d).
		// show the detection result and overlay AR image
		//imshow("scene image", imgScene);

		// last parameter: show steps (true/false)
		try
		{
			detectChocolate(imgObj, imgScene, arObj);
			imshow("scene image", imgScene);
		}
		catch (Exception)
		{
			cout << "Aborting process. Problem with frame occured." << endl;
			imshow("scene image", imgScene);


			continue;
		}

		int key = waitKey(1);
		if (key == 'q' || key == 27)
		{
			break;
		}

	} while (true);

	return 0;
}

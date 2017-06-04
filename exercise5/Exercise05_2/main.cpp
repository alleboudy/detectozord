#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include <OpenNI.h>
#include <PS1080.h>
#include<cmath>
using namespace cv;
using namespace std;
using namespace openni;
/**
* @brief Saving point clouds data as ply file
* @param [in] filename : filename for saving point clouds (the extention should be .ply)
* @param [in] vertices : list of vertex point
* @param [in] colors   : list of vertex color
* @return Success:true, Failure:false
*/
bool savePointCloudsPLY(std::string filename, std::vector<Eigen::Vector4f>& vertices, std::vector<cv::Vec3b>& colors)
{
	std::ofstream fout;
	fout.open(filename.c_str());
	if (fout.fail()){
		std::cerr << "file open error:" << filename << std::endl;
		return false;
	}

	int pointNum = vertices.size();

	fout << "ply" << std::endl;
	fout << "format ascii 1.0" << std::endl;
	fout << "element vertex " << pointNum << std::endl;
	fout << "property float x" << std::endl;
	fout << "property float y" << std::endl;
	fout << "property float z" << std::endl;
	fout << "property uchar red" << std::endl;
	fout << "property uchar green" << std::endl;
	fout << "property uchar blue" << std::endl;
	fout << "property uchar alpha" << std::endl;
	fout << "end_header" << std::endl;

	for (int i = 0; i < pointNum; i++){
		Eigen::Vector4f& vertex = vertices[i];
		cv::Vec3b& col = colors[i];

		fout << vertex[0] << " " << vertex[1] << " " << vertex[2] << " " << static_cast<int>(col[2]) << " " << static_cast<int>(col[1]) << " " << static_cast<int>(col[0]) << " " << 255 << std::endl;
	}

	fout.close();

	return true;
}


// Convert to colored depth image
cv::Mat convColoredDepth(cv::Mat& depthImg, float minThresh = 0, float maxThresh = 0){
	cv::Mat coloredDepth = depthImg.clone();

	double min;
	double max;
	if (minThresh == 0 && maxThresh == 0){
		cv::minMaxIdx(depthImg, &min, &max);
	}
	else{
		min = minThresh;
		max = maxThresh;
	}
	coloredDepth -= min;
	cv::convertScaleAbs(coloredDepth, coloredDepth, 255 / (max - min));
	cv::applyColorMap(coloredDepth, coloredDepth, cv::COLORMAP_JET);

	return coloredDepth;
}



int main(int argc, char* argv[])
{
	int WIDTH = 640, HEIGHT = 480;
	////
	// Initialize OpenNI video capture
	openni::Status status;
	status = OpenNI::initialize();
	if (status != openni::STATUS_OK)
	{
		std::cout << "Error initializing!" << std::endl;
		std::cout << status << std::endl;
		std::cout << OpenNI::getExtendedError() << std::endl;

	}
	Device niDevice;
	status = niDevice.open(ANY_DEVICE);
	if (status != openni::STATUS_OK)
	{
		std::cout << "Error opening device!" << std::endl;
		std::cout << status << std::endl;
	}
	VideoMode cameraMode;
	cameraMode.setResolution(WIDTH, HEIGHT);
	cameraMode.setFps(30);
	// Create depth camera stream
	VideoStream niDepthStream;
	niDepthStream.create(niDevice, SENSOR_DEPTH);
	niDepthStream.setVideoMode(cameraMode);
	niDepthStream.start();
	// Create color camera stream
	VideoStream niColorStream;
	niColorStream.create(niDevice, SENSOR_COLOR);
	//niColorStream.setVideoMode(cameraMode);
	niColorStream.start();

	////
	// Set flag for synchronization between color camera and depth camera
	niDevice.setDepthColorSyncEnabled(true);
	// Set flag for registration between color camera and depth camera
	niDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	////
	// Get focal length of IR camera in mm for VGA resolution
	double pixelSize = 0;
	niDepthStream.getProperty<double>(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE, &pixelSize);
	// Convert focal length from mm -> pixels (valid for 640x480)
	pixelSize *= 2.0; // in mm, valid for VGA resolution
	int zeroPlaneDistance; // focal in mm
	niDepthStream.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE, &zeroPlaneDistance);
	double depthFocalLength_VGA = zeroPlaneDistance / pixelSize;



	float minRange = 0;
	float maxRange = 4000;
	VideoFrameRef niDepth, niColor;
	int nCols = niDepthStream.getVideoMode().getResolutionX();
	int nRows = niDepthStream.getVideoMode().getResolutionY();
	// compute principal points (just image size / 2)

	float px = nCols / 2.;
	float py = nRows / 2.;
	// Start loop
	cv::Mat depthImage, matDepth16U;

	vector<Mat> hundredDepths; // a buffer of 100 depth frames that we use to compute the std and mean of the depthstream 

	while (true)
	{

		////
		// Grab depth image
		niDepthStream.readFrame(&niDepth);
		if (niDepth.isValid()) {
			matDepth16U = cv::Mat(nRows, nCols, CV_16UC1, (char*)niDepth.getData());
			matDepth16U.convertTo(depthImage, CV_32FC1);
			cv::flip(depthImage, depthImage, 1);

			if (hundredDepths.size() == 100)
			{
				hundredDepths.erase(hundredDepths.begin());
			}

			if (hundredDepths.size() < 100)
			{
				hundredDepths.push_back(depthImage.clone());
			}


		}




		// Grab color image
		cv::Mat colorImage;

		niColorStream.readFrame(&niColor);
		if (niColor.isValid()) {
			colorImage = cv::Mat(nRows, nCols, CV_8UC3, (char*)niColor.getData());
			cv::flip(colorImage, colorImage, 1);
		}

		// Show images
		cv::cvtColor(colorImage, colorImage, CV_BGR2RGB); //this will put colors right
		cv::imshow("colorImage", colorImage);
		cv::imshow("depthImage", convColoredDepth(depthImage, minRange, maxRange));
		//cv::imwrite("depthImage.png", convColoredDepth(depthImage, minRange, maxRange));

		int key = cv::waitKey(10);

		////
		// Convert the depth iamge to point clouds and save it.
		if (key == 's')
		{
			vector<Eigen::Vector4f> vertices;
			vector<Vec3b> colors;
			//DepthImg2PointCloud(matDepth16U, colorImage, vertices, colors, depthFocalLength_VGA, px, py);

			for (int j = 0; j < matDepth16U.cols; j++)
			{
				for (int i = 0; i < matDepth16U.rows; i++)
				{
					auto point = Eigen::Vector4f((j - px)*matDepth16U.at<ushort>(i, j) / depthFocalLength_VGA, (i - py)*matDepth16U.at<ushort>(i, j) / depthFocalLength_VGA, matDepth16U.at<ushort>(i, j), 1);

					// (3) Add the 3D point to vertices in point clouds data.
					vertices.push_back(point);
					// (4) Also compute the color of 3D point and add it to colors in point clouds data.
					colors.push_back(colorImage.at<Vec3b>(i, j));

				}
			}

			savePointCloudsPLY("SavedpointClouds.ply", vertices, colors);
		}

		////
		// Compute standard deviation of depth stream and save it
		if (key == 'd')
		{

			if (hundredDepths.size() == 100)
			{
				Mat stnd(Size(hundredDepths[0].cols, hundredDepths[0].rows), CV_32FC1), mean(Size(hundredDepths[0].cols, hundredDepths[0].rows), CV_32FC1);
				mean *= 0.0;
				stnd *= 0.0;

				for (int i = 0; i < hundredDepths.size(); i++)
				{
					mean = mean + hundredDepths[i];
				}

				mean /= 100.;

				//	cout << mean << endl;


				for (int i = 0; i < hundredDepths.size(); i++)
				{
					for (int x = 0; x < stnd.rows; x++)
					{
						for (int y = 0; y < stnd.cols; y++)
						{

							stnd.at<float>(x, y) = stnd.at<float>(x, y) + pow((hundredDepths[i].at<float>(x, y) - mean.at<float>(x, y)), 2);
						}
					}


				}
				stnd /= 100.;
				//cout << stnd;
				for (int x = 0; x < stnd.rows; x++)
				{
					for (int y = 0; y < stnd.cols; y++)
					{
						stnd.at<float>(x, y) = sqrt(stnd.at<float>(x, y));
					}
				}
				imwrite("std.png", convColoredDepth(stnd, minRange, maxRange));
				imwrite("mean.png", convColoredDepth(mean, minRange, maxRange));




				vector<Eigen::Vector4f> vertices;
				vector<Vec3b> colors;
				//DepthImg2PointCloud(matDepth16U, colorImage, vertices, colors, depthFocalLength_VGA, px, py);
				
				for (int j = 0; j < mean.cols; j++)
				{
					for (int i = 0; i < mean.rows; i++)
					{
						auto point = Eigen::Vector4f((j - px)*mean.at<float>(i, j) / depthFocalLength_VGA, (i - py)*mean.at<float>(i, j) / depthFocalLength_VGA, mean.at<float>(i, j), 1);

						// (3) Add the 3D point to vertices in point clouds data.
						vertices.push_back(point);
						// (4) Also compute the color of 3D point and add it to colors in point clouds data.
						Vec3b b;
						b[0] = b[1] = b[2] = stnd.at<float>(i, j);
						colors.push_back(b);

					}
				}

				savePointCloudsPLY("mean.ply", vertices, colors);



			}
			else
			{
				cout << "Buffered only " << hundredDepths.size() << " still need " << 100 - hundredDepths.size() << " wait a bit and push d again" << endl;
			}

		}

		////
		// Compute mesh of point clouds and save it
		if (key == 'm'){
		}

		if (key == 'q' || key == 27){
			break;
		}
	}

	////
	// destroy image streams and close the OpenNI device
	niDepthStream.destroy();
	niColorStream.destroy();
	//for all open streams
	niDevice.close();
	//for all open 
	OpenNI::shutdown();

	return 0;


}







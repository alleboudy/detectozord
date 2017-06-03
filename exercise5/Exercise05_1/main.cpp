#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <Eigen/Dense>

using namespace std;
using namespace cv;
/**
 * @brief Saving point clouds data as ply file
 * @param [in] filename : filename for saving point clouds (the extention should be .ply)
 * @param [in] vertices : list of vertex point
 * @param [in] colors   : list of vertex color
 * @return Success:true, Failure:false
 */
bool savePointCloudsPLY(string filename, vector<Eigen::Vector4f>& vertices, vector<Vec3b>& colors)
{
	ofstream fout;
	fout.open(filename.c_str());
	if (fout.fail()){
		cerr << "file open error:" << filename << endl;
		return false;
	}

	int pointNum = vertices.size();

	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "element vertex " << pointNum << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "property uchar red" << endl;
	fout << "property uchar green" << endl;
	fout << "property uchar blue" << endl;
	fout << "property uchar alpha" << endl;
	fout << "end_header" << endl;

	for (int i = 0; i < pointNum; i++){
		Eigen::Vector4f& vertex = vertices[i];
		Vec3b& col = colors[i];

		fout << vertex[0] << " " << vertex[1] << " " << vertex[2] << " " << static_cast<int>(col[2]) << " " << static_cast<int>(col[1]) << " " << static_cast<int>(col[0]) << " " << 255 << endl;
	}

	fout.close();

	return true;
}


/**
 * @brief Loading camera pose file (6Dof rigid transformation)
 * @param [in]  filename : filename for loading text file
 * @param [out] pose   : 4x4 transformation matrix
 * @return Success:true, Failure:false
 */
bool loadCameraPose(string filename, Eigen::Matrix4f& poseMat)
{
	ifstream fin;
	fin.open(filename.c_str());
	if (fin.fail()){
		cerr << "file open error:" << filename << endl;
		return false;
	}

	// Loading 4x4 transformation matrix
	for (int y = 0; y < 4; y++){
		for (int x = 0; x < 4; x++){
			fin >> poseMat(y, x);
		}
	}
	return true;
}



int main(int argc, char* argv[])
{
	string projectSrcDir = PROJECT_SOURCE_DIR;

	// Data for point clouds consisted of 3D points and their colors
	vector<Eigen::Vector4f> vertices; // 3D points
	vector<Vec3b> colors;   // color of the points

	for (int index = 0; index <= 5; index++)
	{


		// Loading depth image and color image
		//int index = 5;
		string depthFilename = projectSrcDir + "/data/depth/depth" + to_string(index) + ".png";
		string colorFilename = projectSrcDir + "/data/color/color" + to_string(index) + ".png";
		Mat depthImg = imread(depthFilename, CV_LOAD_IMAGE_UNCHANGED);
		Mat colorImg = imread(colorFilename, CV_LOAD_IMAGE_UNCHANGED);

		// Loading camera pose
		string poseFilename = projectSrcDir + "/data/pose/pose" + to_string(index) + ".txt";
		Eigen::Matrix4f poseMat;   // 4x4 transformation matrix
		loadCameraPose(poseFilename, poseMat);
		cout << "Transformation matrix" << endl << poseMat << endl;

		// Setting camera intrinsic parameters of depth camera
		float focal = 570.f;  // focal length
		float px = 319.5f; // principal point x
		float py = 239.5f; // principal point y



		// Create point clouds from depth image and color image using camera intrinsic parameters
		// (1) Compute 3D point from depth values and pixel locations on depth image using camera intrinsic parameters.
		for (int j = 0; j < depthImg.cols; j++)
		{
			for (int i = 0; i < depthImg.rows; i++)
			{
				auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focal, (i-py)*depthImg.at<ushort>(i, j) / focal, depthImg.at<ushort>(i, j), 1);


				// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
				point = poseMat *point;
				// (3) Add the 3D point to vertices in point clouds data.
				vertices.push_back(point);
				// (4) Also compute the color of 3D point and add it to colors in point clouds data.
				colors.push_back(colorImg.at<Vec3b>(i, j));

			}
		}

	}
	// Save point clouds
	savePointCloudsPLY("pointClouds.ply", vertices, colors);

	return 0;
}




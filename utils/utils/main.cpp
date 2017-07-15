#include <sstream>
#include <string>
#include <unordered_map>

// including pcl headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
// including boost headers
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

// including opencv2 headers
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

// writing into a file
#include <ostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
using namespace std;
using namespace cv;
using namespace boost::filesystem;

/**
 * @brief Saving point clouds data as ply file
 * @param [in] filename : filename for saving point clouds (the extention should be .ply)
 * @param [in] vertices : list of vertex point
 * @param [in] colors   : list of vertex color
 * @return Success:true, Failure:false
 */
bool savePointCloudsPLY(string filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	std::ofstream fout;
	fout.open(filename.c_str());
	if (fout.fail()){
		cerr << "file open error:" << filename << endl;
		return false;
	}

	int pointNum = points->size();

	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "element vertex " << pointNum << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	if (normals != NULL)
	{
		fout << "property float normal_x" << endl;
		fout << "property float normal_y" << endl;
		fout << "property float normal_z" << endl;
	}

	fout << "property uchar red" << endl;
	fout << "property uchar green" << endl;
	fout << "property uchar blue" << endl;
	fout << "property uchar alpha" << endl;
	fout << "end_header" << endl;

	for (int i = 0; i < pointNum; i++){



		fout << points->at(i).x << " " << points->at(i).y << " " << points->at(i).z;
		if (normals != NULL)
			fout << " " << normals->at(i).normal_x << " " << normals->at(i).normal_y << " " << normals->at(i).normal_z;
		fout << " " << static_cast<int>(points->at(i).r) << " " << static_cast<int>(points->at(i).g) << " " << static_cast<int>(points->at(i).b) << " " << 255 << endl;
	}

	fout.close();

	return true;
}



int main(int argc, char* argv[])
{
	string projectSrcDir = PROJECT_SOURCE_DIR;
	string dataMainPath = "C:\\Users\\ahmad\\Downloads\\dataset\\train\\";
	string outputCloudsDir = "D:\\plarr\\trainplyfiles";
	for (auto modelsIT : directory_iterator(dataMainPath))
	{
		string modelPathIT = modelsIT.path().string();//path to a model folder, e.g. bird
		boost::replace_all(modelPathIT, "\\", "/");
		string modelName = modelPathIT.substr(modelPathIT.find_last_of("/") + 1);
		string modelRGBDir = modelPathIT + "/rgb/";
		string modelDepthDir = modelPathIT + "/depth/";


		std::string line;

		// loading camera intrinsic parameters
		std::ifstream ifStreamInfo(modelPathIT + "/info.yml");
		vector<vector<float>> cameraIntrinsicParamtersList;
		while (std::getline(ifStreamInfo, line))
		{
			std::istringstream iss(line);
			if (isdigit(line[0]))
				continue;
			unsigned first = line.find("[");
			unsigned last = line.find("]");
			string strNew = line.substr(first + 1, last - first - 1);
			std::vector<float> camIntrinsicParams;
			std::stringstream ss(strNew);
			string i;
			while (ss >> i)
			{
				boost::replace_all(i, ",", "");
				camIntrinsicParams.push_back(atof(i.c_str()));
			}
			cameraIntrinsicParamtersList.push_back(camIntrinsicParams);
		}
		// loading rotation and transformation matrices for all models
		vector<vector<float>> rotationValuesList;
		vector<vector<float>> translationValuesList;
		std::ifstream ifStreamGT(modelPathIT + "/gt.yml");
		bool processingRotationValues = true;
		while (std::getline(ifStreamGT, line))
		{
			std::istringstream iss(line);
			if (isdigit(line[0]) || boost::starts_with(line, "  obj_id:")){
				continue;
			}
			unsigned first = line.find("[");
			unsigned last = line.find("]");
			string strNew = line.substr(first + 1, last - first - 1);
			std::vector<float> rotationValues;
			std::vector<float> translationValues;
			boost::replace_all(strNew, ",", "");

			std::stringstream ss(strNew);
			string i;
			while (ss >> i)
			{
				if (processingRotationValues){
					rotationValues.push_back(atof(i.c_str()));
				}
				else{
					translationValues.push_back(atof(i.c_str()));
				}
			}
			if (processingRotationValues){
				rotationValuesList.push_back(rotationValues);
			}
			else{
				translationValuesList.push_back(translationValues);
			}
			processingRotationValues = !processingRotationValues;
		}

		int i = 0;
		int modelIndex = -1;
		for (auto it : directory_iterator(modelRGBDir))
		{
			modelIndex++;
			// Loading depth image and color image

			string path = it.path().string();
			boost::replace_all(path, "\\", "/");
			string colorFilename = path;

			//cout << path << endl;
			boost::replace_all(path, "rgb", "depth");
			string depthFilename = path;
			//cout << path << endl;

			cv::Mat depthImg = cv::imread(depthFilename, CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat colorImg = cv::imread(colorFilename, CV_LOAD_IMAGE_COLOR);
			cv::cvtColor(colorImg, colorImg, CV_BGR2RGB); //this will put colors right
			// Loading camera pose
			//string poseFilename = projectSrcDir + "/data/pose/pose" + to_string(index) + ".txt";
			Eigen::Matrix4f poseMat;   // 4x4 transformation matrix

			vector<float> rotationValues = rotationValuesList[i];
			vector<float> translationsValues = translationValuesList[i];
			vector<float> camIntrinsicParams = cameraIntrinsicParamtersList[i++];

			poseMat(0, 0) = rotationValues[0];
			poseMat(0, 1) = rotationValues[1];
			poseMat(0, 2) = rotationValues[2];
			poseMat(0, 3) = translationsValues[0];
			poseMat(1, 0) = rotationValues[3];
			poseMat(1, 1) = rotationValues[4];
			poseMat(1, 2) = rotationValues[5];
			poseMat(1, 3) = translationsValues[1];
			poseMat(2, 0) = rotationValues[6];
			poseMat(2, 1) = rotationValues[7];
			poseMat(2, 2) = rotationValues[8];
			poseMat(2, 3) = translationsValues[2];
			poseMat(3, 0) = 0;
			poseMat(3, 1) = 0;
			poseMat(3, 2) = 0;
			poseMat(3, 3) = 1;

			//cout << "Transformation matrix" << endl << poseMat << endl;

			// Setting camera intrinsic parameters of depth camera
			float focal = camIntrinsicParams[0];  // focal length
			float px = camIntrinsicParams[2]; // principal point x
			float py = camIntrinsicParams[5]; // principal point y


			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);



			pcl::CentroidPoint<pcl::PointXYZRGBA> centroid;


			//Create point clouds from depth image and color image using camera intrinsic parameters
			// (1) Compute 3D point from depth values and pixel locations on depth image using camera intrinsic parameters.
			for (int j = 0; j < depthImg.cols; j += 1)
			{
				for (int i = 0; i < depthImg.rows; i += 1)
				{
					auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focal, (i - py)*depthImg.at<ushort>(i, j) / focal, depthImg.at<ushort>(i, j), 1);

					// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
					point = poseMat *point;
					// (3) Add the 3D point to vertices in point clouds data.
					pcl::PointXYZRGBA p;
					p.x = point[0] / 1000.0f;
					p.y = point[1] / 1000.0f;
					p.z = point[2] / 1000.0f;
					p.r = colorImg.at<cv::Vec3b>(i, j)[0];
					p.g = colorImg.at<cv::Vec3b>(i, j)[1];
					p.b = colorImg.at<cv::Vec3b>(i, j)[2];
					p.a = 255;
					if (p.x == 0 && p.y == 0 && p.r == 0 && p.g == 0 && p.b == 0)
					{
						continue;
					}
					modelCloud->push_back(p);
					centroid.add(p);
				}
			}


			// Create and accumulate points


			pcl::PointXYZRGBA c2;
			centroid.get(c2);
			for (size_t x = 0; x < modelCloud->size(); x++)
			{
				modelCloud->points[x].x -= c2.x;
				modelCloud->points[x].y -= c2.y;
				modelCloud->points[x].z -= c2.z;
				modelCloud->points[x].r -= c2.r;
				modelCloud->points[x].g -= c2.g;
				modelCloud->points[x].b -= c2.b;
			}

		//	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroidCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
		//	centroidCloud->push_back(c2);
		//	savePointCloudsPLY(outputCloudsDir + "\\" + modelName + "-" + "CENTROID.ply", centroidCloud, NULL);

			pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
			pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
			ne.setKSearch(10);
			ne.setInputCloud(modelCloud);
			ne.compute(*model_normals);

			// Save point clouds
			savePointCloudsPLY(outputCloudsDir + "\\" + modelName + "-" + to_string(modelIndex) + ".ply", modelCloud, model_normals);

		}
	}

	return 0;
}




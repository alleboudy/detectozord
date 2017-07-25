#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/triangulation.h>
// including opencv2 headers
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/uniform_sampling.h>
using namespace std;
using namespace cv;
using namespace boost::filesystem;

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
int
main(int argc, char** argv)
{
	// Read in the cloud data


	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>), cloud_f(new pcl::PointCloud<pcl::PointXYZRGBA>);
	string msg = "Couldn't read file C:\\Users\\ahmad\\Downloads\\challenge2_val\\scenesClouds\\05-0.ply \n";
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>("C:\\Users\\ahmad\\Desktop\\testscenes\\challenge1_1-4.ply", *cloud) == -1){ PCL_ERROR(msg.c_str()); return (-1); }
	std::cout << "Loaded" << cloud->width * cloud->height << "points" << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	/*pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(0.0009f, 0.0009f, 0.0009f);
	vg.filter(*cloud_filtered);
	*/
	pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
	uniform_sampling.setInputCloud(cloud);
	uniform_sampling.setRadiusSearch(0.001);
	uniform_sampling.filter(*cloud_filtered);


	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; //*

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.02);

	int i = 0, nr_points = (int)cloud_filtered->points.size();
	while (cloud_filtered->points.size() > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Get the points associated with the planar surface
		extract.filter(*cloud_plane);
		std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

		// Remove the planar inliers, extract the rest
		extract.setNegative(true);
		extract.filter(*cloud_f);
		*cloud_filtered = *cloud_f;
	}

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance(0.02); // 2cm
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		pcl::visualization::PCLVisualizer viewer3("clustered instances");

		//densifying the clouds

		

		pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA> mls;
		mls.setInputCloud(cloud_cluster);
		mls.setSearchRadius(0.03);
		mls.setPolynomialFit(true);
		mls.setPolynomialOrder(2);
		mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::SAMPLE_LOCAL_PLANE);
		mls.setUpsamplingRadius(0.01);// has 2 be larger than setUpsamplingStepSize
		mls.setUpsamplingStepSize(0.008);//smaller increases the generated points
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGBA>());
		mls.process(*cloud_smoothed);
		cout << "upsampled cloud" << cloud_smoothed->size() << endl;
		/*pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> ne;
		ne.setNumberOfThreads(8);
		ne.setInputCloud(cloud_smoothed);
		ne.setRadiusSearch(0.01);
		Eigen::Vector4f centroid;
		compute3DCentroid(*cloud_smoothed, centroid);
		ne.setViewPoint(centroid[0], centroid[1], centroid[2]);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normals);
		for (size_t i = 0; i < cloud_normals->size(); ++i)
		{
			cloud_normals->points[i].normal_x *= -1;
			cloud_normals->points[i].normal_y *= -1;
			cloud_normals->points[i].normal_z *= -1;
		}
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<pcl::PointNormal>());
		concatenateFields(*cloud_smoothed, *cloud_normals, *cloud_smoothed_normals);
*/
		/*
	pcl::Poisson<pcl::PointNormal> poisson;
		poisson.setDepth(9);
		poisson.setInputCloud
			(cloud_smoothed_normals);
		pcl::PolygonMesh mesh;
		poisson.reconstruct(mesh);
		*/
		//float step = 0.1f;
		/*for (size_t z = 0; z < cloud_cluster->size(); z++)
		{

		dense_cloud_cluster->push_back(cloud_cluster->points[z]);
		pcl::PointXYZRGBA newPoint;
		newPoint.r = cloud_cluster->points[z].r;
		newPoint.g = cloud_cluster->points[z].g;
		newPoint.b = cloud_cluster->points[z].b;
		newPoint.a = 255;


		newPoint.x = cloud_cluster->points[z].x + cloud_cluster->points[z].x*step;
		newPoint.y = cloud_cluster->points[z].y;
		newPoint.z = cloud_cluster->points[z].z;
		dense_cloud_cluster->push_back(newPoint);
		newPoint.y = cloud_cluster->points[z].y + cloud_cluster->points[z].y*step;
		newPoint.x = cloud_cluster->points[z].x;
		newPoint.z = cloud_cluster->points[z].z;
		dense_cloud_cluster->push_back(newPoint);
		newPoint.z = cloud_cluster->points[z].z + cloud_cluster->points[z].z*step;
		newPoint.x = cloud_cluster->points[z].x;
		newPoint.y = cloud_cluster->points[z].y;
		dense_cloud_cluster->push_back(newPoint);


		}*/

		viewer3.addPointCloud(cloud_smoothed, "instance");
		viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "instance");


		while (!viewer3.wasStopped() /*&& !viewer4.wasStopped()*/)
		{
			viewer3.spinOnce(100);
			//		viewer4.spinOnce(100);

		}
		savePointCloudsPLY("cloud_cluster_" + to_string(j) + ".ply", cloud_smoothed, NULL);
		j++;
	}

	return (0);
}

//#include <sstream>
//#include <string>
//#include <unordered_map>
//
//// including pcl headers
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_cloud.h>
//#include <pcl/correspondence.h>
//#include <pcl/features/normal_3d_omp.h>
//#include <pcl/features/shot_omp.h>
//#include <pcl/features/board.h>
//#include <pcl/filters/uniform_sampling.h>
//#include <pcl/recognition/cg/hough_3d.h>
//#include <pcl/recognition/cg/geometric_consistency.h>
//#include <pcl/recognition/hv/hv_go.h>
//#include <pcl/registration/icp.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/kdtree/impl/kdtree_flann.hpp>
//#include <pcl/common/transforms.h>
//#include <pcl/console/parse.h>
//#include <pcl/keypoints/iss_3d.h>
//#include <pcl/recognition/hv/greedy_verification.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/ModelCoefficients.h>
//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/sample_consensus/model_types.h>
//#include <pcl/segmentation/sac_segmentation.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl/common/centroid.h>
//
//#include <pcl/surface/grid_projection.h>
//// including boost headers
//#include <boost/algorithm/string/replace.hpp>
//#include <boost/filesystem.hpp>
//#include <boost/algorithm/string/predicate.hpp>
//#include <boost/lexical_cast.hpp>
//
//// including opencv2 headers
//#include <opencv2/imgproc.hpp>
//#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//// writing into a file
//#include <ostream>
//#include <vector>
//#include <algorithm>
//#include <iterator>
//#include <iostream>
//#include <boost/iostreams/device/file.hpp>
//#include <boost/iostreams/stream.hpp>
//using namespace std;
//using namespace cv;
//using namespace boost::filesystem;
//
//

//
//int
//main(int argc, char** argv)
//{
//	// Read in the cloud data
//	string projectSrcDir = PROJECT_SOURCE_DIR;
//	string dataMainPath = "C:\\Users\\ahmad\\Downloads\\challenge2_val\\last";
//	string outputCloudsDir = "C:\\Users\\ahmad\\Downloads\\challenge2_val\\scenesClouds";
//	for (auto modelsIT : directory_iterator(dataMainPath))
//	{
//		string modelPathIT = modelsIT.path().string();//path to a model folder, e.g. bird
//		boost::replace_all(modelPathIT, "\\", "/");
//		string modelName = modelPathIT.substr(modelPathIT.find_last_of("/") + 1);
//		string modelRGBDir = modelPathIT + "/rgb/";
//		string modelDepthDir = modelPathIT + "/depth/";
//
//
//		std::string line;
//
//		// loading camera intrinsic parameters
//		std::ifstream ifStreamInfo(modelPathIT + "/info.yml");
//		vector<vector<float>> cameraIntrinsicParamtersList;
//		while (std::getline(ifStreamInfo, line))
//		{
//			std::istringstream iss(line);
//			if (isdigit(line[0]))
//				continue;
//			unsigned first = line.find("[");
//			unsigned last = line.find("]");
//			string strNew = line.substr(first + 1, last - first - 1);
//			std::vector<float> camIntrinsicParams;
//			std::stringstream ss(strNew);
//			string i;
//			while (ss >> i)
//			{
//				boost::replace_all(i, ",", "");
//				camIntrinsicParams.push_back(atof(i.c_str()));
//			}
//			cameraIntrinsicParamtersList.push_back(camIntrinsicParams);
//		}
//		// loading rotation and transformation matrices for all models
//		vector<vector<float>> rotationValuesList;
//		vector<vector<float>> translationValuesList;
//		std::ifstream ifStreamGT(modelPathIT + "/gt.yml");
//		bool processingRotationValues = true;
//		while (std::getline(ifStreamGT, line))
//		{
//			std::istringstream iss(line);
//			if (isdigit(line[0]) || boost::starts_with(line, "  obj_id:")){
//				continue;
//			}
//			unsigned first = line.find("[");
//			unsigned last = line.find("]");
//			string strNew = line.substr(first + 1, last - first - 1);
//			std::vector<float> rotationValues;
//			std::vector<float> translationValues;
//			boost::replace_all(strNew, ",", "");
//
//			std::stringstream ss(strNew);
//			string i;
//			while (ss >> i)
//			{
//				if (processingRotationValues){
//					rotationValues.push_back(atof(i.c_str()));
//				}
//				else{
//					translationValues.push_back(atof(i.c_str()));
//				}
//			}
//			if (processingRotationValues){
//				rotationValuesList.push_back(rotationValues);
//			}
//			else{
//				translationValuesList.push_back(translationValues);
//			}
//			processingRotationValues = !processingRotationValues;
//		}
//
//		int i = 0;
//		int modelIndex = -1;
//		for (auto it : directory_iterator(modelRGBDir))
//		{
//			modelIndex++;
//			// Loading depth image and color image
//
//			string path = it.path().string();
//			boost::replace_all(path, "\\", "/");
//			string colorFilename = path;
//
//			//cout << path << endl;
//			boost::replace_all(path, "rgb", "depth");
//			string depthFilename = path;
//			//cout << path << endl;
//
//			cv::Mat depthImg = cv::imread(depthFilename, CV_LOAD_IMAGE_UNCHANGED);
//			cv::Mat colorImg = cv::imread(colorFilename, CV_LOAD_IMAGE_COLOR);
//			cv::cvtColor(colorImg, colorImg, CV_BGR2RGB); //this will put colors right
//			// Loading camera pose
//			//string poseFilename = projectSrcDir + "/data/pose/pose" + to_string(index) + ".txt";
//			Eigen::Matrix4f poseMat;   // 4x4 transformation matrix
//
//			vector<float> rotationValues = rotationValuesList[i];
//			vector<float> translationsValues = translationValuesList[i];
//			vector<float> camIntrinsicParams = cameraIntrinsicParamtersList[i++];
//
//			poseMat(0, 0) = rotationValues[0];
//			poseMat(0, 1) = rotationValues[1];
//			poseMat(0, 2) = rotationValues[2];
//			poseMat(0, 3) = translationsValues[0];
//			poseMat(1, 0) = rotationValues[3];
//			poseMat(1, 1) = rotationValues[4];
//			poseMat(1, 2) = rotationValues[5];
//			poseMat(1, 3) = translationsValues[1];
//			poseMat(2, 0) = rotationValues[6];
//			poseMat(2, 1) = rotationValues[7];
//			poseMat(2, 2) = rotationValues[8];
//			poseMat(2, 3) = translationsValues[2];
//			poseMat(3, 0) = 0;
//			poseMat(3, 1) = 0;
//			poseMat(3, 2) = 0;
//			poseMat(3, 3) = 1;
//
//			//cout << "Transformation matrix" << endl << poseMat << endl;
//
//			// Setting camera intrinsic parameters of depth camera
//			float focal = camIntrinsicParams[0];  // focal length
//
//			float px = camIntrinsicParams[2]; // principal point x
//			float py = camIntrinsicParams[5]; // principal point y
//
//
//			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
//
//
//
//			//	pcl::CentroidPoint<pcl::PointXYZRGBA> centroid;
//
//
//			//Create point clouds from depth image and color image using camera intrinsic parameters
//			// (1) Compute 3D point from depth values and pixel locations on depth image using camera intrinsic parameters.
//			for (int j = 0; j < depthImg.cols; j += 1)
//			{
//				for (int i = 0; i < depthImg.rows; i += 1)
//				{
//					auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focal, (i - py)*depthImg.at<ushort>(i, j) / focal, depthImg.at<ushort>(i, j), 1);
//
//					// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
//					point = poseMat *point;
//					// (3) Add the 3D point to vertices in point clouds data.
//					pcl::PointXYZRGBA p;
//					p.x = point[0] / 1000.0f;
//					p.y = point[1] / 1000.0f;
//					p.z = point[2] / 1000.0f;
//					p.r = colorImg.at<cv::Vec3b>(i, j)[0];
//					p.g = colorImg.at<cv::Vec3b>(i, j)[1];
//					p.b = colorImg.at<cv::Vec3b>(i, j)[2];
//					p.a = 255;
//					if (p.x == 0 && p.y == 0 && p.r == 0 && p.g == 0 && p.b == 0)
//					{
//						continue;
//					}
//					modelCloud->push_back(p);
//					//	centroid.add(p);
//				}
//			}
//
//			/*	pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
//			pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
//			ne.setKSearch(2);
//			ne.setInputCloud(modelCloud);
//			ne.compute(*model_normals);
//			*/
//
//			// Create and accumulate points
//
//			/*
//			float largestx = -std::numeric_limits<float>::max(), largesty = -std::numeric_limits<float>::max(), largestz = -std::numeric_limits<float>::max();
//			float smallestx = std::numeric_limits<float>::max(), smallesty = std::numeric_limits<float>::max(), smallestz = std::numeric_limits<float>::max();
//			for (size_t x = 0; x < modelCloud->size(); x++)
//			{
//
//
//				largestx = max(largestx, modelCloud->points[x].x);
//				largesty = max(largesty, modelCloud->points[x].y);
//				largestz = max(largestz, modelCloud->points[x].z);
//
//				smallestx = min(smallestx, modelCloud->points[x].x);
//				smallesty = min(smallesty, modelCloud->points[x].y);
//				smallestz = min(smallestz, modelCloud->points[x].z);
//				//modelCloud->points[x].r -= c2.r;
//				//modelCloud->points[x].g -= c2.g;
//				//modelCloud->points[x].b -= c2.b;
//			}*/
//
//		//	double scale = min((1 / (largestx - smallestx)), min(1 / (largesty - smallesty),1 / (largestz - smallestz)));
//
//
//
//			//pcl::CentroidPoint<pcl::PointXYZRGBA> centroid;
//			//pcl::PointXYZRGBA c2;
//			//for (size_t x = 0; x < modelCloud->size(); x++)
//			//{
//			//	modelCloud->points[x].x = (modelCloud->points[x].x - 0.5*(smallestx + largestx))*scale + 0.5;
//			//	modelCloud->points[x].y = (modelCloud->points[x].y - 0.5*(smallesty + largesty))*scale + 0.5;
//			//	modelCloud->points[x].z = (modelCloud->points[x].z - 0.5*(smallestz + largestz))*scale + 0.5;
//			//	centroid.add(modelCloud->points[x]);
//
//			//}
//
//		/*	centroid.get(c2);
//			for (size_t x = 0; x < modelCloud->size(); x++)
//			{
//
//				modelCloud->points[x].x -= c2.x;
//				modelCloud->points[x].y -= c2.y;
//				modelCloud->points[x].z -= c2.z;
//
//
//			}*/
//
//
//
//
//			/*	pcl::PointXYZRGBA c2;
//			centroid.get(c2);
//			for (size_t x = 0; x < modelCloud->size(); x++)
//			{
//			modelCloud->points[x].x -= c2.x;
//			modelCloud->points[x].y -= c2.y;
//			modelCloud->points[x].z -= c2.z;
//			modelCloud->points[x].r -= c2.r;
//			modelCloud->points[x].g -= c2.g;
//			modelCloud->points[x].b -= c2.b;
//			}*/
//
//			//	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroidCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
//			//	centroidCloud->push_back(c2);
//			//	savePointCloudsPLY(outputCloudsDir + "\\" + modelName + "-" + "CENTROID.ply", centroidCloud, NULL);
//
//
//			// Save point clouds
//			savePointCloudsPLY(outputCloudsDir + "\\" + modelName + "-" + to_string(modelIndex) + ".ply", modelCloud, NULL);
//
//		}
//	}
//
//
//	*/
//
//	return (0);
//}





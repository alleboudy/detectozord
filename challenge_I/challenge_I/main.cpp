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
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>

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

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

using namespace std;
using namespace pcl;
using namespace boost::filesystem;


double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

int main(int argc, char** argv)
{
	std::string projectSrcDir = PROJECT_SOURCE_DIR;

	//0-  loading all (10) of the scene clouds in the testing folder and keeping them for testing.

	//	std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> Scenes;

	//pipeline parameters

	/*
	BIRD PARAMETERS
	float SegMentationDistanceThreshold = 0.01;
	float sceneUniformSamplingRadius = 0.002f;
	float scenedescriberRadiusSearch = 0.02f;
	float modelSamplingRadiusSearch = 0.008f;
	float gcClusteringSize = 0.005f;
	float gcClusteringThreshold = 20;
	int icpsetMaximumIterations = 50;
	float icpsetMaxCorrespondenceDistance = 0.05;
	float GoHvsetInlierThreshold = 0.05f;
	float GoHvsetOcclusionThreshold = 0.01;
	int GoHvRegularizer = 3;
	float GoHvsetRadiusClutter = 0.03;
	int GoHvClutterRegularizer = 5;
	float GoHvRadiusNormals = 0.05f;
	int sceneLoadLoopStep = 1;
	int normalEstimationK=10;
	float kdnearestDistance = 0.25f;
	*/



	// CRACKER PARAMETERS - only first scene is matched
	//float SegMentationDistanceThreshold = 0.015;//eating up the floor (more will eat up lots of points, less is undertuned)
	//float sceneUniformSamplingRadius = 0.002f;//for the keypoints of the scene(more will allow more points, less will remove less points)
	//float scenedescriberRadiusSearch = 0.02f;//for the describer of the scene and the model
	//float modelSamplingRadiusSearch = 0.008f;//keypoints of the model, larger allows more points, smaller, less points
	//float gcClusteringSize = 0.007f;//clustering size //increasing it allows more clusters
	//float gcClusteringThreshold = 20;//how many points in a cluster at least
	//int icpsetMaximumIterations = 50;//for the alignment with icp
	//float icpsetMaxCorrespondenceDistance = 0.05;
	//int sceneLoadLoopStep = 2;

	//Hypothesis verrification parameters:
	//float GoHvsetInlierThreshold = 0.05f;//HV, increasing allows more correspondences to pass
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;

	//downsampling the scene cloud
	//int sceneLoadLoopStep = 2;
	//int normalEstimationK=10;
	//float kdnearestDistance = 0.25f;


	//SHOE PARAMS, BROKEN!!!
	//float SegMentationDistanceThreshold = 0.03;//eating up the floor (more will eat up lots of points, less is undertuned)
	//float sceneUniformSamplingRadius = 0.002f;//for the keypoints of the scene(more will allow more points, less will remove less points)
	//float scenedescriberRadiusSearch = 0.2f;//for the describer of the scene and the model
	//float modelSamplingRadiusSearch = 0.002f;//keypoints of the model, larger allows less points, smaller, more points
	//float gcClusteringSize = 0.005f;//clustering size //increasing it allows more clusters
	//float gcClusteringThreshold = 25;//how many points in a cluster at least
	//int icpsetMaximumIterations = 50;//for the alignment with icp
	//float icpsetMaxCorrespondenceDistance = 0.05;

	//Hypothesis verrification parameters:
	//float GoHvsetInlierThreshold = 0.05f;//HV, increasing allows more clusters/instances to pass
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;

	//downsampling the scene cloud
	//int sceneLoadLoopStep = 1;
	//int normalEstimationK = 10;
	//float kdnearestDistance = 0.25f;


	//HOUSE PARAMETERS
	//float SegMentationDistanceThreshold = 0.01;
	//float wallsegThreshold = 0.01;
	//float sceneUniformSamplingRadius = 0.002f;
	//float scenedescriberRadiusSearch = 0.035f;
	//float modeldescriberRadiusSearch = 0.05f;
	//float modelSamplingRadiusSearch = 0.01f;
	//float gcClusteringSize = 0.009f;
	//float gcClusteringThreshold = 20;
	//int icpsetMaximumIterations = 50;
	//float icpsetMaxCorrespondenceDistance = 0.05;

	//Hypothesis verrification parameters:
	//float GoHvsetInlierThreshold = 0.05f;
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;

	//downsampling the scene cloud
	//int sceneLoadLoopStep = 2;
	//int normalEstimationK = 100;
	//float kdnearestDistance = 0.8f;
	//float SegMentationDistanceThreshold = 0.02;


	//SHOE PARAMETERS
	//float wallsegThreshold = 0.01;
	//float sceneUniformSamplingRadius = 0.002f;
	//float scenedescriberRadiusSearch = 0.035f;
	//float modeldescriberRadiusSearch = 0.05f;
	//float modelSamplingRadiusSearch = 0.01f;
	//float gcClusteringSize = 0.007f;
	//float gcClusteringThreshold = 20;
	//int icpsetMaximumIterations = 50;
	//float icpsetMaxCorrespondenceDistance = 0.05;

	//Hypothesis verrification parameters:
	//float GoHvsetInlierThreshold = 0.05f;
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;

	//downsampling the scene cloud
	//int sceneLoadLoopStep = 2;
	//int normalEstimationK = 100;
	//float kdnearestDistance = 0.8f;

	bool split = false;//for taking splits of the model against the axis 

	//CAN PARAMETERS
	//float SegMentationDistanceThreshold = 0.01;
	//float wallsegThreshold = 0.02;
	//float sceneUniformSamplingRadius = 0.0008f;
	//float scenedescriberRadiusSearch = 0.035f;
	//float modeldescriberRadiusSearch = 0.05f;
	//float modelSamplingRadiusSearch = 0.001f;
	//float gcClusteringSize = 0.007f;
	//float gcClusteringThreshold = 10;
	//int icpsetMaximumIterations = 50;
	//float icpsetMaxCorrespondenceDistance = 0.05;
	//float GoHvsetInlierThreshold = 0.05f;
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;
	//int sceneLoadLoopStep = 1;
	//int normalEstimationK = 100;
	//float kdnearestDistance = 1.0f;

	float SegMentationDistanceThreshold;
	float wallsegThreshold;
	float sceneUniformSamplingRadius;
	float scenedescriberRadiusSearch;
	float modeldescriberRadiusSearch;
	float modelSamplingRadiusSearch;
	float gcClusteringSize;
	float gcClusteringThreshold;
	int icpsetMaximumIterations;
	float icpsetMaxCorrespondenceDistance;
	float GoHvsetInlierThreshold;
	float GoHvsetOcclusionThreshold;
	int GoHvRegularizer;
	float GoHvsetRadiusClutter;
	int GoHvClutterRegularizer;
	float GoHvRadiusNormals;
	int sceneLoadLoopStep;
	int normalEstimationK;
	float kdnearestDistance;

	//ISS PARAMS
	bool useISS = false;
	double scene_resolution;
	double model_resolution;

	double iss_salient_radius_;
	double iss_non_max_radius_;
	double iss_gamma_21_(0.975);
	double iss_gamma_32_(0.975);
	double iss_min_neighbors_(5);
	double iss_normal_radius_;
	double iss_border_radius_;
	int iss_threads_(4);

	//float SegMentationDistanceThreshold = 0.01;
	//float sceneUniformSamplingRadius = 0.005f;
	//float scenedescriberRadiusSearch = 0.07f;
	//float modelSamplingRadiusSearch = 0.005f;
	//float gcClusteringSize = 0.05f;
	//float gcClusteringThreshold = 20;
	//int icpsetMaximumIterations = 50;
	//float icpsetMaxCorrespondenceDistance = 0.05;
	//float GoHvsetInlierThreshold = 0.05f;
	//float GoHvsetOcclusionThreshold = 0.01;
	//int GoHvRegularizer = 3;
	//float GoHvsetRadiusClutter = 0.03;
	//int GoHvClutterRegularizer = 5;
	//float GoHvRadiusNormals = 0.05f;
	//int sceneLoadLoopStep = 1;
	//int normalEstimationK = 100;
	//float kdnearestDistance = 0.1f;


	//1 - Load each RGB image, its corresponding depth image and their provided camera parameters and rotation and translation


	string modelMainPath = "";
	string objRGBSrcDir = "";
	string objDepthSrcDir = "";
	string modelName = "";

	int index = -1;
	for (auto it : directory_iterator(projectSrcDir + "/data/challenge_train/models/"))
	{
		string challengesMainPath = projectSrcDir + "/data/challenge1_val/test/";
		string challengePath = "";
		string challengeName = "";
		string sceneRGBDir = "";
		string sceneDepthDir = "";
		clock_t tStart = clock();

		index++;
		string path = it.path().string();
		boost::replace_all(path, "\\", "/");

		modelName = path.substr(path.find_last_of("/") + 1);
		boost::replace_all(modelName, ".ply", "");


		modelMainPath = path;
		cout << "Model name:" << modelName << endl;
		if (modelName == "shoe")
		{
			SegMentationDistanceThreshold = 0.01;
			wallsegThreshold = 0.01;
			sceneUniformSamplingRadius = 0.002f;
			scenedescriberRadiusSearch = 0.035f;
			modeldescriberRadiusSearch = 0.05f;
			modelSamplingRadiusSearch = 0.01f;
			gcClusteringSize = 0.007f;
			gcClusteringThreshold = 20;
			icpsetMaximumIterations = 50;
			icpsetMaxCorrespondenceDistance = 0.05;
			GoHvsetInlierThreshold = 0.05f;
			GoHvsetOcclusionThreshold = 0.01;
			GoHvRegularizer = 3;
			GoHvsetRadiusClutter = 0.03;
			GoHvClutterRegularizer = 5;
			GoHvRadiusNormals = 0.05f;
			sceneLoadLoopStep = 2;
			normalEstimationK = 100;
			kdnearestDistance = 0.8f;
		}
		else if (modelName == "bird" || modelName == "teapot" || modelName == "bond")
		{
			SegMentationDistanceThreshold = 0.01;
			wallsegThreshold = 0.01;
			sceneUniformSamplingRadius = 0.002f;
			scenedescriberRadiusSearch = 0.02f;
			modeldescriberRadiusSearch = 0.02f;
			modelSamplingRadiusSearch = 0.008f;
			gcClusteringSize = 0.005f;
			gcClusteringThreshold = 20;
			icpsetMaximumIterations = 50;
			icpsetMaxCorrespondenceDistance = 0.05;
			GoHvsetInlierThreshold = 0.05f;
			GoHvsetOcclusionThreshold = 0.01;
			GoHvRegularizer = 3;
			GoHvsetRadiusClutter = 0.03;
			GoHvClutterRegularizer = 5;
			GoHvRadiusNormals = 0.05f;
			sceneLoadLoopStep = 1;
			normalEstimationK = 10;
			kdnearestDistance = 0.25f;
		}

		else if (modelName == "house")
		{
			SegMentationDistanceThreshold = 0.01;
			wallsegThreshold = 0.01;
			sceneUniformSamplingRadius = 0.002f;
			scenedescriberRadiusSearch = 0.035f;
			modeldescriberRadiusSearch = 0.05f;
			modelSamplingRadiusSearch = 0.01f;
			gcClusteringSize = 0.009f;
			gcClusteringThreshold = 20;
			icpsetMaximumIterations = 50;
			icpsetMaxCorrespondenceDistance = 0.05;
			GoHvsetInlierThreshold = 0.05f;
			GoHvsetOcclusionThreshold = 0.01;
			GoHvRegularizer = 3;
			GoHvsetRadiusClutter = 0.03;
			GoHvClutterRegularizer = 5;
			GoHvRadiusNormals = 0.05f;
			sceneLoadLoopStep = 2;
			normalEstimationK = 100;
			kdnearestDistance = 0.8f;
		}

		else if (modelName == "cracker")
		{
			SegMentationDistanceThreshold = 0.015;
			wallsegThreshold = 0.01;
			sceneUniformSamplingRadius = 0.002f;
			scenedescriberRadiusSearch = 0.02f;
			modeldescriberRadiusSearch = 0.05f;
			modelSamplingRadiusSearch = 0.008f;
			gcClusteringSize = 0.007f;
			gcClusteringThreshold = 20;
			icpsetMaximumIterations = 50;
			icpsetMaxCorrespondenceDistance = 0.05;
			GoHvsetInlierThreshold = 0.05f;
			GoHvsetOcclusionThreshold = 0.01;
			GoHvRegularizer = 3;
			GoHvsetRadiusClutter = 0.03;
			GoHvClutterRegularizer = 5;
			GoHvRadiusNormals = 0.05f;
			sceneLoadLoopStep = 2;
			normalEstimationK = 100;
			kdnearestDistance = 0.8f;
		}
		else
		{
			cout << "it is the can, sorry we didn't fine good parameters for it yet" << endl;
			SegMentationDistanceThreshold = 0.01;
			wallsegThreshold = 0.02;
			sceneUniformSamplingRadius = 0.0008f;
			scenedescriberRadiusSearch = 0.035f;
			modeldescriberRadiusSearch = 0.05f;
			modelSamplingRadiusSearch = 0.001f;
			gcClusteringSize = 0.007f;
			gcClusteringThreshold = 10;
			icpsetMaximumIterations = 50;
			icpsetMaxCorrespondenceDistance = 0.05;
			GoHvsetInlierThreshold = 0.05f;
			GoHvsetOcclusionThreshold = 0.01;
			GoHvRegularizer = 3;
			GoHvsetRadiusClutter = 0.03;
			GoHvClutterRegularizer = 5;
			GoHvRadiusNormals = 0.05f;
			sceneLoadLoopStep = 1;
			normalEstimationK = 100;
			kdnearestDistance = 1.0f;
		}

		for (auto challengesIT : directory_iterator(challengesMainPath))
		{
			try{
				string pathIT = challengesIT.path().string();
				boost::replace_all(pathIT, "\\", "/");

				challengeName = pathIT.substr(pathIT.find_last_of("/") + 1);
				cerr << challengeName << endl;
				challengePath = pathIT;
				sceneRGBDir = challengePath + "/rgb/";
				sceneDepthDir = challengePath + "/depth/";
				// create directory for team name if not yet exist
				string teamname = "/Team7";
				const char* directorypathTeamName = projectSrcDir.c_str();
				boost::filesystem::path dirTeamName(directorypathTeamName + teamname);
				if (boost::filesystem::create_directory(dirTeamName))
				{
					std::cerr << "Teamname directory created: " << dirTeamName << std::endl;
				}
				// create a directory for each challenge (e.g. challenge1_1) if not yet exist
				const char* directorypathChallenges = projectSrcDir.c_str();
				boost::filesystem::path dirChallenge(directorypathChallenges + teamname + "/" + challengeName);
				if (boost::filesystem::create_directory(dirChallenge))
				{
					std::cerr << "Challenge directory created: " << dirChallenge << std::endl;
				}

				for (auto it : directory_iterator(sceneRGBDir))
				{

					string path = it.path().string();
					boost::replace_all(path, "\\", "/");
					string colorSceneFilepath = path;
					string colorSceneFilename = colorSceneFilepath.substr(colorSceneFilepath.find_last_of("/") + 1);
					boost::replace_all(colorSceneFilename, ".png", "");


					//cout << path << endl;
					boost::replace_all(path, "rgb", "depth");
					string depthSceneFilepath = path;
					//cout << path << endl;

					cv::Mat depthImg = cv::imread(depthSceneFilepath, CV_LOAD_IMAGE_UNCHANGED);
					cv::Mat colorImg = cv::imread(colorSceneFilepath, CV_LOAD_IMAGE_COLOR);
					cv::cvtColor(colorImg, colorImg, CV_BGR2RGB); //this will put colors right
					//[570.342, 0, 320, 0, 570.342, 240, 0, 0, 1]

					// Setting camera intrinsic parameters of depth camera
					float focalx = 579.470894;  // focal length
					float focaly = 580.338802;
					float px = 322.229447; // principal point x
					float py = 244.567608; // principal point y

					pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
					pcl::PointCloud<pcl::PointXYZRGBA>::Ptr originalSceneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);



					// Create point clouds from depth image and color image using camera intrinsic parameters
					// (1) Compute 3D point from depth values and pixel locations on depth image using camera intrinsic parameters.
					for (int j = 0; j < depthImg.cols; j += sceneLoadLoopStep)
					{
						for (int i = 0; i < depthImg.rows; i += sceneLoadLoopStep)
						{
							auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focalx, (i - py)*depthImg.at<ushort>(i, j) / focaly, depthImg.at<ushort>(i, j), 1);

							// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
							//	point = poseMat *point;
							// (3) Add the 3D point to vertices in point clouds data.
							pcl::PointXYZRGBA p;
							p.x = point[0] / 1000.0f;
							p.y = point[1] / 1000.0f;
							p.z = point[2] / 1000.0f;
							p.r = colorImg.at<cv::Vec3b>(i, j)[0];
							p.g = colorImg.at<cv::Vec3b>(i, j)[1];
							p.b = colorImg.at<cv::Vec3b>(i, j)[2];
							p.a = 255;
							if (p.r == 0 && p.g == 0 && p.b == 0)
							{
								continue;
							}
							sceneCloud->push_back(p);
						}
					}


					/*	std::string scene_filename_ ="C:\\Users\\ahmad\\Documents\\PLARR2017\\plarr17\\exercise7\\data\\scene_clutter.pcd";

					pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
					if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(scene_filename_, *sceneCloud) == -1){ PCL_ERROR("Couldn't read file scene.pcd \n"); return (-1); }
					std::cout << "Loaded" << sceneCloud->width * sceneCloud->height << "points" << std::endl;


					for (size_t j = 0; j < sceneCloud->size(); j++)
					{
					sceneCloud->at(j).a = 255;

					}*/

					copyPointCloud(*sceneCloud, *originalSceneCloud);
					pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
					pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
					// Create the segmentation object
					pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
					// Optional
					seg.setOptimizeCoefficients(true);
					// Mandatory: remove the ground plane
					seg.setModelType(pcl::SACMODEL_PLANE);
					seg.setMethodType(pcl::SAC_RANSAC);
					seg.setDistanceThreshold(SegMentationDistanceThreshold);

					seg.setInputCloud(sceneCloud);
					seg.segment(*inliers, *coefficients);

					pcl::ExtractIndices<PointType> eifilter(true); // Initializing with true will allow us to extract the removed indices
					eifilter.setInputCloud(sceneCloud);
					eifilter.setIndices(inliers);
					//eifilter.setNegative(true);
					pcl::PointCloud<pcl::PointXYZRGBA>::Ptr inlierCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());

					eifilter.filter(*inlierCloud);

					eifilter.setInputCloud(sceneCloud);
					eifilter.setIndices(inliers);
					eifilter.setNegative(true);
					eifilter.filterDirectly(sceneCloud);



					double z_min = -1.f, z_max = 0;
					pcl::PointCloud<pcl::PointXYZRGBA>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZRGBA>());
					pcl::ConvexHull<pcl::PointXYZRGBA> hull;
					// hull.setDimension (2); // not necessarily needed, but we need to check the dimensionality of the output
					hull.setInputCloud(inlierCloud);
					hull.reconstruct(*hull_points);
					if (hull.getDimension() == 2)
					{
						pcl::ExtractPolygonalPrismData<pcl::PointXYZRGBA> prism;
						prism.setInputCloud(sceneCloud);
						prism.setInputPlanarHull(hull_points);
						prism.setHeightLimits(z_min, z_max);

						prism.segment(*inliers);
					}
					else
						PCL_ERROR("The input cloud does not represent a planar surface.\n");




					//pcl::ExtractIndices<PointType> eifilter(true); // Initializing with true will allow us to extract the removed indices
					eifilter.setInputCloud(sceneCloud);
					eifilter.setIndices(inliers);
					eifilter.setNegative(true);
					eifilter.filterDirectly(sceneCloud);




					seg.setModelType(pcl::SACMODEL_PLANE);
					seg.setDistanceThreshold(wallsegThreshold);
					seg.setInputCloud(sceneCloud);
					seg.segment(*inliers, *coefficients);

					//pcl::ExtractIndices<PointType> eifilter(true); // Initializing with true will allow us to extract the removed indices
					eifilter.setInputCloud(sceneCloud);
					eifilter.setIndices(inliers);
					eifilter.setNegative(true);
					eifilter.filterDirectly(sceneCloud);
					//	copyPointCloud(*sceneCloud, inliers->indices, *sceneCloud);



					//seg.setModelType(pcl::SACMODEL_PLANE);
					//seg.setRadiusLimits(0.01,0.1);
					//seg.setDistanceThreshold(0.1);
					//seg.setInputCloud(sceneCloud);
					//seg.segment(*inliers, *coefficients);

					////pcl::ExtractIndices<PointType> eifilter(true); // Initializing with true will allow us to extract the removed indices
					//eifilter.setInputCloud(sceneCloud);
					//eifilter.setIndices(inliers);
					//eifilter.setNegative(true);
					//eifilter.filterDirectly(sceneCloud);




					pcl::visualization::PCLVisualizer viewer3("3d scene after segmentation");
					viewer3.addPointCloud(sceneCloud, "scene");
					viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");
					//	viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "scene");


					pcl::visualization::PCLVisualizer viewer4("3d scene original cloud");
					viewer4.addPointCloud(originalSceneCloud, "scene");
					viewer4.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");
					//	viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "scene");



					while (!viewer3.wasStopped() && !viewer4.wasStopped())
					{
						viewer3.spinOnce(100);
						viewer4.spinOnce(100);

					}

					//if (viewer3.wasStopped() || viewer4.wasStopped())
			{
				viewer3.close();
				viewer4.close();


			}

			cout << challengeName << " cloud size " << sceneCloud->size() << endl;
			pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
			pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
			ne.setKSearch(normalEstimationK);



			cout << "Computing scene " << challengeName << " normals" << endl;
			/*pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
			ne.setSearchMethod(tree);
			ne.setRadiusSearch(100.0f);*/


			ne.setInputCloud(sceneCloud);
			ne.compute(*scene_normals);

			//	Scenes.insert(std::make_pair(challengeName, sceneCloud));
			cout << "Computing scene " << challengeName << " keypoints" << endl;

			pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
			uniform_sampling.setRadiusSearch(sceneUniformSamplingRadius); //the 3D grid leaf size

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneSampledCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());//used in case of iss
			if (!useISS)
			{
				uniform_sampling.setInputCloud(sceneCloud);
				uniform_sampling.filter(*sceneSampledCloud);
				cout << "sampled sceneSampledCloud :" + to_string(sceneSampledCloud->size()) << endl;

			}
			else
			{

				//
				// Compute keypoints
				//
				pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> iss_detector;

				iss_detector.setSearchMethod(tree);


				scene_resolution = computeCloudResolution(sceneCloud);
				cout << "Scene resolution" << scene_resolution << endl;
				iss_salient_radius_ = 6 * scene_resolution;
				iss_non_max_radius_ = 4 * scene_resolution;
				iss_normal_radius_ = 4 * scene_resolution;
				iss_border_radius_ = 1 * scene_resolution;
				iss_detector.setSalientRadius(iss_salient_radius_);
				iss_detector.setNonMaxRadius(iss_non_max_radius_);
				iss_detector.setThreshold21(iss_gamma_21_);
				iss_detector.setThreshold32(iss_gamma_32_);
				iss_detector.setNormalRadius(iss_normal_radius_);
				iss_detector.setBorderRadius(iss_border_radius_);
				iss_detector.setMinNeighbors(iss_min_neighbors_);
				iss_detector.setNumberOfThreads(iss_threads_);
				iss_detector.setInputCloud(sceneCloud);
				iss_detector.compute(*sceneSampledCloud);



				cout << "ISS Keypoints detected for scene :" + to_string(sceneSampledCloud->size()) << endl;

			}

			cout << "Creating scene " << challengeName << " Descriptors" << endl;

			pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> describer;
			describer.setRadiusSearch(scenedescriberRadiusSearch);
			pcl::PointCloud<pcl::SHOT352>::Ptr sceneDescriptors(new pcl::PointCloud<pcl::SHOT352>);
			//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneSampledCloudPtr(&sceneSampledCloud);

			describer.setInputCloud(sceneSampledCloud);
			describer.setInputNormals(scene_normals);
			describer.setSearchSurface(sceneCloud);
			describer.compute(*sceneDescriptors);

			cout << "scene desribers" << sceneDescriptors->size() << endl;


			//Scenes.insert(std::make_pair(s, 1));
			//int i = myDictionary[s];
			//i = myDictionary.size();
			//bool b = myDictionary.empty();
			//myDictionary.erase(s);


			//objRGBSrcDir = modelMainPath + "/rgb";
			//objDepthSrcDir = modelMainPath + "/depth";
			//std::string line;

			// loading camera intrinsic parameters
			//std::ifstream ifStreamInfo(modelMainPath + "/info.yml");
			//vector<vector<float>> cameraIntrinsicParamtersList;
			/*while (std::getline(ifStreamInfo, line))
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
			}*/
			// loading rotation and transformation matrices for all models
			/*vector<vector<float>> rotationValuesList;
			vector<vector<float>> translationValuesList;
			std::ifstream ifStreamGT(modelMainPath + "/gt.yml");
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
			}*/

			//int i = 0;
			//int modelIndex = -1;
			//for (auto it : recursive_directory_range(objRGBSrcDir))
			//{
			//	modelIndex++;
			//	// Loading depth image and color image

			//	string path = it.path().string();
			//	boost::replace_all(path, "\\", "/");
			//	string colorFilename = path;

			//	//cout << path << endl;
			//	boost::replace_all(path, "rgb", "depth");
			//	string depthFilename = path;
			//	//cout << path << endl;

			//	cv::Mat depthImg = cv::imread(depthFilename, CV_LOAD_IMAGE_UNCHANGED);
			//	cv::Mat colorImg = cv::imread(colorFilename, CV_LOAD_IMAGE_COLOR);
			//	cv::cvtColor(colorImg, colorImg, CV_BGR2RGB); //this will put colors right
			//	// Loading camera pose
			//	string poseFilename = projectSrcDir + "/data/pose/pose" + to_string(index) + ".txt";
			//	Eigen::Matrix4f poseMat;   // 4x4 transformation matrix

			//	vector<float> rotationValues = rotationValuesList[i];
			//	vector<float> translationsValues = translationValuesList[i];
			//	vector<float> camIntrinsicParams = cameraIntrinsicParamtersList[i++];

			//	poseMat(0, 0) = rotationValues[0];
			//	poseMat(0, 1) = rotationValues[1];
			//	poseMat(0, 2) = rotationValues[2];
			//	poseMat(0, 3) = translationsValues[0];
			//	poseMat(1, 0) = rotationValues[3];
			//	poseMat(1, 1) = rotationValues[4];
			//	poseMat(1, 2) = rotationValues[5];
			//	poseMat(1, 3) = translationsValues[1];
			//	poseMat(2, 0) = rotationValues[6];
			//	poseMat(2, 1) = rotationValues[7];
			//	poseMat(2, 2) = rotationValues[8];
			//	poseMat(2, 3) = translationsValues[2];
			//	poseMat(3, 0) = 0;
			//	poseMat(3, 1) = 0;
			//	poseMat(3, 2) = 0;
			//	poseMat(3, 3) = 1;

			//	//cout << "Transformation matrix" << endl << poseMat << endl;

			//	// Setting camera intrinsic parameters of depth camera
			//	float focal = camIntrinsicParams[0];  // focal length
			//	float px = camIntrinsicParams[2]; // principal point x
			//	float py = camIntrinsicParams[5]; // principal point y

			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modelCloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

			if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(modelMainPath, *modelCloudWithNormals) == -1){ PCL_ERROR("Couldn't read file model.ply \n"); return (-1); }
			std::cout << "Loaded" << modelCloudWithNormals->width * modelCloudWithNormals->height << "points" << std::endl;

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);


			pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
			//ne.compute(*model_normals);


			//ne.compute(*scene_normals);

			//cout << "Computing model " << modelName << "  normals" << endl;
			//ne.setInputCloud(modelCloud);
			//ne.compute(*model_normals);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ZLmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			/*pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ZSmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr XLmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr XSmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr YLmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr YSmodelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

			pcl::PointCloud<pcl::Normal>::Ptr ZSmodel_normals(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr XLmodel_normals(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr XSmodel_normals(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr YLmodel_normals(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr YSmodel_normals(new pcl::PointCloud<pcl::Normal>);

			*/
			pcl::PointCloud<pcl::Normal>::Ptr ZLmodel_normals(new pcl::PointCloud<pcl::Normal>);

			modelCloud->points.resize(modelCloudWithNormals->size());
			model_normals->points.resize(modelCloudWithNormals->size());

			for (size_t i = 0; i < modelCloudWithNormals->points.size(); i++) {

				/*if (modelCloudWithNormals->points[i].x<=0)
				{
				PointXYZRGBA p;
				p.x = modelCloudWithNormals->points[i].x;
				p.y = modelCloudWithNormals->points[i].y;
				p.z = modelCloudWithNormals->points[i].z;
				p.r = modelCloudWithNormals->points[i].r;
				p.g = modelCloudWithNormals->points[i].g;
				p.b = modelCloudWithNormals->points[i].b;
				p.a = 255;
				XSmodelCloud->push_back(p);

				Normal n;
				n.normal_x = modelCloudWithNormals->points[i].normal_x;
				n.normal_y = modelCloudWithNormals->points[i].normal_y;
				n.normal_z = modelCloudWithNormals->points[i].normal_z;

				XSmodel_normals->push_back(n);





				}*/

				if (modelCloudWithNormals->points[i].y >= 0 /*&& modelCloudWithNormals->points[i].z > 0 && modelCloudWithNormals->points[i].y > 0*/)
				{

					PointXYZRGBA p;
					p.x = modelCloudWithNormals->points[i].x;
					p.y = modelCloudWithNormals->points[i].y;
					p.z = modelCloudWithNormals->points[i].z;
					p.r = modelCloudWithNormals->points[i].r;
					p.g = modelCloudWithNormals->points[i].g;
					p.b = modelCloudWithNormals->points[i].b;
					p.a = 255;
					ZLmodelCloud->push_back(p);

					Normal n;
					n.normal_x = modelCloudWithNormals->points[i].normal_x;
					n.normal_y = modelCloudWithNormals->points[i].normal_y;
					n.normal_z = modelCloudWithNormals->points[i].normal_z;

					ZLmodel_normals->push_back(n);

				}

				/*	if (modelCloudWithNormals->points[i].y > 0)
				{

				YLmodelCloud->points[i].x = modelCloudWithNormals->points[i].x;
				YLmodelCloud->points[i].y = modelCloudWithNormals->points[i].y;
				YLmodelCloud->points[i].z = modelCloudWithNormals->points[i].z;
				YLmodelCloud->points[i].r = modelCloudWithNormals->points[i].r;
				YLmodelCloud->points[i].g = modelCloudWithNormals->points[i].g;
				YLmodelCloud->points[i].b = modelCloudWithNormals->points[i].b;
				YLmodelCloud->points[i].a = 255;
				YLmodel_normals->points[i].normal_x = modelCloudWithNormals->points[i].normal[0];
				YLmodel_normals->points[i].normal_y = modelCloudWithNormals->points[i].normal[1];
				YLmodel_normals->points[i].normal_z = modelCloudWithNormals->points[i].normal[2];

				}
				if (modelCloudWithNormals->points[i].y <= 0)
				{
				PointXYZRGBA p;
				p.x = modelCloudWithNormals->points[i].x;
				p.y = modelCloudWithNormals->points[i].y;
				p.z = modelCloudWithNormals->points[i].z;
				p.r = modelCloudWithNormals->points[i].r;
				p.g = modelCloudWithNormals->points[i].g;
				p.b = modelCloudWithNormals->points[i].b;
				p.a = 255;
				XSmodelCloud->push_back(p);

				Normal n;
				n.normal_x = modelCloudWithNormals->points[i].normal_x;
				n.normal_y = modelCloudWithNormals->points[i].normal_y;
				n.normal_z = modelCloudWithNormals->points[i].normal_z;

				XSmodel_normals->push_back(n);

				}
				if (modelCloudWithNormals->points[i].z <= 0)
				{

				ZSmodelCloud->points[i].x = modelCloudWithNormals->points[i].x;
				ZSmodelCloud->points[i].y = modelCloudWithNormals->points[i].y;
				ZSmodelCloud->points[i].z = modelCloudWithNormals->points[i].z;
				ZSmodelCloud->points[i].r = modelCloudWithNormals->points[i].r;
				ZSmodelCloud->points[i].g = modelCloudWithNormals->points[i].g;
				ZSmodelCloud->points[i].b = modelCloudWithNormals->points[i].b;
				ZSmodelCloud->points[i].a = 255;
				ZSmodel_normals->points[i].normal_x = modelCloudWithNormals->points[i].normal[0];
				ZSmodel_normals->points[i].normal_y = modelCloudWithNormals->points[i].normal[1];
				ZSmodel_normals->points[i].normal_z = modelCloudWithNormals->points[i].normal[2];

				}
				if (modelCloudWithNormals->points[i].z > 0)
				{

				ZLmodelCloud->points[i].x = modelCloudWithNormals->points[i].x;
				ZLmodelCloud->points[i].y = modelCloudWithNormals->points[i].y;
				ZLmodelCloud->points[i].z = modelCloudWithNormals->points[i].z;
				ZLmodelCloud->points[i].r = modelCloudWithNormals->points[i].r;
				ZLmodelCloud->points[i].g = modelCloudWithNormals->points[i].g;
				ZLmodelCloud->points[i].b = modelCloudWithNormals->points[i].b;
				ZLmodelCloud->points[i].a = 255;
				ZLmodel_normals->points[i].normal_x = modelCloudWithNormals->points[i].normal[0];
				ZLmodel_normals->points[i].normal_y = modelCloudWithNormals->points[i].normal[1];
				ZLmodel_normals->points[i].normal_z = modelCloudWithNormals->points[i].normal[2];

				}*/


				modelCloud->points[i].x = modelCloudWithNormals->points[i].x;
				modelCloud->points[i].y = modelCloudWithNormals->points[i].y;
				modelCloud->points[i].z = modelCloudWithNormals->points[i].z;
				modelCloud->points[i].r = modelCloudWithNormals->points[i].r;
				modelCloud->points[i].g = modelCloudWithNormals->points[i].g;
				modelCloud->points[i].b = modelCloudWithNormals->points[i].b;
				modelCloud->points[i].a = 255;
				model_normals->points[i].normal_x = modelCloudWithNormals->points[i].normal[0];
				model_normals->points[i].normal_y = modelCloudWithNormals->points[i].normal[1];
				model_normals->points[i].normal_z = modelCloudWithNormals->points[i].normal[2];

			}


			if (split)
			{
				modelCloud = ZLmodelCloud;
				model_normals = ZLmodel_normals;
			}


			cout << "Computing model " << modelName << "  normals" << endl;
			ne.setInputCloud(modelCloud);
			ne.compute(*model_normals);

			/*for (size_t i = 0; i < modelCloud->size(); i++)
			{
			modelCloud->at(i).a = 255;
			}*/


			// Create point clouds from depth image and color image using camera intrinsic parameters
			// (1) Compute 3D point from depth values and pixel locations on depth image using camera intrinsic parameters.
			//for (int j = 0; j < depthImg.cols; j+=3)
			//{
			//	for (int i = 0; i < depthImg.rows; i+=3)
			//	{
			//		auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focal, (i - py)*depthImg.at<ushort>(i, j) / focal, depthImg.at<ushort>(i, j), 1);

			//		// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
			//		point = poseMat *point;
			//		// (3) Add the 3D point to vertices in point clouds data.
			//		pcl::PointXYZRGBA p;
			//		p.x = point[0]/1000.0f;
			//		p.y = point[1]/1000.0f;
			//		p.z = point[2]/1000.0f;
			//		p.r = colorImg.at<cv::Vec3b>(i, j)[0];
			//		p.g = colorImg.at<cv::Vec3b>(i, j)[1];
			//		p.b = colorImg.at<cv::Vec3b>(i, j)[2];
			//		p.a = 255;
			//		if (p.x == 0 && p.y == 0 &&p.r==0&&p.g==0&&p.b==0)
			//		{
			//			continue;
			//		}
			//		modelCloud->push_back(p);
			//	}
			//}


			cout << "SCENE: " << challengeName << "  - MODEL: " << modelName << endl;
			//std::cout << " " << it.first << ":" << it.second;

			pcl::visualization::PCLVisualizer viewer2("3d viewer");
			viewer2.addPointCloud(modelCloud, "model");
			viewer2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model");
			while (!viewer2.wasStopped())
			{
				viewer2.spinOnce(100);
			}







			//// a) Load Point clouds (model and scene)
			/*cout << "a) Load Point clouds (model and scene)" << endl;
			std::string model_filename_ = projectSrcDir + "/Data/model_house.pcd";
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

			if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(model_filename_, *modelCloud) == -1){ PCL_ERROR("Couldn't read file model.pcd \n"); return (-1); }
			std::cout << "Loaded" << modelCloud->width * modelCloud->height << "points" << std::endl;

			for (size_t i = 0; i < modelCloud->size(); i++)
			{
			modelCloud->at(i).a = 255;
			}


			std::string scene_filename_ = projectSrcDir + "/Data/scene_clutter.pcd";

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(scene_filename_, *sceneCloud) == -1){ PCL_ERROR("Couldn't read file scene.pcd \n"); return (-1); }
			std::cout << "Loaded" << sceneCloud->width * sceneCloud->height << "points" << std::endl;


			for (size_t j = 0; j < sceneCloud->size(); j++)
			{
			sceneCloud->at(j).a = 255;

			}*/




			////// a) Compute normals



			cout << "a) Compute normals" << endl;


			//pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
			//ne.setSearchMethod(tree);
			//ne.setRadiusSearch(0.01f);




			//// b) Extract key-points from point clouds by downsampling point clouds
			cout << "b) Extract key-points from point clouds by downsampling point clouds" << endl;

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelSampledCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

			if (useISS)
			{

				//
				// Compute keypoints
				//
				pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> iss_detector;

				iss_detector.setSearchMethod(tree);


				model_resolution = computeCloudResolution(modelCloud);

				iss_salient_radius_ = 6 * model_resolution;
				iss_non_max_radius_ = 4 * model_resolution;
				/*iss_normal_radius_ = 4 * model_resolution;
				iss_border_radius_ = 1 * model_resolution;*/
				/*	iss_detector.setNormalRadius(iss_normal_radius_);
				iss_detector.setBorderRadius(iss_border_radius_);*/
				iss_detector.setSalientRadius(iss_salient_radius_);
				iss_detector.setNonMaxRadius(iss_non_max_radius_);
				iss_detector.setThreshold21(iss_gamma_21_);
				iss_detector.setThreshold32(iss_gamma_32_);
				iss_detector.setMinNeighbors(iss_min_neighbors_);
				iss_detector.setNumberOfThreads(iss_threads_);
				iss_detector.setInputCloud(modelCloud);
				iss_detector.compute(*modelSampledCloud);

			}

			else{
				uniform_sampling.setRadiusSearch(modelSamplingRadiusSearch); //the 3D grid leaf size
				uniform_sampling.setInputCloud(modelCloud);
				uniform_sampling.filter(*modelSampledCloud);
			}


			cout << "sampled modelSampledCloud :" + to_string(modelSampledCloud->size()) << endl;

			pcl::visualization::PCLVisualizer viewer5("sampled model");
			viewer5.addPointCloud(modelSampledCloud, "sampledmodel");
			viewer5.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sampledmodel");

			pcl::visualization::PCLVisualizer viewer6("sampled scene");
			viewer6.addPointCloud(sceneSampledCloud, "sampledscene");
			viewer6.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sampledscene");

			//	viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "scene");



			while (!viewer5.wasStopped() && !viewer6.wasStopped())
			{
				viewer5.spinOnce(100);
				viewer6.spinOnce(100);

			}

			viewer5.close();
			viewer6.close();


			//// c) Compute descriptor for keypoints
			cout << "c) Compute descriptor for keypoints" << endl;


			pcl::PointCloud<pcl::SHOT352>::Ptr modelDescriptors(new pcl::PointCloud<pcl::SHOT352>);
			//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelSampledCloudPtr(&modelSampledCloud);
			describer.setRadiusSearch(modeldescriberRadiusSearch);

			describer.setInputCloud(modelSampledCloud);
			describer.setInputNormals(model_normals);
			describer.setSearchSurface(modelCloud);
			describer.compute(*modelDescriptors);




			//// d) Find model-scene key-points correspondences with KdTree

			cout << "d) Find model-scene key-points correspondences with KdTree" << endl;
			pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
			pcl::KdTreeFLANN<SHOT352> match_search;
			match_search.setInputCloud(modelDescriptors);
			std::vector<int> modelKPindices;
			std::vector<int> sceneKPindices;

			for (size_t i = 0; i < sceneDescriptors->size(); ++i)
			{
				std::vector<int> neigh_indices(1);
				std::vector<float> neigh_sqr_dists(1);

				if (!pcl_isfinite(sceneDescriptors->at(i).descriptor[0])) //skipping NaNs
				{
					continue;
				}

				int found_neighs = match_search.nearestKSearch(sceneDescriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
				if (found_neighs == 1 && neigh_sqr_dists[0] < kdnearestDistance)
				{
					pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
					model_scene_corrs->push_back(corr);
					modelKPindices.push_back(corr.index_query);
					sceneKPindices.push_back(corr.index_match);
				}
			}
			pcl::PointCloud<PointType>::Ptr modelKeyPoints(new pcl::PointCloud<PointType>());
			pcl::PointCloud<PointType>::Ptr sceneKeyPoints(new pcl::PointCloud<PointType>());
			pcl::copyPointCloud(*modelSampledCloud, modelKPindices, *modelKeyPoints);
			pcl::copyPointCloud(*sceneSampledCloud, sceneKPindices, *sceneKeyPoints);

			std::cout << "model_scene_corrs: " << model_scene_corrs->size() << std::endl;


			pcl::visualization::PCLVisualizer viewer7("visualizing correspondences");
			viewer7.addPointCloud(originalSceneCloud, "scene");
			//viewer7.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");
			viewer7.addPointCloud(modelCloud, "model");
			viewer7.addCorrespondences<PointXYZRGBA>(modelCloud, sceneCloud, *model_scene_corrs);

			while (!viewer7.wasStopped())
			{
				viewer7.spinOnce(100);
			}
			/*std::vector<pcl::Correspondence> model_scene_corrs;

			pcl::KdTreeFLANN<DescriptorType> match_search;
			pcl::PointCloud<pcl::SHOT352>::Ptr descriptorsPtr(&descriptors);
			match_search.setInputCloud(descriptorsPtr);
			for (int i = 0; i< descriptorsPtr->size(); ++i)
			{
			std::vector<int> neigh_indices(1);
			std::vector<float> neigh_sqr_dists(1);
			int found_neighs = match_search.nearestKSearch(descriptorsPtr->at(i), 1, neigh_indices, neigh_sqr_dists);
			if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)
			{
			pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs.push_back(corr);
			}
			}*/
			//// e) Cluster geometrical correspondence, and finding object instances
			cout << "e) Cluster geometrical correspondence, and finding object instances" << endl;
			//std::vector<pcl::Correspondences> clusters; //output
			pcl::GeometricConsistencyGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA> gc_clusterer;
			gc_clusterer.setGCSize(gcClusteringSize); //1st param
			gc_clusterer.setGCThreshold(gcClusteringThreshold); //2nd param//minimum cluster size, shouldn't be less than 3
			gc_clusterer.setInputCloud(modelSampledCloud);
			gc_clusterer.setSceneCloud(sceneSampledCloud);
			gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);
			std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
				rototranslations;
			std::vector < pcl::Correspondences > clustered_corrs;
			gc_clusterer.recognize(rototranslations, clustered_corrs);
			if (rototranslations.size() <= 0)
			{
				cout << "no instances found, exiting" << endl;
				continue;
			}
			else
			{
				cout << "number of instances: " << rototranslations.size() << endl << endl;


			}
			std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;
			for (size_t i = 0; i < rototranslations.size(); ++i)
			{
				pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>());
				pcl::transformPointCloud(*modelCloud, *rotated_model, rototranslations[i]);
				instances.push_back(rotated_model);
			}
			cout << "f) Refine pose of each instance by using ICP" << endl;

			std::vector<pcl::PointCloud<PointType>::ConstPtr> registeredModelClusteredKeyPoints;// (new pcl::PointCloud<PointType>());
			vector<Eigen::Matrix4f> finalTransformations;
			vector<double> efs;
			for (size_t i = 0; i < rototranslations.size(); ++i)
			{
				pcl::IterativeClosestPoint<PointType, PointType> icp;
				icp.setMaximumIterations(icpsetMaximumIterations);
				icp.setMaxCorrespondenceDistance(icpsetMaxCorrespondenceDistance);
				icp.setUseReciprocalCorrespondences(true);//
				icp.setInputTarget(sceneCloud);
				icp.setInputSource(instances[i]);
				pcl::PointCloud<PointType>::Ptr registered(new pcl::PointCloud<PointType>);
				icp.align(*registered);
				efs.push_back(icp.getEuclideanFitnessEpsilon());
				registeredModelClusteredKeyPoints.push_back(registered);
				finalTransformations.push_back(icp.getFinalTransformation());
				cout << "cluster " << i << " ";
				if (icp.hasConverged())
				{
					cout << "is aligned" << endl;
				}
				else
				{
					cout << "not aligned" << endl;
				}
			}

			//gc_clusterer.cluster(clusters);

			//// f) Refine pose of each instance by using ICP
			/*	vector<pcl::PointCloud<PointType>::Ptr> modelClusteredKeyPoints;//(new pcl::PointCloud<PointType>());

			vector<pcl::PointCloud<PointType>::Ptr> sceneClusteredKeyPoints;// (new pcl::PointCloud<PointType>());
			vector<vector<int>> modelClusteredKPindices;
			vector<vector<int>> sceneClusteredKPindices;
			pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;

			for (size_t i = 0; i < clusters.size(); ++i)
			{
			modelClusteredKPindices.push_back(vector<int>());
			sceneClusteredKPindices.push_back(vector<int>());

			for (size_t j = 0; j < clusters[i].size(); j++)
			{
			modelClusteredKPindices[i].push_back(clusters[i][j].index_query);
			sceneClusteredKPindices[i].push_back(clusters[i][j].index_match);

			}
			pcl::PointCloud<PointType>::Ptr modelKeyPoints(new pcl::PointCloud<PointType>());
			pcl::PointCloud<PointType>::ConstPtr registeredmodelKeyPoints(new pcl::PointCloud<PointType>());

			pcl::PointCloud<PointType>::Ptr sceneKeyPoints(new pcl::PointCloud<PointType>());
			modelClusteredKeyPoints.push_back(modelKeyPoints);
			//registeredModelClusteredKeyPoints.push_back(registeredmodelKeyPoints);

			sceneClusteredKeyPoints.push_back(sceneKeyPoints);



			pcl::copyPointCloud(*modelSampledCloudPtr, modelClusteredKPindices[i], *modelClusteredKeyPoints[i]);
			pcl::copyPointCloud(*sceneSampledCloudPtr, sceneClusteredKPindices[i], *sceneClusteredKeyPoints[i]);

			icp.setInputCloud(modelClusteredKeyPoints[i]);
			//icp.setInputTarget(sceneClusteredKeyPoints[i]);
			icp.setInputTarget(sceneCloud);

			cout << "setting icp parameters" << endl;

			// Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
			icp.setMaxCorrespondenceDistance(0.005);
			// Set the maximum number of iterations (criterion 1)
			icp.setMaximumIterations(5);
			// Set the transformation epsilon (criterion 2)
			//icp.setTransformationEpsilon(1e-8);
			// Set the euclidean distance difference epsilon (criterion 3)
			//icp.setEuclideanFitnessEpsilon(1);


			//icp.setMaxCorrespondenceDistance(0.1f);
			cout << "starting icp align" << endl;
			//icp.setMaximumIterations(50);
			//icp.setMaxCorrespondenceDistance(0.005);
			pcl::PointCloud<PointXYZRGBA>::Ptr registered(new pcl::PointCloud<PointXYZRGBA>);

			icp.align(*registered);

			cout << "done!" << endl;

			//	registeredModelClusteredKeyPoints.push_back(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr(new pcl::PointCloud<pcl::PointXYZRGBA>(*registeredptr)));
			registeredModelClusteredKeyPoints.push_back(registered);
			//registered.clear();
			//	registeredptr.reset();

			}
			*/
			//// g) Do hypothesis verification
			cout << "g) Do hypothesis verification" << endl;

			pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;
			GoHv.setSceneCloud(sceneCloud);
			GoHv.addModels(registeredModelClusteredKeyPoints, true);
			GoHv.setInlierThreshold(GoHvsetInlierThreshold);
			GoHv.setOcclusionThreshold(GoHvsetOcclusionThreshold);
			GoHv.setRegularizer(GoHvRegularizer);
			GoHv.setRadiusClutter(GoHvsetRadiusClutter);
			GoHv.setClutterRegularizer(GoHvClutterRegularizer);
			GoHv.setDetectClutter(true);


			GoHv.setRadiusNormals(GoHvRadiusNormals);
			GoHv.verify();
			std::vector<bool> mask_hv;

			GoHv.getMask(mask_hv);




			/*pcl::GreedyVerification<pcl::PointXYZRGBA,
			pcl::PointXYZRGBA> greedy_hv(3);
			greedy_hv.setResolution(0.02f); //voxel grid is applied beforehand
			greedy_hv.setInlierThreshold(0.005f);
			greedy_hv.setOcclusionThreshold(0.01);

			greedy_hv.setSceneCloud(sceneCloud);
			greedy_hv.addModels(registeredModelClusteredKeyPoints, true);
			greedy_hv.verify();
			std::vector<bool> mask_hv;
			greedy_hv.getMask(mask_hv);*/

			/// Visualize detection result

			cout << "Visualize detection result" << endl;
			pcl::visualization::PCLVisualizer viewer("3d viewer");
			viewer.addPointCloud(originalSceneCloud, "scene");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");
			//viewer.addCorrespondences<PointXYZRGBA>(modelCloud, sceneCloud, *model_scene_corrs);
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "scene");

			double timeTaken = (double)(((double)(clock() - tStart)) / CLOCKS_PER_SEC);

			// run algorithm on all scenes in the test folder and produce output files
			// using the following format:  
			// https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_results_format.md
			// XXXX_YY.yml
			// XXXX: test image
			// YY: object that is present in the image (e.g. bird)
			// 6D pose is written to output file

			string outputFileName = dirChallenge.string() + "/" + colorSceneFilename + "_" + modelName + ".yml";
			boost::iostreams::stream_buffer<boost::iostreams::file_sink> buf(outputFileName);
			std::ostream osout(&buf);

			for (size_t i = 0; i < registeredModelClusteredKeyPoints.size(); i++)
			{
				viewer.addPointCloud(registeredModelClusteredKeyPoints[i], "instance" + to_string(i));

				if (mask_hv[i])
				{
					// output transformation matrix 
					// TODO: find out how to compute score
					double score = 0;//efs[i];

					string rotationValues = to_string(finalTransformations[i](0, 1))
						+", "+ to_string(finalTransformations[i](0, 2))
						+", "+ to_string(finalTransformations[i](1, 0))
						+", "+ to_string(finalTransformations[i](1, 1))
						+", "+ to_string(finalTransformations[i](1, 2))
						+", "+ to_string(finalTransformations[i](2, 0))
						+", "+ to_string(finalTransformations[i](2, 1))
						+", "+ to_string(finalTransformations[i](2, 2));

					string translationValues = to_string(finalTransformations[i](0, 3))
						+", "+ to_string(finalTransformations[i](1, 3))
						+", "+ to_string(finalTransformations[i](2, 3));



					// out writes to file XXXX_YY.txt
					osout << "run_time: " + to_string(timeTaken) + "\r\n";
					osout << "ests:\r\n";
					osout << "- {score: " + to_string(score) + ", R : [" + rotationValues + "], t: [" + translationValues + "]}";
					cout << "instance" + to_string(i) + " good" << endl;
					viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "instance" + to_string(i));

				}
				else
				{
					cout << "instance" + to_string(i) + " bad" << endl;

					viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "instance" + to_string(i));

				}
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "instance" + to_string(i));

			}


			while (!viewer.wasStopped())
			{
				viewer.spinOnce(100);
			}
				}

			}
			catch (const std::exception& e){
				cout << "Instance not found" << endl;
				cout << e.what() << endl;
				cout << "Moving to next test scene!" << endl;
			}
		}

	}
	return 0;

}











#include <iostream>

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

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

using namespace std;
using namespace pcl;


int
main(int argc, char** argv)
{
	std::string projectSrcDir = PROJECT_SOURCE_DIR;

	//// a) Load Point clouds (model and scene)
	std::string model_filename_ = projectSrcDir + "/Data/model_house.pcd";
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(model_filename_, *modelCloud) == -1){ PCL_ERROR("Couldn't read file model.pcd \n"); return (-1); }
	std::cout << "Loaded" << modelCloud->width * modelCloud->height << "points" << std::endl;

	/*for (size_t i = 0; i < modelCloud->width; i++)
	{
		for (size_t j = 0; j < modelCloud->height; j++)
		{
			modelCloud->at(i, j).a = 255;

		}

	}*/

	
	
	std::string scene_filename_ = projectSrcDir + "/Data/scene_clutter.pcd";

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scenelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(scene_filename_, *scenelCloud) == -1){ PCL_ERROR("Couldn't read file scene.pcd \n"); return (-1); }
	std::cout << "Loaded" << scenelCloud->width * scenelCloud->height << "points" << std::endl;

	/*for (size_t i = 0; i < scenelCloud->width; i++)
	{
		for (size_t j = 0; j < scenelCloud->height; j++)
		{
			scenelCloud->at(i, j).a = 255;

		}

	}*/
	

	//// a) Compute normals

	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.01f);

	ne.setInputCloud(modelCloud);
	pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*model_normals);

	ne.setInputCloud(scenelCloud);
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*scene_normals);

	//// b) Extract key-points from point clouds by downsampling point clouds

	pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
	uniform_sampling.setRadiusSearch(0.02f); //the 3D grid leaf size

	pcl::PointCloud<pcl::PointXYZRGBA> modelSampledCloud;
	uniform_sampling.setInputCloud(modelCloud);
	uniform_sampling.filter(modelSampledCloud);


	pcl::PointCloud<pcl::PointXYZRGBA> sceneSampledCloud;
	uniform_sampling.setInputCloud(scenelCloud);
	uniform_sampling.filter(sceneSampledCloud);


	//// c) Compute descriptor for keypoints
	pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> describer;
	describer.setRadiusSearch(0.02f);

	pcl::PointCloud<pcl::SHOT352>::Ptr modelDescriptors(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelSampledCloudPtr(&modelSampledCloud);

	describer.setInputCloud(modelSampledCloudPtr);
	describer.setInputNormals(model_normals);
	describer.setSearchSurface(modelCloud);
	describer.compute(*modelDescriptors);

	pcl::PointCloud<pcl::SHOT352>::Ptr sceneDescriptors(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneSampledCloudPtr(&sceneSampledCloud);

	describer.setInputCloud(sceneSampledCloudPtr);
	describer.setInputNormals(scene_normals);
	describer.setSearchSurface(scenelCloud);
	describer.compute(*sceneDescriptors);


	//// d) Find model-scene key-points correspondences with KdTree


	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud(modelDescriptors);
	//std::vector<int> modelKPindices;
	//std::vector<int> sceneKPindices;

	for (size_t i = 0; i < sceneDescriptors->size(); ++i)
	{
		std::vector<int> neigh_indices(1);
		std::vector<float> neigh_sqr_dists(1);

		if (!pcl_isfinite(sceneDescriptors->at(i).descriptor[0])) //skipping NaNs
		{
			continue;
		}

		int found_neighs = match_search.nearestKSearch(sceneDescriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)
		{
			pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back(corr);
			//modelKPindices.push_back(corr.index_query);
			//sceneKPindices.push_back(corr.index_match);
		}
	}
	//pcl::PointCloud<PointType>::Ptr modelKeyPoints(new pcl::PointCloud<PointType>());
	//pcl::PointCloud<PointType>::Ptr sceneKeyPoints(new pcl::PointCloud<PointType>());
	//pcl::copyPointCloud(*modelSampledCloudPtr, modelKPindices, *modelKeyPoints);
	//pcl::copyPointCloud(*sceneSampledCloudPtr, sceneKPindices, *sceneKeyPoints);

	std::cout << "model_scene_corrs: " << model_scene_corrs->size() << std::endl;




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

	std::vector<pcl::Correspondences> clusters; //output
	pcl::GeometricConsistencyGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA> gc_clusterer;
	gc_clusterer.setGCSize(0.5f); //1st param
	gc_clusterer.setGCThreshold(5); //2nd param
	gc_clusterer.setInputCloud(modelSampledCloudPtr);
	gc_clusterer.setSceneCloud(sceneSampledCloudPtr);
	gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);
	gc_clusterer.cluster(clusters);

	//// f) Refine pose of each instance by using ICP
	vector<pcl::PointCloud<PointType>::Ptr> modelClusteredKeyPoints;//(new pcl::PointCloud<PointType>());
	std::vector<pcl::PointCloud<PointType>::ConstPtr> registeredModelClusteredKeyPoints;// (new pcl::PointCloud<PointType>());

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
		icp.setInputTarget(sceneClusteredKeyPoints[i]);
		cout << "setting icp parameters" << endl;

		// Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
		icp.setMaxCorrespondenceDistance(0.05);
		// Set the maximum number of iterations (criterion 1)
		//icp.setMaximumIterations(50);
		// Set the transformation epsilon (criterion 2)
		//icp.setTransformationEpsilon(1e-8);
		// Set the euclidean distance difference epsilon (criterion 3)
		icp.setEuclideanFitnessEpsilon(1);


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

	//// g) Do hypothesis verification

	pcl::GreedyVerification<pcl::PointXYZRGBA,
		pcl::PointXYZRGBA> greedy_hv(5);
	greedy_hv.setResolution(0.02f); //voxel grid is applied beforehand
	greedy_hv.setInlierThreshold(0.05f);
	greedy_hv.setOcclusionThreshold(0.01);

	greedy_hv.setSceneCloud(scenelCloud);
	greedy_hv.addModels(registeredModelClusteredKeyPoints, true);
	greedy_hv.verify();
	std::vector<bool> mask_hv;
	greedy_hv.getMask(mask_hv);

	/// Visualize detection result


	pcl::visualization::PCLVisualizer viewer("3d viewer");
	viewer.addPointCloud(scenelCloud, "scene");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "scene");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 255, "scene");

	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "scene");



	for (size_t i = 0; i < registeredModelClusteredKeyPoints.size(); i++)
	{

		viewer.addPointCloud(registeredModelClusteredKeyPoints[i], "instance" + to_string(i));
		if (mask_hv[i])
		{
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "instance" + to_string(i));

		}
		else
		{
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "instance" + to_string(i));

		}
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "instance" + to_string(i));

	}


	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}




	return 0;
}













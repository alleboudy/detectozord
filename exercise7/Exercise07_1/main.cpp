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
main (int argc, char** argv)
{
    std::string projectSrcDir = PROJECT_SOURCE_DIR;
    
    //// a) Load Point clouds (model and scene)
    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
    std::string model_filename_ = projectSrcDir + "/Data/model_house.pcd";
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(model_filename_, *modelCloud) == -1){ PCL_ERROR("Couldn't read file model.pcd \n"); return (-1); }
	std::cout << "Loaded" << modelCloud->width * modelCloud->height << "points" << std::endl;

    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
    std::string scene_filename_ = projectSrcDir + "/Data/scene_clutter.pcd";
    
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scenelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(scene_filename_, *scenelCloud) == -1){ PCL_ERROR("Couldn't read file scene.pcd \n"); return (-1); }
	std::cout << "Loaded" << scenelCloud->width * scenelCloud->height << "points" << std::endl;
   
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
	uniform_sampling.setRadiusSearch(0.05f); //the 3D grid leaf size

	pcl::PointCloud<pcl::PointXYZRGBA> modelSampledCloud;
	uniform_sampling.setInputCloud(modelCloud);
	uniform_sampling.filter(modelSampledCloud);


	pcl::PointCloud<pcl::PointXYZRGBA> sceneSampledCloud;
	uniform_sampling.setInputCloud(scenelCloud);
	uniform_sampling.filter(sceneSampledCloud);
    
    
    //// c) Compute descriptor for keypoints
    
	pcl::PointCloud<pcl::SHOT352> descriptors;//(pcl::PointCloud<pcl::SHOT352>());
	pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> describer;
	describer.setRadiusSearch(0.05f);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelSampledCloudPtr(&modelSampledCloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sceneSampledCloudPtr(&sceneSampledCloud);

	describer.setInputCloud(modelSampledCloudPtr);
	describer.setInputNormals(model_normals);
	describer.setSearchSurface(sceneSampledCloudPtr);
	describer.compute(descriptors);
	
    //// d) Find model-scene key-points correspondences with KdTree
	std::vector<pcl::Correspondence> model_scene_corrs;

	//pcl::CorrespondencesPtr model_scene_corr(new pcl::Correspondences());
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
	}
    //// e) Cluster geometrical correspondence, and finding object instances
    
	//pcl::CorrespondencesPtr m_s_corrs; //fill it
	std::vector<pcl::Correspondences> clusters; //output
	pcl::GeometricConsistencyGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA> gc_clusterer;
	gc_clusterer.setGCSize(0.01f); //1st param
	gc_clusterer.setGCThreshold(5); //2nd param
	gc_clusterer.setInputCloud(modelSampledCloudPtr);
	gc_clusterer.setSceneCloud(sceneSampledCloudPtr);
	gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);
	gc_clusterer.cluster(clusters);
    //// f) Refine pose of each instance by using ICP
    
    
    //// g) Do hypothesis verification
    
    
    /// Visualize detection result
    
    
    
    
    return 0;
}











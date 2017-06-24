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


int
main (int argc, char** argv)
{
    std::string projectSrcDir = PROJECT_SOURCE_DIR;
    
    //// Load Point clouds (3 models and 2 scenes)
    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
    std::string model_filename_ = projectSrcDir + "/Data/model_house.pcd";
    //std::string model_filename_ = projectSrcDir + "/Data/model_bird.pcd";
    //std::string model_filename_ = projectSrcDir + "/Data/model_bond.pcd";
    
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
    //std::string scene_filename_ = projectSrcDir + "/Data/scene_clutter.pcd";
    std::string scene_filename_ = projectSrcDir + "/Data/scene_planar.pcd";
    
    // Copy the pipeline from exercise 7.1.
    // Then try to detect 3 object in 2 scene (total 6 combination)
    // If it does not work well, improve the pipeline

    
    
    /// Visualize detection result
   
    
    
    
    
    return 0;
}











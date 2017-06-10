


#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>

int
main (int argc, char** argv)
{
    std::string projectSrcDir = PROJECT_SOURCE_DIR;
    
    // Load point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ref (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trg (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB> (projectSrcDir + "/data/pointcloud_chair0.pcd", *cloud_ref);
    pcl::io::loadPCDFile<pcl::PointXYZRGB> (projectSrcDir + "/data/pointcloud_chair1.pcd", *cloud_trg);
    
    
    // Register two point clouds using ICP according to slides
    
    
    // Visualize the registered point clouds

    
    return 0;
}



#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>

int
main (int argc, char** argv)
{
    std::string projectSrcDir = PROJECT_SOURCE_DIR;
    
    // Load [.pcd::] file to pcl::PointCloud
    std::string modelFilename = projectSrcDir + "/data/bun0.pcd";
    
    
    // Visualize the point clouds
    
    
    // Compute normals of the point clouds
    
    
    // Visualize the point clouds with computed normals
   
    
    return 0;
}




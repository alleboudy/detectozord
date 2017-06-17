#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
    std::string projectSrcDir = PROJECT_SOURCE_DIR;

    // Camera intrinsic parameters of depth camera
    float focal = 570.f; // focal length
    float px = 319.5f;   // principal point x
    float py = 239.5f;   // principal point y

    // loop for depth images on multiple view
    for (int i = 0; i < 8; i++)
    {

        // Loading depth image and color image using OpenCV
        std::string depthFilename = projectSrcDir + "/data/depth/depth" + std::to_string(i) + ".png";
        std::string colorFilename = projectSrcDir + "/data/color/color" + std::to_string(i) + ".png";

        // Create point clouds in pcl::PointCloud type from depth image and color image using camera intrinsic parameters
        // Part of this process is similar to Exercise 5-1, so you can reuse the part of the code.
        // The provided depth image is millimeter scale but the default scale in PCL is meter scale
        // So the point clouds should be scaled to meter scale, during point cloud computation

        // Downsample point clouds so that the point density is 1cm by uisng pcl::VoxelGrid<pcl::PointXYZRGB > function
        // point density can be set by using pcl::VoxelGrid::setLeafSize() function

        // Visualize the point clouds

        // Save the point clouds as [.pcd] file
        std::string pointFilename = projectSrcDir + "/data/pointclouds/pointclouds" + std::to_string(i) + ".pcd";
    }

    // Try to register the 8 point clouds on multiple views using ICP
    // First, you can try similar code (ICP with point to point metrics) in exercise 6-2, and show the result.
    // Then you can try pcl::IterativeClosestPointWithNormals which is an ICP implementation using point to plane metrics instead.
    // In case of point to plane metrics, you need to compute normals of point clouds.

    // Visualize the registered and merged point clouds

    return 0;
}

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

using namespace std;
using namespace cv;


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
		Mat depthImg = imread(depthFilename, CV_LOAD_IMAGE_UNCHANGED);
		Mat colorImg = imread(colorFilename, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(colorImg, colorImg, CV_BGR2RGB); //this will put colors right
        // Create point clouds in pcl::PointCloud type from depth image and color image using camera intrinsic parameters
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Part of this process is similar to Exercise 5-1, so you can reuse the part of the code.
		for (int j = 0; j < depthImg.cols; j++)
		{
			for (int i = 0; i < depthImg.rows; i++)
			{
				auto point = Eigen::Vector4f((j - px)*depthImg.at<ushort>(i, j) / focal, (i - py)*depthImg.at<ushort>(i, j) / focal, depthImg.at<ushort>(i, j), 1);

				/*
				// (2) Translate 3D point in local coordinate system to 3D point in global coordinate system using camera pose.
				point = poseMat *point;*/
				// (3) Add the 3D point to vertices in point clouds data.
				pcl::PointXYZRGB p;
				p.x = point[0]/1000.;
				p.y = point[1]/1000.;
				p.z = point[2]/1000.;
				p.r = colorImg.at<Vec3b>(i, j)[0];
				p.g = colorImg.at<Vec3b>(i, j)[1];
				p.b = colorImg.at<Vec3b>(i, j)[2];

				cloud->push_back(p);
			/*	vertices.push_back(point);
				// (4) Also compute the color of 3D point and add it to colors in point clouds data.
				colors.push_back(colorImg.at<Vec3b>(i, j));*/
				

			}
		}


        // The provided depth image is millimeter scale but the default scale in PCL is meter scale
        // So the point clouds should be scaled to meter scale, during point cloud computation

        // Downsample point clouds so that the point density is 1cm by uisng pcl::VoxelGrid<pcl::PointXYZRGB > function
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr downSampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::VoxelGrid<pcl::PointXYZRGB> vxl;
		vxl.setInputCloud(cloud);
		vxl.setLeafSize(0.01f, 0.01f, 0.01f);
		vxl.filter(*downSampledCloud);
        // point density can be set by using pcl::VoxelGrid::setLeafSize() function

        // Visualize the point clouds

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZRGB>(downSampledCloud, "img" + i);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "img"+i);
		viewer->initCameraParameters();
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
		}
        // Save the point clouds as [.pcd] file
        std::string pointFilename = projectSrcDir + "/data/pointclouds/pointclouds" + std::to_string(i) + ".pcd";
		pcl::io::savePCDFileASCII(pointFilename, *downSampledCloud);
		std::cerr << "Saved " << i << std::endl;

    }

    // Try to register the 8 point clouds on multiple views using ICP
    // First, you can try similar code (ICP with point to point metrics) in exercise 6-2, and show the result.
    // Then you can try pcl::IterativeClosestPointWithNormals which is an ICP implementation using point to plane metrics instead.
    // In case of point to plane metrics, you need to compute normals of point clouds.

    // Visualize the registered and merged point clouds

    return 0;
}

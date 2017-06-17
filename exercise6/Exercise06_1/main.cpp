#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>

using namespace std;
int main(int argc, char** argv)
{
	std::string projectSrcDir = PROJECT_SOURCE_DIR;

	// Load [.pcd::] file to pcl::PointCloud
	std::string modelFilename = projectSrcDir + "/data/bun0.pcd";

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(modelFilename, *cloud) == -1){ PCL_ERROR("Couldn't read file bun0.pcd \n"); return (-1); }
	std::cout << "Loaded" << cloud->width * cloud->height << "points" << std::endl;

	// Visualize the point clouds

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "Bun0");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Bun0");
	viewer->initCameraParameters();
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}

	// Compute normals of the point clouds


	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.01f);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*cloud_normals);

	// Visualize the point clouds with computed normals
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer1->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 1, 0.01, "normals");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "normals");
	while (!viewer1->wasStopped())
	{
		viewer1->spinOnce(100);
	}

	return 0;
}




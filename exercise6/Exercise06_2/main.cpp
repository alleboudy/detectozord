#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>

int
main(int argc, char** argv)
{
	std::string projectSrcDir = PROJECT_SOURCE_DIR;

	// Load point clouds
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ref(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trg(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::io::loadPCDFile<pcl::PointXYZRGB>(projectSrcDir + "/data/pointcloud_chair0.pcd", *cloud_ref);
	pcl::io::loadPCDFile<pcl::PointXYZRGB>(projectSrcDir + "/data/pointcloud_chair1.pcd", *cloud_trg);
	


	// Register two point clouds using ICP according to slides

	/*pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ref(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trg(new pcl::PointCloud<pcl::PointXYZ>);*/
	//open or fill in the two clouds
	pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
	icp.setInputCloud(cloud_ref);
	icp.setInputTarget(cloud_trg);
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
	pcl::PointCloud<pcl::PointXYZRGB> registered;
	cout << "starting icp align" << endl;
	icp.align(registered);
	cout << "done!" << endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr registeredptr(&registered);

	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;



	// Visualize the registered point clouds
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ref, "cloud_ref");
	viewer->addPointCloud<pcl::PointXYZRGB>(registeredptr, "registeredptr");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "registeredptr");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud_ref");
	viewer->initCameraParameters();
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("3D Viewer before registeration"));
	viewer1->setBackgroundColor(0, 0, 0);

	viewer1->addPointCloud<pcl::PointXYZRGB>(cloud_trg, "cloud_trgOriginal");
	viewer1->addPointCloud<pcl::PointXYZRGB>(cloud_ref, "cloud_ref");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud_ref");
	viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud_trgOriginal");
	viewer1->initCameraParameters();


	while (!viewer->wasStopped() || !viewer1->wasStopped())
	{
		viewer->spinOnce(100);
		viewer1->spinOnce(100);
	}







	return 0;
}



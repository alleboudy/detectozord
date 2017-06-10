#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>

using namespace plc;
using namespace io;
using namespace std;

int main(int argc, char **argv)
{
    string projectSrcDir = PROJECT_SOURCE_DIR;

    // Load [.pcd::] file to pcl::PointCloud
    string modelFilename = projectSrcDir + "/data/bun0.pcd";

    // Visualize the point clouds
    // create a cloud object to store the point cloud
    PointCloud<PointXYZ>::Ptr cloud = new PointCloud<PointXYZ>();
    if (loadPCDFile<PointXYZ>(modelFilename, *cloud) == -1)
    {
        PLC_ERROR("Couldn't read file: " + modelFilename);
        return (-1);
    }
    cout << "Loaded" << cloud->width * cloud->height << "points" << endl;
    visualization::PCLVisualizer plcVisualizer = new PCLVisualizer();
    plcVisualizer->addPointCloud(cloud, "pointcloud");

    // add a second point cloud (colored) to the visualization
    visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
    plcVisualizer->addPointCloud<PointXYZRGB>(cloud, rgb, "colored point cloud");

    // 2nd example
    boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));

    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<PointXYZ>(cloud, "Bun0");
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Bun0");
    viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
    // Compute normals of the point clouds

    // Visualize the point clouds with computed normals

    return 0;
}

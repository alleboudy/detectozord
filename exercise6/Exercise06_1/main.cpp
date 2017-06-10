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
    std::string projectSrcDir = PROJECT_SOURCE_DIR;

    // Load [.pcd::] file to pcl::PointCloud
    std::string modelFilename = projectSrcDir + "/data/bun0.pcd";

    // Visualize the point clouds
    // create a cloud object to store the point cloud
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    if (loadPCDFile<PointXYZ>(modelFilename, *cloud) == -1)
    {
        PLC_ERROR("Couldn't read file: " + modelFilename);
        return (-1);
    }
    cout << "Loaded" << cloud->width * cloud->height << "points" <<endl;

    // Compute normals of the point clouds

    // Visualize the point clouds with computed normals

    return 0;
}

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/triangulation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/registration/icp.h>
// including opencv2 headers
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/uniform_sampling.h>

#include <pcl/io/openni2_grabber.h>
#include <pcl/visualization/cloud_viewer.h>


#include <OpenNI.h>
#include <PS1080.h>
#include<cmath>
#include <Eigen/Dense>
#include <thread>


using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace openni;

bool debug = true;


/*pcl::PointCloud<pcl::PointXYZRGBA>::Ptr birdCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr houseCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr canCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr crackerCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr shoeCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
*/





template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

std::string exec(string path2classifier, string path2plyFile, string flag, std::vector<std::string>& paths, std::vector<std::string>& labels) {
	string cmd = "cd " + path2classifier + " & python " + path2classifier + "requestClassification.py " + flag + " " + path2plyFile;
	std::array<char, 128> buffer;
	std::string result;
	std::shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
	if (!pipe) throw std::runtime_error("popen() failed!");
	while (!feof(pipe.get())) {
		if (fgets(buffer.data(), 128, pipe.get()) != NULL)
			result += buffer.data();
	}
	if(debug) cout << "El classification yastaaaaaaa!!!" << endl;;
	if(debug) cout << result << endl;
	char delim = '\n';
	vector<string>alllines;
	alllines = split(result, delim);
	delim = ',';
	for (size_t i = 0; i < alllines.size(); i++)
	{
		vector<string>line;

		line = split(alllines[i], delim);
		paths.push_back(line[0]);
		labels.push_back(line[1]);

	}
	return result;

}

// Convert to colored depth image
cv::Mat convColoredDepth(cv::Mat& depthImg, float minThresh = 0, float maxThresh = 0){
	cv::Mat coloredDepth = depthImg.clone();

	double min;
	double max;
	if (minThresh == 0 && maxThresh == 0){
		cv::minMaxIdx(depthImg, &min, &max);
	}
	else{
		min = minThresh;
		max = maxThresh;
	}
	coloredDepth -= min;
	cv::convertScaleAbs(coloredDepth, coloredDepth, 255 / (max - min));
	cv::applyColorMap(coloredDepth, coloredDepth, cv::COLORMAP_JET);

	return coloredDepth;
}


bool savePointCloudsPLY(string filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	std::ofstream fout;
	fout.open(filename.c_str());
	if (fout.fail()){
		cerr << "file open error:" << filename << endl;
		return false;
	}

	int pointNum = points->size();

	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "element vertex " << pointNum << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	if (normals != NULL)
	{
		fout << "property float normal_x" << endl;
		fout << "property float normal_y" << endl;
		fout << "property float normal_z" << endl;
	}

	fout << "property uchar red" << endl;
	fout << "property uchar green" << endl;
	fout << "property uchar blue" << endl;
	fout << "property uchar alpha" << endl;
	fout << "end_header" << endl;

	for (int i = 0; i < pointNum; i++){



		fout << points->at(i).x << " " << points->at(i).y << " " << points->at(i).z;
		if (normals != NULL)
			fout << " " << normals->at(i).normal_x << " " << normals->at(i).normal_y << " " << normals->at(i).normal_z;
		fout << " " << static_cast<int>(points->at(i).r) << " " << static_cast<int>(points->at(i).g) << " " << static_cast<int>(points->at(i).b) << " " << 255 << endl;
	}

	fout.close();

	return true;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr  processCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){


	string locationForOutputClouds = "C:/Users/ahmad/Documents/locationforclouds/";
	// Read in the cloud data
	std::string projectSrcDir = PROJECT_SOURCE_DIR;

	string path2classifier = "C:/Users/ahmad/Documents/pointnet/pointnet/";

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGBA>);
	/*string msg = "Couldn't read file C:\\Users\\ahmad\\Downloads\\challenge2_val\\scenesClouds\\05-0.ply \n";
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>("C:\\Users\\ahmad\\Desktop\\testscenes\\challenge1_5-1.ply", *cloud) == -1){ PCL_ERROR(msg.c_str()); return (-1); }
	if(debug) cout << "Loaded" << cloud->width * cloud->height << "points" << std::endl;

	*/


	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);





	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr originalSceneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);









	pcl::copyPointCloud(*cloud, *originalSceneCloud);


	//pcl::visualization::PCLVisualizer viewer3("clustered instances");
	//viewer3.addPointCloud(originalSceneCloud, "scene");
	/*
	while (!viewer3.wasStopped())
	{
	viewer3.spinOnce(100);
	//		viewer4.spinOnce(100);

	}
	*/


	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	//pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	//vg.setInputCloud(cloud);
	//vg.setLeafSize(0.0009f, 0.0009f, 0.0009f);
	//vg.filter(*cloud_filtered);

	pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
	uniform_sampling.setInputCloud(cloud);
	uniform_sampling.setRadiusSearch(0.001);
	uniform_sampling.filter(*cloud_filtered);


	if(debug) cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; //*

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.02);

	int i = 0, nr_points = (int)cloud_filtered->points.size();
	int prevSize = 0;
	int repeatCounter = 5;
	while (cloud_filtered->points.size() > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			if(debug) cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Get the points associated with the planar surface
		extract.filter(*cloud_plane);
		if(debug) cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;
		if (prevSize == cloud_plane->points.size())
		{
			repeatCounter--;
		}
		else
		{
			prevSize = cloud_plane->points.size();

		}
		// Remove the planar inliers, extract the rest
		if (repeatCounter != 0)
		{


			double z_min = -1.f, z_max = 0;

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZRGBA>());
			pcl::ConvexHull<pcl::PointXYZRGBA> hull;
			// hull.setDimension (2); // not necessarily needed, but we need to check the dimensionality of the output
			hull.setInputCloud(cloud_plane);
			hull.reconstruct(*hull_points);

			if (hull.getDimension() == 2)
			{
				if(debug) cout << "using prism to remove outlier" << endl;
				pcl::ExtractPolygonalPrismData<pcl::PointXYZRGBA> prism;
				prism.setInputCloud(cloud_filtered);
				prism.setInputPlanarHull(hull_points);
				prism.setHeightLimits(z_min, z_max);

				prism.segment(*inliers);
			}
			else
				PCL_ERROR("The input cloud does not represent a planar surface.\n");


		}


		extract.setIndices(inliers);

		extract.setNegative(true);
		extract.filter(*cloud_f);
		*cloud_filtered = *cloud_f;
		if (repeatCounter == 0)
		{
			repeatCounter = 5;
		}
	}

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance(0.02); // 2cm
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);

	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);
	if(debug) cout << "Euclidean cluster done!" << endl;

	int j = 0;
	vector<	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> finalClouds;// (new pcl::PointCloud<pcl::PointXYZRGBA>);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		if(debug) cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		if (cloud_cluster->size() < 500)
		{
			continue;
			if(debug) cout << "skipping instance" << endl;
		}

		//densifying the clouds

		if (cloud_cluster->size() < 2048)
		{



			pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA> mls;
			mls.setInputCloud(cloud_cluster);
			mls.setSearchRadius(0.03);
			mls.setPolynomialFit(true);
			mls.setPolynomialOrder(2);
			mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::SAMPLE_LOCAL_PLANE);
			mls.setUpsamplingRadius(0.01);// has 2 be larger than setUpsamplingStepSize
			mls.setUpsamplingStepSize(0.008);//smaller increases the generated points
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGBA>());
			mls.process(*cloud_cluster);
			if(debug) cout << "upsampled cloud" << cloud_cluster->size() << endl;
		}
		if (cloud_cluster->size() == 0)
		{
			continue;
		}

		/*pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> ne;
		ne.setNumberOfThreads(8);
		ne.setInputCloud(cloud_smoothed);
		ne.setRadiusSearch(0.01);
		Eigen::Vector4f centroid;
		compute3DCentroid(*cloud_smoothed, centroid);
		ne.setViewPoint(centroid[0], centroid[1], centroid[2]);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normals);
		for (size_t i = 0; i < cloud_normals->size(); ++i)
		{
		cloud_normals->points[i].normal_x *= -1;
		cloud_normals->points[i].normal_y *= -1;
		cloud_normals->points[i].normal_z *= -1;
		}
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<pcl::PointNormal>());
		concatenateFields(*cloud_smoothed, *cloud_normals, *cloud_smoothed_normals);
		*/
		/*
		pcl::Poisson<pcl::PointNormal> poisson;
		poisson.setDepth(9);
		poisson.setInputCloud
		(cloud_smoothed_normals);
		pcl::PolygonMesh mesh;
		poisson.reconstruct(mesh);
		*/
		//float step = 0.1f;
		/*for (size_t z = 0; z < cloud_cluster->size(); z++)
		{

		dense_cloud_cluster->push_back(cloud_cluster->points[z]);
		pcl::PointXYZRGBA newPoint;
		newPoint.r = cloud_cluster->points[z].r;
		newPoint.g = cloud_cluster->points[z].g;
		newPoint.b = cloud_cluster->points[z].b;
		newPoint.a = 255;


		newPoint.x = cloud_cluster->points[z].x + cloud_cluster->points[z].x*step;
		newPoint.y = cloud_cluster->points[z].y;
		newPoint.z = cloud_cluster->points[z].z;
		dense_cloud_cluster->push_back(newPoint);
		newPoint.y = cloud_cluster->points[z].y + cloud_cluster->points[z].y*step;
		newPoint.x = cloud_cluster->points[z].x;
		newPoint.z = cloud_cluster->points[z].z;
		dense_cloud_cluster->push_back(newPoint);
		newPoint.z = cloud_cluster->points[z].z + cloud_cluster->points[z].z*step;
		newPoint.x = cloud_cluster->points[z].x;
		newPoint.y = cloud_cluster->points[z].y;
		dense_cloud_cluster->push_back(newPoint);


		}*/

		string plyPath = locationForOutputClouds + "cloud_cluster_" + to_string(j) + ".ply";
		savePointCloudsPLY(plyPath, cloud_cluster, NULL);
		std::vector<std::string> paths;
		std::vector<std::string> labels;
		string res = exec(path2classifier, plyPath, "--ply_path ", paths, labels);
		if (labels[0] == "bird")//sorry I'm doing this, I'm really desperate now -.-
		{
			int r = 0, g = 0, b = 0;
			for (size_t h = 0; h < cloud_cluster->size(); h++)
			{
				r += cloud_cluster->points[h].r;
				g += cloud_cluster->points[h].g;
				b += cloud_cluster->points[h].b;

			}
			if (g > r + b)
			{
				if(debug) cout << "it is a shoe " << endl;
				labels[0] = "shoe";

			}
			float averageYellow = (g + r) / 2.0;
			if (averageYellow - 0.1*averageYellow < g&& g < averageYellow + 0.1*averageYellow && averageYellow - 0.1*averageYellow < r&& r < averageYellow + 0.1*averageYellow)
			{
				if(debug) cout << "Possibly a yellow piece!" << endl;
				//	continue;
			}
		}
		
		if(debug) cout << labels[0] << ":" << paths[0] << endl;


		

		//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr currentModel(new pcl::PointCloud<pcl::PointXYZRGBA>);

		int r=0, g=0, b=0;
		if (labels[0] == "bird")
		{
			r = 255;
			b = 255;
			//currentModel = birdCloud;
		}
		else if (labels[0] == "house")
		{
			b = 255;
			//currentModel = houseCloud;
		}
		else if (labels[0] == "cracker")
		{
			b = 255;
			g = 255;
			//currentModel = crackerCloud;
		}
		else if (labels[0] == "can")
		{
			
			r = 255;
			//currentModel = canCloud;
		}
		else if (labels[0] == "shoe")
		{
			g = 255;
			//currentModel = shoeCloud;
		}
		
		if (debug) cout << "alignment" << endl;


		/*vector<Eigen::Matrix4f> finalTransformations;
		vector<double> efs;
		//int icpsetMaximumIterations = 50;
		//float icpsetMaxCorrespondenceDistance = 0.02f;
		pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
		icp.setMaximumIterations(50);
		//icp.setMaxCorrespondenceDistance(icpsetMaxCorrespondenceDistance);
		//icp.setUseReciprocalCorrespondences(false);//
		icp.setInputTarget(cloud_cluster);
		icp.setInputSource(currentModel);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr registered(new pcl::PointCloud<pcl::PointXYZRGBA>);
		icp.align(*registered);
		efs.push_back(icp.getEuclideanFitnessEpsilon());
		//registeredModelClusteredKeyPoints.push_back(registered);
		finalTransformations.push_back(icp.getFinalTransformation());
		cout << "cluster " << i << " ";
		if (icp.hasConverged())
		{
			cout << "is aligned" << endl;
		}
		else
		{
			cout << "not aligned" << endl;
		}

		if (debug)cout << "showing result" << endl;
		pcl::visualization::PCLVisualizer viewer3("clustered instances");
		//viewer3.addPointCloud(birdCloud, "birdCloud");
		//viewer3.addPointCloud(canCloud, "canCloud");
		//viewer3.addPointCloud(shoeCloud, "shoeCloud");
		for (size_t f = 0; f < registered->size(); f++)
		{
			registered->points[f].g = 0;
			registered->points[f].b = 0;


		}
		viewer3.addPointCloud(registered, "registered");
		viewer3.addPointCloud(cloud, "cloud");

		//viewer3.addPointCloud(crackerCloud, "crackerCloud");
		viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "registered" );



		while (!viewer3.wasStopped())
		{
		viewer3.spinOnce(100);


		}*/
		

		


		

		for (size_t n = 0; n < cloud_cluster->size(); n++)
		{
			//finalClouds[i]->points[l];
			cloud_cluster->points[n].r = r;
			//cloud_cluster->points[n].g = 255;
			cloud_cluster->points[n].b = b;
			cloud_cluster->points[n].g = g;

			cloud->push_back(cloud_cluster->points[n]);
			//	if(debug) cout << "changed colors";
		}
		finalClouds.push_back(cloud_cluster);

		//	viewer3.addPointCloud(cloud_cluster, "instance" + to_string(j));
		//viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "instance");
		//viewer3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "instance" + to_string(j));


		//while (!viewer3.wasStopped())
		//{
		//	viewer3.spinOnce(100);
		//	//		viewer4.spinOnce(100);

		//}



		j++;
	}



	return cloud;

}
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr changeAlittle(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){
	for (size_t i = 0; i < cloud->size(); i++)
	{
		cloud->points[i].r = 0;
		cloud->points[i].g = 0;
	}
	return cloud;
}
class SimpleOpenNIViewer
{
public:
	SimpleOpenNIViewer() : viewer("PCL OpenNI Viewer") {}
	//			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

	void cloud_cb_(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
	{
		if (!viewer.wasStopped())
		{

			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
			if(debug) cout << "new cloud" << endl;
			//processCloud(modelCloud);
			//modelCloud
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*cloud, *modelCloud, indices);
			//savePointCloudsPLY("C:\\Users\\ahmad\\Desktop\\scene\\scene.ply", modelCloud, NULL);
			viewer.showCloud(processCloud(modelCloud));
		}
	}

	void run()
	{
		pcl::Grabber* interface = new pcl::io::OpenNI2Grabber();

		boost::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
			boost::bind(&SimpleOpenNIViewer::cloud_cb_, this, _1);

	

		interface->registerCallback(f);

		interface->start();

		while (!viewer.wasStopped())
		{
			boost::this_thread::sleep(boost::posix_time::seconds(1));
		}

		interface->stop();
	}

	pcl::visualization::CloudViewer viewer;
};


int main(int argc, char** argv)
{
	/*std::string projectSrcDir = PROJECT_SOURCE_DIR;

	string mainModelsPath = projectSrcDir+"/models";
	if(debug) cout << projectSrcDir << endl;

	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>(mainModelsPath+"/bird.ply", *birdCloud) == -1){ PCL_ERROR("Couldn't read file birdCloud.ply \n"); return (-1); }
	std::cout << "Loaded" << birdCloud->width * birdCloud->height << "points" << std::endl;
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>(mainModelsPath + "/can.ply", *canCloud) == -1){ PCL_ERROR("Couldn't read file can.ply \n"); return (-1); }
	std::cout << "Loaded" << canCloud->width * canCloud->height << "points" << std::endl;
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>(mainModelsPath + "/shoe.ply", *shoeCloud) == -1){ PCL_ERROR("Couldn't read file shoe.ply \n"); return (-1); }
	std::cout << "Loaded" << shoeCloud->width * shoeCloud->height << "points" << std::endl;
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>(mainModelsPath + "/house.ply", *houseCloud) == -1){ PCL_ERROR("Couldn't read file house.ply \n"); return (-1); }
	std::cout << "Loaded" << houseCloud->width * houseCloud->height << "points" << std::endl;
	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>(mainModelsPath + "/cracker.ply", *crackerCloud) == -1){ PCL_ERROR("Couldn't read file crackerCloud.ply \n"); return (-1); }
	std::cout << "Loaded" << crackerCloud->width * crackerCloud->height << "points" << std::endl;
	
	pcl::console::print_highlight("Downsampling...\n");
	pcl::VoxelGrid<pcl::PointXYZRGBA> grid;
	const float leaf = 0.005f;
	grid.setLeafSize(leaf, leaf, leaf);


	grid.setInputCloud(birdCloud);
	grid.filter(*birdCloud);



	grid.setInputCloud(houseCloud);
	grid.filter(*houseCloud);


	grid.setInputCloud(canCloud);
	grid.filter(*canCloud);


	grid.setInputCloud(shoeCloud);
	grid.filter(*shoeCloud);



	grid.setInputCloud(crackerCloud);
	grid.filter(*crackerCloud);
	*/




	/*pcl::visualization::PCLVisualizer viewer3("clustered instances");
	//viewer3.addPointCloud(birdCloud, "birdCloud");
	//viewer3.addPointCloud(canCloud, "canCloud");
	//viewer3.addPointCloud(shoeCloud, "shoeCloud");
	viewer3.addPointCloud(houseCloud, "houseCloud");
	//viewer3.addPointCloud(crackerCloud, "crackerCloud");


	
	while (!viewer3.wasStopped())
	{
	viewer3.spinOnce(100);
		

	}
	*/


	SimpleOpenNIViewer v;
	v.run();
	/*pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBA>);

	if (pcl::io::loadPLYFile<pcl::PointXYZRGBA>("C:\\Users\\ahmad\\Desktop\\scene\\scene.ply", *scene) == -1){ PCL_ERROR("Couldn't read file scene.ply \n"); return (-1); }
	std::cout << "Loaded" << scene->width * scene->height << "points" << std::endl;
	
	pcl::visualization::PCLVisualizer viewer3("scene instances");
	viewer3.addPointCloud(processCloud(scene), "scene");
	while (!viewer3.wasStopped())
	{
		viewer3.spinOnce(100);


	}*/
	return (0);
}


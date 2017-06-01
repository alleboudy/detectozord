

#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include <OpenNI.h>
#include <PS1080.h>

/**
* @brief Saving point clouds data as ply file
* @param [in] filename : filename for saving point clouds (the extention should be .ply)
* @param [in] vertices : list of vertex point
* @param [in] colors   : list of vertex color
* @return Success:true, Failure:false
*/
bool savePointCloudsPLY(std::string filename, std::vector<Eigen::Vector4f>& vertices, std::vector<cv::Vec3b>& colors)
{
	std::ofstream fout;
	fout.open(filename.c_str());
	if (fout.fail()){
		std::cerr << "file open error:" << filename << std::endl;
		return false;
	}

	int pointNum = vertices.size();

	fout << "ply" << std::endl;
	fout << "format ascii 1.0" << std::endl;
	fout << "element vertex " << pointNum << std::endl;
	fout << "property float x" << std::endl;
	fout << "property float y" << std::endl;
	fout << "property float z" << std::endl;
	fout << "property uchar red" << std::endl;
	fout << "property uchar green" << std::endl;
	fout << "property uchar blue" << std::endl;
	fout << "property uchar alpha" << std::endl;
	fout << "end_header" << std::endl;

	for (int i = 0; i<pointNum; i++){
		Eigen::Vector4f& vertex = vertices[i];
		cv::Vec3b& col = colors[i];

		fout << vertex[0] << " " << vertex[1] << " " << vertex[2] << " " << static_cast<int>(col[2]) << " " << static_cast<int>(col[1]) << " " << static_cast<int>(col[0]) << " " << 255 << std::endl;
	}

	fout.close();

	return true;
}


// Convert to colored depth image
cv::Mat convColoredDepth(cv::Mat& depthImg, float minThresh = 0, float maxThresh = 0){
    cv::Mat coloredDepth = depthImg.clone();
    
    double min;
    double max;
    if(minThresh == 0 && maxThresh == 0){
        cv::minMaxIdx(depthImg, &min, &max);
    }else{
        min = minThresh;
        max = maxThresh;
    }
    coloredDepth -= min;
    cv::convertScaleAbs(coloredDepth, coloredDepth, 255 / (max-min));
    cv::applyColorMap(coloredDepth, coloredDepth, cv::COLORMAP_JET);
    
    return coloredDepth;
}

int main(int argc, char* argv[])
{
    ////
    // Initialize OpenNI video capture
    
    ////
    // Create depth camera stream
    // Create color camera stream
    
    ////
    // Set flag for synchronization between color camera and depth camera
    // Set flag for registration between color camera and depth camera
    
    ////
    // Get focal length of IR camera in mm for VGA resolution
    // Convert focal length from mm -> pixels (valid for 640x480)
    
    // compute principal points (just image size / 2)

    float minRange = 0;
    float maxRange = 4000;
    
    // Start loop
    while (true)
    {
        ////
        // Grab depth image
        cv::Mat depthImage;
        
        // Grab color image
        cv::Mat colorImage;
        
        
        // Show images
        cv::imshow( "colorImage", colorImage);
        cv::imshow( "depthImage", convColoredDepth(depthImage, minRange, maxRange));
        
        int key = cv::waitKey(10);

        ////
        // Convert the depth iamge to point clouds and save it.
		if (key == 's')
		{
		}
		
        ////
        // Compute standard deviation of depth stream and save it
        if (key == 'd' )
        {
        }
        
        ////
        // Compute mesh of point clouds and save it
        if (key == 'm'){
        }
        
		if (key=='q' || key==27){
            break;
        }
    }

    ////
    // destroy image streams and close the OpenNI device

    return 0;
}




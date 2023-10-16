#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

int main( int argc, char** argv ){
    std::string dataset_dir = "../rgbd_dataset";
    std::ifstream fin(dataset_dir+"/fr1_xyz.txt");
    if(!fin){
        std::cout<<"please generate the associate file called associate.txt!"<<std::endl;
        return 1;
    }

    std::vector<std::string> rgb_files, depth_files;
    std::vector<double> rgb_times, depth_times;
    while(!fin.eof()){
        std::string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir+"/"+rgb_file);
        depth_files.push_back(dataset_dir+"/"+depth_file);

        if(fin.good() == false){
            break;
        }
    }
    fin.close();
    
    std::cout<< "generating features ... " <<std::endl;
    std::vector<cv::Mat> descriptors;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    int index = 1;
    for (std::string& rgb_file : rgb_files){
        cv::Mat image = cv::imread(rgb_file);
        std::vector<cv::KeyPoint> keypoints; 
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        std::cout<<"extracting features from image " << index++ <<std::endl;
    }
    std::cout<<"extract total "<<descriptors.size()*500<<" features."<<std::endl;
    
    // create vocabulary 
    std::cout<<"creating vocabulary, please wait ... "<<std::endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    //std::cout<<"vocabulary info: "<<vocab<<std::endl;
    vocab.save("../tum_rgbd_vocab.yml.gz");
    std::cout<<"done"<<std::endl;
    
    return 0;
}
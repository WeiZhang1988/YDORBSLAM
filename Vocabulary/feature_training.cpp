#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"

int main(int argc, char **argv){
    // read the image
    std::cout << "reading images... " << std::endl;
    std::vector<cv::Mat> images;
    for(int i = 0; i < 10; i++){
    	std::string path = "../data/" + std::to_string(i + 1) + ".png";
    	//std::cout << "image path: " << path << std::endl;
    	//cv::Mat image = cv::imread(path);
    	//cv::imshow("image", image);
    	//cv::waitKey(0);
    	images.push_back(cv::imread(path));
    }
    // detect ORB feature
    std::cout << "detecting ORB features... " << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for(cv::Mat& img : images){
    	std::vector<cv::KeyPoint> keypoints;
    	cv::Mat descriptor;
    	detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
    	descriptors.push_back(descriptor);
    }
    // create vocabulary
    std::cout << "create vocabulary... " << std::endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    std::cout << "vocabulary info: " << std::endl;
    vocab.save("../vocabulary.yml.gz");
    std::cout << "done" << std::endl;
    return 0;
}

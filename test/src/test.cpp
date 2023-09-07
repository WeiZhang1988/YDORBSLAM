#include <iostream>
#include <string>
#include "orbExtractor.hpp"
#include "orbMatcher.hpp"

std::string first_file="./data/img1.png";
std::string second_file="./data/img2.png";

int main(int argc, char **argv) {
  YDORBSLAM::OrbExtractor orbExtractor(10000,1.2,30,20,7);
  cv::Mat first_image = cv::imread(first_file,0);
  cv::Mat second_image = cv::imread(second_file,0);
  std::vector<cv::KeyPoint> features;
  cv::Mat descriptors;
  orbExtractor.extractAndCompute(first_image,features,descriptors); 
  std::cout<<features.size()<<std::endl; 
  //for(auto item : features){
  //  std::cout<<item.pt.x<<"\t"<<item.pt.y<<std::endl;
  //}
  cv::Mat outimg1;
  cv::drawKeypoints(first_image,features,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
  cv::imshow("features",outimg1);   
  cv::waitKey(0); 
  //std::cout<<YDORBSLAM::OrbExtractor::m_v_pattern[0].x<<"\t"<<YDORBSLAM::OrbExtractor::m_v_pattern[0].y<<std::endl;
}
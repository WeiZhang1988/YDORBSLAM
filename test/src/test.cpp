#include <iostream>
#include <string>
#include <memory>
#include "orbExtractor.hpp"
#include "frame.hpp"
#include "keyFrame.hpp"
#include "keyFrameDatabase.hpp"
#include "map.hpp"
#include "mapPoint.hpp"
#include "orbMatcher.hpp"
#include "DBoW3/Vocabulary.h"
using namespace YDORBSLAM;

std::string lfirst_file="./data/cam0/1.png";
std::string lsecond_file="./data/cam0/2.png";
std::string lthird_file="./data/cam0/3.png";

std::string rfirst_file="./data/cam1/1.png";
std::string rsecond_file="./data/cam1/2.png";
std::string rthird_file="./data/cam1/3.png";

cv::Mat SkewSymmetricMatrix(const cv::Mat &v)
{
  return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                                 v.at<float>(2),               0,-v.at<float>(0),
                                -v.at<float>(1),  v.at<float>(0),              0);
}

int main(int argc, char **argv) {
  //KeyPoint extract
  OrbExtractor orbExtractor(1000,1.2,8,20,7);
  cv::Mat lfirst_image = cv::imread(lfirst_file,0);
  cv::Mat lsecond_image = cv::imread(lsecond_file,0);
  std::vector<cv::KeyPoint> lfirst_features, lsecond_features;
  cv::Mat lfirst_descriptors, lsecond_descriptors;
  orbExtractor.extractAndCompute(lfirst_image,lfirst_features,lfirst_descriptors);
  orbExtractor.extractAndCompute(lsecond_image,lsecond_features,lsecond_descriptors);

  cv::Mat outimgl1, outimgl2;
  cv::drawKeypoints(lfirst_image,lfirst_features,outimgl1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
  cv::drawKeypoints(lsecond_image,lsecond_features,outimgl2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);  
  cv::imshow("l1_features",outimgl1);
  cv::imshow("l2_features",outimgl2);
  cv::waitKey(0);
  

  //Create LastFrame & CurrentFrame
  cv::Mat lfirstImage, rfirstImage, lsecondImage, rsecondImage;
  lfirstImage = cv::imread(lfirst_file,cv::IMREAD_UNCHANGED);
  rfirstImage = cv::imread(rfirst_file,cv::IMREAD_UNCHANGED);
  const double firsttimestamp = 1;

  lsecondImage = cv::imread(lsecond_file,cv::IMREAD_UNCHANGED);
  rsecondImage = cv::imread(rsecond_file,cv::IMREAD_UNCHANGED);
  //every frame's timestamp means the video sequence
  const double secondtimestamp = 2;
  cv::Mat camIntParMat = cv::Mat::eye(3,3,CV_32F);
  camIntParMat.at<float>(0,0) = 458.654f; //fx
  camIntParMat.at<float>(1,1) = 457.296f; //fy
  camIntParMat.at<float>(0,2) = 367.215f; //cx
  camIntParMat.at<float>(1,2) = 248.375f; //cy
  cv::Mat imageDistCoef(4,1,CV_32F);
  imageDistCoef.at<float>(0) = 0.0f; //-0.28340811f; //k1
  imageDistCoef.at<float>(1) = 0.0f; //0.07395907f; //k2
  imageDistCoef.at<float>(2) = 0.0f; //0.00019359f; //p1
  imageDistCoef.at<float>(3) = 0.0f; //1.76187114e-05f; //p2
  const float baseLineTimesFx = 40;
  const float depthThd = 40;
  std::shared_ptr<OrbExtractor> mpORBextractorLeft = std::make_shared<OrbExtractor>(1000,1.2,8,20,7);
  std::shared_ptr<OrbExtractor> mpORBextractorRight = std::make_shared<OrbExtractor>(1000,1.2,8,20,7);
  std::shared_ptr<DBoW3::Vocabulary> sptrVocab = std::make_shared<DBoW3::Vocabulary>();

  Frame firstFrame = Frame(lfirstImage,rfirstImage,firsttimestamp,camIntParMat,imageDistCoef,camIntParMat,imageDistCoef,baseLineTimesFx,depthThd,mpORBextractorLeft,mpORBextractorRight,sptrVocab);
  firstFrame.setCameraPoseByTransrom_c2w(cv::Mat::eye(4,4,CV_32F));

  std::shared_ptr<KeyFrameDatabase> sptrKeyFrameDatabase = std::make_shared<KeyFrameDatabase>(*sptrVocab);
  std::shared_ptr<Map> sptrMap = std::make_shared<Map>();

  //Create sptrCurrentKeyFrame
  std::shared_ptr<KeyFrame> sptrfirstKeyFrame = std::make_shared<KeyFrame>(firstFrame,sptrMap,sptrKeyFrameDatabase);
  
  sptrMap->addKeyFrame(sptrfirstKeyFrame);
  //MapPoint Initialization
  for(int i = 0; i < firstFrame.m_int_keyPointsNum; i++){
    if(firstFrame.m_v_depth[i] > 0){
      std::shared_ptr<MapPoint> sptrMP = std::make_shared<MapPoint>(firstFrame.inverseProject(i),sptrMap,sptrfirstKeyFrame);
      sptrMP->addObservation(sptrfirstKeyFrame,i);
      sptrfirstKeyFrame->addMapPoint(sptrMP,i);
      sptrMP->computeDistinctiveDescriptors();
      sptrMP->updateNormalAndDepth();
      sptrMap->addMapPoint(sptrMP);

      firstFrame.m_v_sptrMapPoints[i] = sptrMP;
    }
  }
  std::cout<<"FirstMap's MapPointsNum= "<<sptrMap->getMapPointsNum()<<std::endl;
  cv::waitKey(0);

  //update the LastFrame
  Frame LastFrame = Frame(firstFrame);
  std::shared_ptr<KeyFrame> sptrLastKeyFrame = sptrfirstKeyFrame;

  std::vector<std::shared_ptr<MapPoint>> LocalMapPoints = sptrMap->getAllMapPoints();

  Frame CurrentFrame = Frame(lsecondImage,rsecondImage,secondtimestamp,camIntParMat,imageDistCoef,camIntParMat,imageDistCoef,baseLineTimesFx,depthThd,mpORBextractorLeft,mpORBextractorRight,sptrVocab);
  CurrentFrame.setCameraPoseByTransrom_c2w(cv::Mat::eye(4,4,CV_32F));
  std::shared_ptr<KeyFrame> sptrCurrentKeyFrame = std::make_shared<KeyFrame>(CurrentFrame,sptrMap,sptrKeyFrameDatabase);

  sptrMap->addKeyFrame(sptrCurrentKeyFrame);
  for(int i = 0; i < CurrentFrame.m_int_keyPointsNum; i++){
    if(CurrentFrame.m_v_depth[i] > 0){
      std::shared_ptr<MapPoint> sptrMP = std::make_shared<MapPoint>(CurrentFrame.inverseProject(i),sptrMap,sptrCurrentKeyFrame);
      sptrMP->addObservation(sptrCurrentKeyFrame,i);
      sptrCurrentKeyFrame->addMapPoint(sptrMP,i);
      sptrMP->computeDistinctiveDescriptors();
      sptrMP->updateNormalAndDepth();
      sptrMap->addMapPoint(sptrMP);

      CurrentFrame.m_v_sptrMapPoints[i] = sptrMP;
    }
  }
  std::cout<<"SecondMap's MapPointsNum= "<<sptrMap->getMapPointsNum()<<std::endl;
  cv::waitKey(0);

  OrbMatcher Matched(0.9,true);

  //searchByProjectionInFrameAndMapPoint
  //int Frame_MP_MatchedNUMS = Matched.searchByProjectionInFrameAndMapPoint(CurrentFrame,LocalMapPoints);
  //std::cout<<"Frame_MPNums= "<<Frame_MP_MatchedNUMS<<std::endl;

  //searchByProjectionInLastAndCurrentFrame
  //int Frame_Frame_MatchedNUMS = Matched.searchByProjectionInLastAndCurrentFrame(CurrentFrame,LastFrame);
  //std::cout<<"Frame_FrameNums= "<<Frame_Frame_MatchedNUMS<<std::endl;

  //searchByProjectionInKeyFrameAndCurrentFrame
  /*std::set<std::shared_ptr<MapPoint>> FoundMapPoint;
  int Frame_KF_MatchedNUMS = Matched.searchByProjectionInKeyFrameAndCurrentFrame(CurrentFrame,sptrLastKeyFrame,FoundMapPoint,3,100);
  std::cout<<"Frame_KF_MatchedNUMS= "<<Frame_KF_MatchedNUMS<<std::endl;*/

  //searchByProjectionInSim
  /*cv::Mat sim_c2w = cv::Mat::eye(4,4,CV_32F);
  std::vector<std::shared_ptr<MapPoint>> Matched_MapPoints = sptrCurrentKeyFrame->getMatchedMapPointsVec();
  int KF_MP_MatchedNUMS = Matched.searchByProjectionInSim(sptrCurrentKeyFrame,sim_c2w,LocalMapPoints,Matched_MapPoints,3);
  std::cout<<"KF_MP_MatchedNUMS= "<<KF_MP_MatchedNUMS<<std::endl;*/

  //searchByBowInKeyFrameAndFrame & searchByBowInTwoKeyFrames
  
  //searchForTriangulation
  cv::Mat R1w = sptrCurrentKeyFrame->getRotation_c2w();
  cv::Mat t1w = sptrCurrentKeyFrame->getTranslation_c2w();
  cv::Mat R2w = sptrLastKeyFrame->getRotation_c2w();
  cv::Mat t2w = sptrLastKeyFrame->getTranslation_c2w();
  cv::Mat R12 = R1w*R2w.t();
  cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
  cv::Mat t12x = SkewSymmetricMatrix(t12);
  const cv::Mat &K1 = sptrCurrentKeyFrame->m_cvMat_intParMat;
  const cv::Mat &K2 = sptrLastKeyFrame->m_cvMat_intParMat;
  cv::Mat F12 = K1.t().inv()*t12x*R12*K2.inv();
  std::vector<std::pair<int,int>> vMatchedPairs;
  int KF_KF_MatchedNUMS = Matched.searchForTriangulation(sptrCurrentKeyFrame,sptrLastKeyFrame,F12,vMatchedPairs,true);
  std::cout<<"KF_KF_MatchedNUMS= "<<KF_KF_MatchedNUMS<<std::endl;

  //searchBySim3
  /*std::vector<std::shared_ptr<MapPoint>> vSptrMatched_MapPoints = sptrCurrentKeyFrame->getMatchedMapPointsVec();
  cv::Mat Rotation_firstC2secondC = sptrCurrentKeyFrame->getRotation_c2w() * sptrLastKeyFrame->getRotation_w2c();
  cv::Mat Translation_firstC2secondC = sptrCurrentKeyFrame->getRotation_c2w() * (-sptrLastKeyFrame->getRotation_w2c() * sptrLastKeyFrame->getTranslation_c2w()) + sptrCurrentKeyFrame->getTranslation_c2w();
  int KF_KF_MatchedSim3NUMS = Matched.searchBySim3(sptrCurrentKeyFrame,sptrLastKeyFrame,vSptrMatched_MapPoints,1.0f,Rotation_firstC2secondC,Translation_firstC2secondC,7.5f);
  std::cout<<"KF_KF_MatchedSim3NUMS= "<<KF_KF_MatchedSim3NUMS<<std::endl;*/

  //FuseByProjection
  //int CMP_KF_MatchedNUMS = Matched.FuseByProjection(sptrLastKeyFrame,sptrCurrentKeyFrame->getMatchedMapPointsVec());
  //std::cout<<"CMP_KF_MatchedNUMS= "<<CMP_KF_MatchedNUMS<<std::endl;

  //FuseBySim3
  /*cv::Mat sim3_c2w = cv::Mat::eye(4,4,CV_32F);
  std::vector<std::shared_ptr<MapPoint>> vloopclosingPoint = sptrCurrentKeyFrame->getMatchedMapPointsVec();
  std::vector<std::shared_ptr<MapPoint>> vpReplacePoint(vloopclosingPoint.size(),static_cast<std::shared_ptr<MapPoint>>(NULL));;
  int CMP_KF_MatchedSim3NUMS = Matched.FuseBySim3(sptrLastKeyFrame,sim3_c2w,vloopclosingPoint,4,vpReplacePoint);
  std::cout<<"CMP_KF_MatchedSim3NUMS= "<<CMP_KF_MatchedSim3NUMS<<std::endl;*/
}

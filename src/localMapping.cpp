#include "localMapping.hpp"
#include "orbMatcher.hpp"
#include "optimizer.hpp"
#include <vector>

namespace YDORBSLAM{
  void LocalMapping::run(){
    m_b_isFinished = false;
    while(true){
      //tracking will see that local mapping is busy
      setAcceptKeyFrames(false);
      //check if there are key frames in the queue
      if(checkNewKeyFrames()){
        //BoW conversion and insertion in map
        processNewKeyFrame();
        //check recent map points
        cullMapPoint();
        //triangulate new map points
        createNewMapPoints();
        if(!checkNewKeyFrames()){
          //find more matches in neighbor key frames and fuse point duplications
          searchInNeighbors();
        }
        m_b_isToAbortBA = false;
        if(!checkNewKeyFrames() && !isStopRequested()){
          //local BA
          if(m_sptr_map->getKeyFramesNum()>2){
            Optimizer::localBundleAdjust(m_sptr_currentKeyFrame,m_sptr_map,m_b_isToAbortBA);
          }
          //check redundant local key frames
          cullKeyFrame();
        }
        m_sptr_loopcloser->insertKeyFrame(m_sptr_currentKeyFrame);
      }else if(stop()){
        //safe area to stop
        while(isStopped() && !checkWhetherFinished()){
          usleep(3000);
        }
        if(!checkWhetherFinished()){
          break;
        }
      }
      resetIfRequested();
      //tracking will see that local mapping is busy
      setAcceptKeyFrames(true);
      if(checkWhetherFinished()){
        break;
      }
      usleep(3000);
    }
    setFinished();
  }
  void LocalMapping::insertKeyFrame(std::shared_ptr<KeyFrame> _sptr_keyFrame){
    std::unique_lock<mutex> lock(m_mutex_newKeyFrames);
    m_list_newKeyFrames.push_back(_sptr_keyFrame);
    m_b_isToAbortBA = true;
  }
  bool LocalMapping::checkNewKeyFrames(){
    std::unique_lock<mutex> lock(m_mutex_newKeyFrames);
    return (!m_list_newKeyFrames.empty());
  }
  void LocalMapping::processNewKeyFrame(){
    {
      std::unique_lock<mutex> lock(m_mutex_newKeyFrames);
      m_sptr_currentKeyFrame = m_list_newKeyFrames.front();
      m_list_newKeyFrames.pop_front();
    }
    //compute bag of words
    m_sptr_currentKeyFrame->computeBoW();
    //associate map points to the new key frame and update normal and descriptors
    const std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPoints = m_sptr_currentKeyFrame->getMatchedMapPointsVec();
    for(int i=0;i<vSptrMatchedMapPoints.size();i++){
      if(vSptrMatchedMapPoints[i] && !vSptrMatchedMapPoints[i]->isBad()){
        if(vSptrMatchedMapPoints[i]->isInKeyFrame(m_sptr_currentKeyFrame)){
          //this can only happen for new stereo points inserted by the tracking
          m_list_recentAddedMapPoints.push_back(vSptrMatchedMapPoints[i]);
        }else{
          vSptrMatchedMapPoints[i]->addObservation(m_sptr_currentKeyFrame, i);
          vSptrMatchedMapPoints[i]->updateNormalAndDepth();
          vSptrMatchedMapPoints[i]->computeDistinctiveDescriptors();
        }
      }
    }
    //update links in the covisibility graph
    m_sptr_currentKeyFrame->updateConnections();
    //insert key frame in map
    m_sptr_map->addKeyFrame(m_sptr_currentKeyFrame);
  }
  void LocalMapping::cullMapPoint(){
    //check recent added map points
    std::list<std::shared_ptr<MapPoint>>::iterator listIter = m_list_recentAddedMapPoints.begin();
    const long int currentKeyFrameID = m_sptr_currentKeyFrame->m_int_keyFrameID;
    while(listIter!=m_list_recentAddedMapPoints.end()){
      if((*listIter) && \
      ((*listIter)->isBad() || \
      ((int)currentKeyFrameID - (int)(*listIter)->m_int_firstKeyFrameID)>=3)){
        listIter = m_list_recentAddedMapPoints.erase(listIter);
      }else if((*listIter) && \
      ((*listIter)->getFoundRatio()<0.25f || \
      ((int)currentKeyFrameID - (int)(*listIter)->m_int_firstKeyFrameID)>=2 && (*listIter)->getObservationsNum()<=3)){
        (*listIter)->setBadFlag();
        listIter = m_list_recentAddedMapPoints.erase(listIter);
      }else {
        listIter++;
      }
    }
  }
  void LocalMapping::createNewMapPoints(){
    //retrieve neighbor key frames in covisibility graph
    const std::vector<std::shared_ptr<KeyFrame>> vSptrConnectedKeyFrames = m_sptr_currentKeyFrame->getFirstNumOrderedConnectedKeyFrames(10);
    OrbMatcher matcher(0.6,false);
    cv::Mat currentKeyFrameRotation_c2w = m_sptr_currentKeyFrame->getRotation_c2w();
    cv::Mat currentKeyFrameRotation_w2c = m_sptr_currentKeyFrame->getRotation_w2c();
    cv::Mat currentKeyFrameRotation_c2w = m_sptr_currentKeyFrame->getTranslation_c2w();
    cv::Mat currentKeyFrameTransformation_c2w(3,4,CV_32F);
    currentKeyFrameRotation_c2w.copyTo(currentKeyFrameTransformation_c2w.colRange(0,3));
    currentKeyFrameRotation_c2w.copyTo(currentKeyFrameTransformation_c2w.col(3));
    cv::Mat currentKeyFrameCameraInWorld = m_sptr_currentKeyFrame->getCameraOriginInWorld();
    const float& currentKeyFrame_fx     = m_sptr_currentKeyFrame->m_flt_fx;
    const float& currentKeyFrame_fy     = m_sptr_currentKeyFrame->m_flt_fy;
    const float& currentKeyFrame_cx     = m_sptr_currentKeyFrame->m_flt_cx;
    const float& currentKeyFrame_cy     = m_sptr_currentKeyFrame->m_flt_cy;
    const float& currentKeyFrame_invFx  = m_sptr_currentKeyFrame->m_flt_invFx;
    const float& currentKeyFrame_invFy  = m_sptr_currentKeyFrame->m_flt_invFy;
    const float ratioFactor = 1.5f * m_sptr_currentKeyFrame->m_flt_scaleFactor;
    //search matches with epipolar restriction and triangulate
    for(const std::shared_ptr<KeyFrame> &connectedKeyFrame : vSptrConnectedKeyFrames){
      if(connectedKeyFrame==vSptrConnectedKeyFrames.front() || !checkNewKeyFrames()){
        //firstly, check baseline is not too short
        cv::Mat connectedKeyFrameCameraInWorld = connectedKeyFrame->getCameraOriginInWorld();
        const float baseLine = cv::norm(connectedKeyFrameCameraInWorld - currentKeyFrameCameraInWorld);
        if(baseLine < connectedKeyFrame->m_flt_baseLine){
          continue;
        }else{
          //compute fundamental matrix
          cv::Mat fundamentalMatrix_cur2cov = computeFundamentalMatrix_first2second(m_sptr_currentKeyFrame,connectedKeyFrame);
          //search matches that fullfil epipolar constraint
          std::vector<std::pair<int,int>> vMatchedIndicesPairs;
          matcher.searchForTriangulation(m_sptr_currentKeyFrame,connectedKeyFrame,fundamentalMatrix_cur2cov,vMatchedIndicesPairs,false);
          cv::Mat connectedKeyFrameRotation_c2w = connectedKeyFrame->getRotation_c2w();
          cv::Mat connectedKeyFrameRotation_w2c = connectedKeyFrame->getRotation_w2c();
          cv::Mat connectedKeyFrameTranslation_c2w = connectedKeyFrame->getTranslation_c2w();
          cv::Mat connectedKeyFrameTransformation(3,4,CV_32F);
          connectedKeyFrameRotation_c2w.copyTo(connectedKeyFrameTransformation.colRange(0,3));
          connectedKeyFrameTranslation_c2w.copyTo(connectedKeyFrameTransformation.col(3));
          const float& connectedKeyFrame_fx     = connectedKeyFrame->m_flt_fx;
          const float& connectedKeyFrame_fy     = connectedKeyFrame->m_flt_fy;
          const float& connectedKeyFrame_cx     = connectedKeyFrame->m_flt_cx;
          const float& connectedKeyFrame_cy     = connectedKeyFrame->m_flt_cy;
          const float& connectedKeyFrame_invFx  = connectedKeyFrame->m_flt_invFx;
          const float& connectedKeyFrame_invFy  = connectedKeyFrame->m_flt_invFy;
          //triangulate each match
          for(const std::pair<int,int> &matchedIndicesPair : vMatchedIndicesPairs){
            const cv::KeyPoint &currentKeyFrameKeyPoint = m_sptr_currentKeyFrame->m_v_keyPoints[matchedIndicesPair.first];
            const float currentKeyFrameKeyPointRightXcords = m_sptr_currentKeyFrame->m_v_rightXcords[matchedIndicesPair.first];
            const cv::KeyPoint &connectedKeyFrameKeyPoint = connectedKeyFrame->m_v_keyPoints[matchedIndicesPair.second];
            const float connectedKeyFrameKeyPointRightXcords = connectedKeyFrame->m_v_rightXcords[matchedIndicesPair.second];
            //check parallax between rays
            cv::Mat currentKeyPointDirection = (cv::Mat_<float>(3,1) << (connectedKeyFrameKeyPoint.pt.x-currentKeyFrame_cx)*currentKeyFrame_invFx, (connectedKeyFrameKeyPoint.pt.y-currentKeyFrame_cy)*currentKeyFrame_invFy, 1.0);
            cv::Mat connectedKeyPointDirection = (cv::Mat_<float>(3,1) << (connectedKeyFrame.pt.x-connectedKeyFrame_cx)*connectedKeyFrame_invFx, (connectedKeyFrame.pt.y-connectedKeyFrame_cy)*connectedKeyFrame_invFy, 1.0);
            cv::Mat currentRayDirection = currentKeyFrameRotation_w2c * currentKeyPointDirection;
            cv::Mat connectedRayDirection = connectedKeyFrameRotation_w2c * connectedKeyPointDirection;
            const float cosParallaxRays = currentRayDirection.dot(connectedRayDirection) / (cv::norm(currentRayDirection) * cv::norm(connectedRayDirection));
            float cosParallaxStereo = cosParallaxRays+1;  //+1 is to make sure cosParallaxStereo initialized to be large
            float cosParallaxStereo1 = cos(2*atan2(m_sptr_currentKeyFrame->m_flt_baseLine/2,m_sptr_currentKeyFrame->m_v_depth[matchedIndicesPair.first]));
            float cosParallaxStereo2 = cosParallaxStereo;
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
            cv::Mat inverseProject3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0){
              //linear triangulation method
              cv::Mat A(4,4,CV_32F);
              A.row(0) = currentKeyPointDirection.at<float>(0)*currentKeyFrameTransformation_c2w.row(2)-currentKeyFrameTransformation_c2w.row(0);
              A.row(1) = currentKeyPointDirection.at<float>(1)*currentKeyFrameTransformation_c2w.row(2)-currentKeyFrameTransformation_c2w.row(1);
              A.row(2) = connectedKeyPointDirection.at<float>(0)*connectedKeyFrameTransformation_c2w.row(2)-connectedKeyFrameTransformation_c2w.row(0);
              A.row(3) = connectedKeyPointDirection.at<float>(1)*connectedKeyFrameTransformation_c2w.row(2)-connectedKeyFrameTransformation_c2w.row(1);
              cv::Mat U,W,Vt; //A = u * w * vt
              cv::SVD::compute(A,W,U,Vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
              inverseProject3D = Vt.row(3).t();
              if(inverseProject3D.at<float>(3) == 0){
                continue;
              }else{
                // Euclidean coordinates
                inverseProject3D = inverseProject3D.rowRange(0,3)/inverseProject3D.at<float>(3);
              }
            }else if(cosParallaxStereo1<cosParallaxStereo2){
              inverseProject3D = m_sptr_currentKeyFrame->inverseProject(matchedIndicesPair.first);
            }else if(cosParallaxStereo1>cosParallaxStereo2){
              inverseProject3D = connectedKeyFrame->inverseProject(matchedIndicesPair.second);
            }else {  //there should be a branch to deal with the case when cosParallaxStereo1==cosParallaxStereo2
              continue;
            }
            //check triangulation in front of cameras
            //stop here
          }
        }
      }else{
        return;
      }
    }
  }
}//namespace YDORBSLAM
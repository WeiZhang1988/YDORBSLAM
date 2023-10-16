#include "localMapping.hpp"
#include "orbMatcher.hpp"
#include "optimizer.hpp"
#include <vector>
#include <cmath>

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
    cv::Mat currentKeyFrameTranslation_c2w = m_sptr_currentKeyFrame->getTranslation_c2w();
    cv::Mat currentKeyFrameTransformation_c2w(3,4,CV_32F);
    currentKeyFrameRotation_c2w.copyTo(currentKeyFrameTransformation_c2w.colRange(0,3));
    currentKeyFrameTranslation_c2w.copyTo(currentKeyFrameTransformation_c2w.col(3));
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
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            if(currentKeyFrameKeyPointRightXcords>=0){
              cosParallaxStereo1 = cos(2*atan2(m_sptr_currentKeyFrame->m_flt_baseLine/2,m_sptr_currentKeyFrame->m_v_depth[matchedIndicesPair.first]));
            }else if(connectedKeyFrameKeyPointRightXcords>=0){
              cosParallaxStereo2 = cosParallaxStereo;
            }
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
            cv::Mat inverseProject3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (currentKeyFrameKeyPointRightXcords>=0 || connectedKeyFrameKeyPointRightXcords>=0 || cosParallaxRays<0.9998)){
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
            }else if(currentKeyFrameKeyPointRightXcords>=0 && cosParallaxStereo1<cosParallaxStereo2){
              inverseProject3D = m_sptr_currentKeyFrame->inverseProject(matchedIndicesPair.first);
            }else if(connectedKeyFrameKeyPointRightXcords>=0 && cosParallaxStereo1>cosParallaxStereo2){
              inverseProject3D = connectedKeyFrame->inverseProject(matchedIndicesPair.second);
            }else {  //there should be a branch to deal with the case when cosParallaxStereo1==cosParallaxStereo2
              continue;
            }
            //check triangulation in front of cameras
            float currentKeyFrameZcord = currentKeyFrameRotation_c2w.row(2).dot(inverseProject3D.t()) + currentKeyFrameTranslation_c2w.at<float>(2);
            float connectedKeyFrameZcord = connectedKeyFrameRotation_c2w.row(2).dot(inverseProject3D.t()) + connectedKeyFrameTranslation_c2w.at<float>(2);
            if(currentKeyFrameZcord>0 && connectedKeyFrameZcord>0){
              //check reprojection error in current key frame
              const float &currentSquaredSigma = m_sptr_currentKeyFrame->m_v_scaleFactorSquares[currentKeyFrameKeyPoint.octave];
              const float currentKeyFrameXcord = currentKeyFrameRotation_c2w.row(0).dot(inverseProject3D.t()) + currentKeyFrameTranslation_c2w.at<float>(0);
              const float currentKeyFrameYcord = currentKeyFrameRotation_c2w.row(1).dot(inverseProject3D.t()) + currentKeyFrameTranslation_c2w.at<float>(1);
              if(currentKeyFrameKeyPointRightXcords<0 && \
              (currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x) * (currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x) + \
              (currentKeyFrame_fy * currentKeyFrameYcord / currentKeyFrameZcord + currentKeyFrame_cy - currentKeyFrameKeyPoint.pt.y) * (currentKeyFrame_fy * currentKeyFrameYcord / currentKeyFrameZcord + currentKeyFrame_cy - currentKeyFrameKeyPoint.pt.y) > 5.991 * currentSquaredSigma){
                continue;
              }else if((currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x) * (currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x) + \
              (currentKeyFrame_fy * currentKeyFrameYcord / currentKeyFrameZcord + currentKeyFrame_cy - currentKeyFrameKeyPoint.pt.y) * (currentKeyFrame_fy * currentKeyFrameYcord / currentKeyFrameZcord + currentKeyFrame_cy - currentKeyFrameKeyPoint.pt.y) + \
              (currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x - m_sptr_currentKeyFrame->m_flt_baseLineTimesFx / currentKeyFrameZcord - currentKeyFrameKeyPointRightXcords) * (currentKeyFrame_fx * currentKeyFrameXcord / currentKeyFrameZcord + currentKeyFrame_cx - currentKeyFrameKeyPoint.pt.x - m_sptr_currentKeyFrame->m_flt_baseLineTimesFx / currentKeyFrameZcord - currentKeyFrameKeyPointRightXcords) > 7.8 * currentSquaredSigma){
                continue;
              }
              //check reprojection error in connectedSquaredSigma key frame
              const float &connectedSquaredSigma = connectedKeyFrame->m_v_scaleFactorSquares[connectedKeyFrameKeyPoint.octave];
              const float connectedKeyFrameXcord = connectedKeyFrameRotation_c2w.row(0).dot(inverseProject3D.t()) + connectedKeyFrameTranslation_c2w.at<float>(0);
              const float connectedKeyFrameYcord = connectedKeyFrameRotation_c2w.row(1).dot(inverseProject3D.t()) + connectedKeyFrameTranslation_c2w.at<float>(1);
              if(connectedKeyFrameKeyPointRightXcords<0 && \
              (connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x) * (connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x) + \
              (connectedKeyFrame_fy * connectedKeyFrameYcord / connectedKeyFrameZcord + connectedKeyFrame_cy - connectedKeyFrameKeyPoint.pt.y) * (connectedKeyFrame_fy * connectedKeyFrameYcord / connectedKeyFrameZcord + connectedKeyFrame_cy - connectedKeyFrameKeyPoint.pt.y) > 5.991 * connectedSquaredSigma){
                continue;
              }else if((connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x) * (connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x) + \
              (connectedKeyFrame_fy * connectedKeyFrameYcord / connectedKeyFrameZcord + connectedKeyFrame_cy - connectedKeyFrameKeyPoint.pt.y) * (connectedKeyFrame_fy * connectedKeyFrameYcord / connectedKeyFrameZcord + connectedKeyFrame_cy - connectedKeyFrameKeyPoint.pt.y) + \
              (connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x - connectedKeyFrame->m_flt_baseLineTimesFx / connectedKeyFrameZcord - connectedKeyFrameKeyPointRightXcords) * (connectedKeyFrame_fx * connectedKeyFrameXcord / connectedKeyFrameZcord + connectedKeyFrame_cx - connectedKeyFrameKeyPoint.pt.x - connectedKeyFrame->m_flt_baseLineTimesFx / connectedKeyFrameZcord - connectedKeyFrameKeyPointRightXcords) > 7.8 * connectedSquaredSigma){
                continue;
              }
              //check scale consistency
              const float currentKeyFrameDist = cv::norm(inverseProject3D-currentKeyFrameCameraInWorld);
              const float connectedKeyFrameDist = cv::norm(inverseProject3D-connectedKeyFrameCameraInWorld);
              const float ratioDist = connectedKeyFrameDist/currentKeyFrameDist;
              const float ratioOctave = m_sptr_currentKeyFrame->m_v_scaleFactors[currentKeyFrameKeyPoint.octave]/connectedKeyFrame->m_v_scaleFactors[connectedKeyFrameKeyPoint.octave];
              if(currentKeyFrameDist!=0 && connectedKeyFrameDist!=0 && ratioDist*ratioFactor>=ratioOctave && ratioDist<=ratioOctave*ratioFactor){
                //triangulation succeeds
                std::shared_ptr<MapPoint> sptrMapPoint = std::make_shared<MapPoint>(inverseProject3D,m_sptr_currentKeyFrame,sptrMapPoint);
                sptrMapPoint->addObservation(m_sptr_currentKeyFrame, matchedIndicesPair.first);
                sptrMapPoint->addObservation(connectedKeyFrame, matchedIndicesPair.second);
                m_sptr_currentKeyFrame->addMapPoint(sptrMapPoint, matchedIndicesPair.first);
                connectedKeyFrame->addMapPoint(sptrMapPoint, matchedIndicesPair.second);
                sptrMapPoint->computeDistinctiveDescriptors();
                sptrMapPoint->updateNormalAndDepth();
                m_sptr_map->addMapPoint(sptrMapPoint);
                m_list_recentAddedMapPoints.push_back(sptrMapPoint);
              }
            }
          }
        }
      }else{
        return;
      }
    }
  }
  void LocalMapping::searchInNeighbors(){
    //retrieve neighbor key frames
    std::vector<std::shared_ptr<KeyFrame>> vSptrTargetKeyFrames;
    for(const std::shared_ptr<KeyFrame> &connectedKeyFrame : m_sptr_currentKeyFrame->getFirstNumOrderedConnectedKeyFrames(10)){
      if(connectedKeyFrame && !connectedKeyFrame->isBad() && connectedKeyFrame->m_int_fuseTargetForKeyFrameID!=m_sptr_currentKeyFrame->m_int_keyFrameID){
        vSptrTargetKeyFrames.push_back(connectedKeyFrame);
        connectedKeyFrame->m_int_fuseTargetForKeyFrameID = m_sptr_currentKeyFrame->m_int_keyFrameID;
        //extend to some indirect neighbors
        for(const std::shared_ptr<KeyFrame> &inDirectedConnectedKeyFrame : connectedKeyFrame->getFirstNumOrderedConnectedKeyFrames(5)){
          if(inDirectedConnectedKeyFrame && !inDirectedConnectedKeyFrame->isBad() && inDirectedConnectedKeyFrame->m_int_fuseTargetForKeyFrameID!=m_sptr_currentKeyFrame->m_int_keyFrameID && inDirectedConnectedKeyFrame->m_int_keyFrameID!=m_sptr_currentKeyFrame->m_int_keyFrameID){
            vSptrTargetKeyFrames.push_back(inDirectedConnectedKeyFrame);
          }
        }
      }
    }
    //search matches by projection from current key frame in target key frames
    OrbMatcher matcher;
    const std::vector<std::shared_ptr<MapPoint>> &vSptrCurrentMatchedMapPoints = m_sptr_currentKeyFrame->getMatchedMapPointsVec();
    for(std::shared_ptr<KeyFrame> &targetKeyFrame : vSptrTargetKeyFrames){
      matcher.FuseByProjection(targetKeyFrame,vSptrCurrentMatchedMapPoints);
    }
    //search matches by projection from target key frames in current key frames
    std::vector<std::shared_ptr<MapPoint>> vSptrFuseCandidateMapPoints;
    vSptrFuseCandidateMapPoints.reserve(vSptrTargetKeyFrames.size() * vSptrCurrentMatchedMapPoints.size());
    for(std::shared_ptr<KeyFrame> &targetKeyFrame : vSptrTargetKeyFrames){
      for(std::shared_ptr<MapPoint> &targetMapPoint : targetKeyFrame->getMatchedMapPointsVec()){
        if(targetMapPoint && !targetMapPoint->isBad() && targetMapPoint->m_int_fuseCandidateForKeyFrameID!=m_sptr_currentKeyFrame->m_int_keyFrameID){
          targetMapPoint->m_int_fuseCandidateForKeyFrameID = m_sptr_currentKeyFrame->m_int_keyFrameID;
          vSptrFuseCandidateMapPoints.push_back(targetMapPoint);
        }
      }
    }
    matcher.FuseByProjection(m_sptr_currentKeyFrame,vSptrFuseCandidateMapPoints);
    //update points
    for(std::shared_ptr<MapPoint> &updateCurrentMapPoint : m_sptr_currentKeyFrame->getMatchedMapPointsVec()){
      if(updateCurrentMapPoint && !updateCurrentMapPoint->isBad()){
        updateCurrentMapPoint->computeDistinctiveDescriptors();
        updateCurrentMapPoint->updateNormalAndDepth();
      }
    }
    m_sptr_currentKeyFrame->updateConnections();
  }
  cv::Mat LocalMapping::computeFundamentalMatrix_first2second(std::shared_ptr<KeyFrame> _sptr_firstKeyFrame,std::shared_ptr<KeyFrame> _sptr_secondKeyFrame){
    cv::Mat firstRotation_c2w = _sptr_firstKeyFrame->getRotation_c2w();
    cv::Mat firstTranslation_c2w = _sptr_firstKeyFrame->getTranslation_c2w();
    cv::Mat secondRotation_c2w = _sptr_secondKeyFrame->getRotation_c2w();
    cv::Mat secondTranslation_c2w = _sptr_secondKeyFrame->getTranslation_c2w();
    cv::Mat rotation_first2second = firstRotation_c2w * secondRotation_c2w.t();
    cv::Mat translation_first2second = -firstRotation_c2w * secondRotation_c2w.t() * secondTranslation_c2w + firstTranslation_c2w;
    cv::Mat skewSymmetric_first2second = skewSymmetricMatrix(translation_first2second);
    const cv::Mat &K1 = _sptr_firstKeyFrame->m_cvMat_intParMat;
    const cv::Mat &K2 = _sptr_secondKeyFrame->m_cvMat_intParMat;
    return K1.t().inv() * skewSymmetric_first2second * rotation_first2second * K2.inv();
  }
  void LocalMapping::requestStop(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    m_b_isStopRequested = true;
    std::unique_lock<std::mutex> lock2(m_mutex_newKeyFrames);
    m_b_isToAbortBA = true;
  }
  bool LocalMapping::stop(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    if(m_b_isStopRequested && !m_b_isNotStoped)
    {
      m_b_isStopped = true;
      std::cout << "Local Mapping STOP" << std::endl;
      return true;
    }
    return false;
  }
  bool LocalMapping::isStopped(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    return m_b_isStopped;
  }
  //stop here
}//namespace YDORBSLAM
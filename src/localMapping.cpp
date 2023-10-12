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
      //stop here
    }
  }
}//namespace YDORBSLAM
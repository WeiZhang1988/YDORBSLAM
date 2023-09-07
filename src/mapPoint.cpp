#include <climits>
#include <cmath>
#include "mapPoint.hpp"
#include "orbMatcher.hpp"

namespace YDORBSLAM{
  long int MapPoint::m_int_reservedID=0;
  std::mutex MapPoint::m_mutex_global;
  std::mutex MapPoint::m_mutex_ID;
  MapPoint::MapPoint(const cv::Mat &_posInWorld, std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrame> _sptrRefKeyFrame):
  m_sptr_map(_sptrMap),m_sptr_refKeyFrame(_sptrRefKeyFrame){
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      _posInWorld.copyTo(m_cvMat_posInWorld);
    }
    //key frame has its own member variable keyFrameID which does not belong to ordinary frame
    m_int_firstKeyFrameID = m_sptr_refKeyFrame->m_int_keyFrameID;
    //key frame keeps its ID variable from ordinary frame
    m_int_firstFrameID = m_sptr_refKeyFrame->m_int_ID;
    m_cvMat_normalVector = cv::Mat::zeros(3,1,CV_32F);
    {
      std::unique_lock<std::mutex> lock(m_mutex_ID);
      m_int_ID = m_int_reservedID++;
    }
  }
  MapPoint::MapPoint(const cv::Mat &_posInWorld, std::shared_ptr<Map> _sptrMap, Frame &_frame, const int &_idxInKeyPoints):
  m_sptr_map(_sptrMap){
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      _posInWorld.copyTo(m_cvMat_posInWorld);
    }
    m_int_firstKeyFrameID = -1;
    m_int_firstFrameID = _frame.m_int_ID;
    cv::Mat cameraOriginInWorld = _frame.getCameraOriginInWorld();
    cv::Mat vectorCameraPos = m_cvMat_posInWorld - cameraOriginInWorld;
    m_cvMat_normalVector = (vectorCameraPos)/cv::norm(vectorCameraPos);
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      m_flt_maxDistance = cv::norm(vectorCameraPos) * _frame.m_v_scaleFactors[_frame.m_v_keyPoints[_idxInKeyPoints].octave];
      m_flt_minDistance = m_flt_maxDistance / _frame.m_v_scaleFactors.back();
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_descriptor);
      _frame.m_cvMat_descriptors.row(_idxInKeyPoints).copyTo(m_cvMat_descriptor);
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_ID);
      m_int_ID = m_int_reservedID++;
    } 
  }
  void MapPoint::setPosInWorld(const cv::Mat &_posInWorld){
    {
      std::unique_lock<std::mutex> lock_global(m_mutex_global);
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      _posInWorld.copyTo(m_cvMat_posInWorld);
    }
  }
  void MapPoint::setBadFlag(){
    std::map<std::shared_ptr<KeyFrame>,int> dicObservations;
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      m_b_bad = true;
      dicObservations = m_dic_observations;
      m_dic_observations.clear();
    }
    //this for loop is time consuming //not sure const inside pair or outside
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : dicObservations){
      observation.first->eraseMatchedMapPoint(observation.second);
    }
    m_sptr_map->eraseMapPoint(shared_from_this());
  }
  int MapPoint::getIdxInKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      if(m_dic_observations.count(_sptrKeyFrame)){
        return m_dic_observations[_sptrKeyFrame];
      }else{
        return -1;
      }
    }
  }
  bool MapPoint::isInKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return (bool)(m_dic_observations.count(_sptrKeyFrame));
    }
  }
  void MapPoint::addObservation(std::shared_ptr<KeyFrame> _sptrKeyFrame,const int &_idx){
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      if(m_dic_observations.count(_sptrKeyFrame)){
        return;
      }else {
        m_dic_observations[_sptrKeyFrame] = _idx;
      }
      if(_sptrKeyFrame->m_v_rightXcords[_idx]>=0){
        m_int_observationsNum += 2;
      }else{
        m_int_observationsNum++;
      }
    }
  }
  void MapPoint::eraseObservation(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    bool bToSetBad = false;
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      if(m_dic_observations.count(_sptrKeyFrame)){
        int idx = m_dic_observations[_sptrKeyFrame];
        if(_sptrKeyFrame->m_v_rightXcords[idx]>=0){
          m_int_observationsNum -= 2;
        }else{
          m_int_observationsNum--;
        }
        m_dic_observations.erase(_sptrKeyFrame);
        if(m_sptr_refKeyFrame==_sptrKeyFrame){
          m_sptr_refKeyFrame = m_dic_observations.begin()->first;
        }
        if(m_int_observationsNum<=2){
          bToSetBad = true;
        }
      }
    }
    //put setBadFlag() seperately to avoid dead lock
    if(bToSetBad){
      setBadFlag();
    }
  }
  void MapPoint::beReplacedBy(std::shared_ptr<MapPoint> _sptrMapPoint){
    if(_sptrMapPoint->m_int_ID == m_int_ID){
      return;
    }
    int visibleNum, foundNum;
    std::map<std::shared_ptr<KeyFrame>,int> dicObservations;
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      m_b_bad = true;
      dicObservations = m_dic_observations;
      m_dic_observations.clear();
      visibleNum = m_int_visibleNum;
      foundNum = m_int_foundNum;
      m_sptr_replacement = _sptrMapPoint;
    }
    //the following operations are time consuming
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : dicObservations){
      if(!_sptrMapPoint->isInKeyFrame(observation.first)){
        observation.first->replaceMapPointMatch(_sptrMapPoint,observation.second);
        _sptrMapPoint->addObservation(observation.first,observation.second);
      }else{
        observation.first->eraseMatchedMapPoint(observation.second);
      }
    }
    _sptrMapPoint->increaseFound(foundNum);
    _sptrMapPoint->increaseVisible(visibleNum);
    _sptrMapPoint->computeDistinctiveDescriptors();
    m_sptr_map->eraseMapPoint(shared_from_this());
  }
  void MapPoint::increaseVisible(const int &_num){
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      m_int_visibleNum += _num;
    }
  }
  void MapPoint::increaseFound(const int &_num){
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      m_int_foundNum += _num;
    }
  }
  void MapPoint::computeDistinctiveDescriptors(){
    std::vector<cv::Mat> vDescriptors;
    std::map<std::shared_ptr<KeyFrame>,int> dicObservations;
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      if(m_b_bad){
        return;
      }
      dicObservations = m_dic_observations;
    }
    if(dicObservations.empty()){
      return;
    }
    vDescriptors.reserve(dicObservations.size());
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : dicObservations){
      if(!observation.first->isBad()){
        vDescriptors.push_back(observation.first->m_cvMat_descriptors.row(observation.second));
      }
    }
    if(vDescriptors.empty()){
      return;
    }
    //compute distances between descriptors
    const int vDescriptorsSize = vDescriptors.size();
    float distances[vDescriptorsSize][vDescriptorsSize];
    for(int i=0;i<vDescriptorsSize;i++){
      distances[i][i]=0;
      for(int j=i+1;j<vDescriptorsSize;j++){
        int distanceIJ = OrbMatcher::computeDescriptorsDistance(vDescriptors[i],vDescriptors[j]);
        distances[i][j] = distanceIJ;
        distances[j][i] = distanceIJ;
      }
    }
    //find the descriptor with least median distance to others
    int bestMedian = INT_MAX;
    int bestMedianIdx = 0;
    for(int i=0;i<vDescriptorsSize;i++){
      std::vector<int> vDistances(distances[i],distances[i]+vDescriptorsSize);
      std::sort(vDistances.begin(),vDistances.end());
      int median = vDistances[0.5*(vDescriptorsSize)];
      if(median<bestMedian){
        bestMedian = median;
        bestMedianIdx = i;
      }
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_descriptor);
      m_cvMat_descriptor = vDescriptors[bestMedianIdx].clone();
    }
  }
  void MapPoint::updateNormalAndDepth(){
    std::map<std::shared_ptr<KeyFrame>,int> dicObservations;
    std::shared_ptr<KeyFrame> sptrRefKeyFrame;
    cv::Mat posInWorld;
    {
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      std::unique_lock<std::mutex> lock1(m_mutex_PosDistNorm);
      if(m_b_bad){
        return;
      }else{
        dicObservations = m_dic_observations;
        sptrRefKeyFrame = m_sptr_refKeyFrame;
        posInWorld = m_cvMat_posInWorld.clone();
      }
    }
    if(dicObservations.empty()){
      return;
    }
    cv::Mat normalVector = cv::Mat::zeros(3,1,CV_32F);
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : dicObservations){
      cv::Mat cameraOriginInWorld = observation.first->getCameraOriginInWorld();
      cv::Mat vectorCameraPos = posInWorld - cameraOriginInWorld;
      normalVector += (vectorCameraPos/cv::norm(vectorCameraPos));
    }
    cv::Mat vectorCameraPos = posInWorld - sptrRefKeyFrame->getCameraOriginInWorld();
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      m_flt_maxDistance = cv::norm(vectorCameraPos)*sptrRefKeyFrame->m_v_scaleFactors[sptrRefKeyFrame->m_v_keyPoints[dicObservations[sptrRefKeyFrame]].octave];
      m_flt_minDistance = m_flt_maxDistance/sptrRefKeyFrame->m_v_scaleFactors.back();
      m_cvMat_normalVector = normalVector/(float)dicObservations.size();
    }
  }
  int MapPoint::predictScaleLevel(const float &_currentDist, std::shared_ptr<KeyFrame> _sptrKeyFrame){
    float ratio;
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      ratio = m_flt_maxDistance/_currentDist;
    }
    int scaleLevel = ceil(log(ratio)/_sptrKeyFrame->m_flt_logScaleFactor);
    if(scaleLevel<0){
      scaleLevel = 0;
    }else if(scaleLevel>=_sptrKeyFrame->m_int_scaleLevelsNum){
      scaleLevel = _sptrKeyFrame->m_int_scaleLevelsNum - 1;
    }
    return scaleLevel;
  }
  int MapPoint::predictScaleLevel(const float &_currentDist, const Frame &_frame){
    float ratio;
    {
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      ratio = m_flt_maxDistance/_currentDist;
    }
    int scaleLevel = ceil(log(ratio)/_frame.m_flt_logScaleFactor);
    if(scaleLevel<0){
      scaleLevel = 0;
    }else if(scaleLevel>=_frame.m_int_scaleLevelsNum){
      scaleLevel = _frame.m_int_scaleLevelsNum - 1;
    }
    return scaleLevel;
  }
}
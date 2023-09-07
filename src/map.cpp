#include "map.hpp"

namespace YDORBSLAM {
  void Map::addKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_set_sptrKeyFrames.insert(_sptrKeyFrame);
    if(_sptrKeyFrame->m_int_keyFrameID>m_int_maxKeyFrameID){
      m_int_maxKeyFrameID = _sptrKeyFrame->m_int_keyFrameID;
    }
  }
  void Map::addMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_set_sptrMapPoints.insert(_sptrMapPoint);
  }
  void Map::eraseKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_set_sptrKeyFrames.erase(_sptrKeyFrame);
  }
  void Map::eraseMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_set_sptrMapPoints.erase(_sptrMapPoint);
  }
  void Map::setReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_v_sptrReferenceMapPoints = _vSptrMapPoints;
  }
  void Map::informNewBigChange(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    m_int_bigChangeIdx++;
  }
  int Map::getLastBigChangeIdx(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return m_int_bigChangeIdx;
  }
  std::vector<std::shared_ptr<KeyFrame>> Map::getAllKeyFrames(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return std::vector<std::shared_ptr<KeyFrame>>(m_set_sptrKeyFrames.begin(),m_set_sptrKeyFrames.end());
  }
  std::vector<std::shared_ptr<MapPoint>> Map::getAllMapPoints(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return std::vector<std::shared_ptr<MapPoint>>(m_set_sptrMapPoints.begin(),m_set_sptrMapPoints.end());
  }
  std::vector<std::shared_ptr<MapPoint>> Map::getReferenceMapPoints(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return m_v_sptrReferenceMapPoints;
  }
  long int Map::getKeyFramesNum(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return m_set_sptrKeyFrames.size();
  }
  long int Map::getMapPointsNum(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return m_set_sptrMapPoints.size();
  }
  long int Map::getMaxKeyFrameID(){
    std::unique_lock<std::mutex> lock(m_mutex_map);
    return m_int_maxKeyFrameID;
  }
  void Map::clearAll(){
    //no need to delete because smart pointer is used.
    m_set_sptrKeyFrames.clear();
    m_set_sptrMapPoints.clear();
    m_int_maxKeyFrameID = 0;
    m_v_sptrReferenceMapPoints.clear();
    m_v_sptrOriginalKeyFrames.clear();
  }
}//YDORBSLAM
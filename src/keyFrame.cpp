#include "keyFrame.hpp"
#include "converter.hpp"
#include "orbMatcher.hpp"

namespace YDORBSLAM{
  long int KeyFrame::m_int_reservedKeyFrameID=0;
  std::mutex KeyFrame::m_mutex_keyFrameID;
  KeyFrame::KeyFrame(const Frame &_frame, std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase):\
  Frame(_frame),m_sptr_map(_sptrMap),m_sptr_keyFrameDatabase(_sptrKeyFrameDatabase){
    std::unique_lock<std::mutex> lock(m_mutex_keyFrameID);
    m_int_keyFrameID = m_int_reservedKeyFrameID++;
  }
  void KeyFrame::addConnection(std::shared_ptr<KeyFrame> _sptrKeyFrame, const int &_weight){
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      if(!m_dic_connectedKeyFrameWeights.count(_sptrKeyFrame) || m_dic_connectedKeyFrameWeights[_sptrKeyFrame]!=_weight){
        m_dic_connectedKeyFrameWeights[_sptrKeyFrame] = _weight;
      }else {
        return;
      }
    }
    orderConnectionsByWeight();
  }
  void KeyFrame::eraseConnection(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    bool b_isToOrder = false;
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      if(m_dic_connectedKeyFrameWeights.count(_sptrKeyFrame)){
        m_dic_connectedKeyFrameWeights.erase(_sptrKeyFrame);
        b_isToOrder = true;
      }
    }
    if(b_isToOrder){
      orderConnectionsByWeight();
    }
  }
  void KeyFrame::updateConnections(){
    std::map<std::shared_ptr<KeyFrame>,int> dicKeyFrameCounter;
    std::vector<std::shared_ptr<MapPoint>> vSptrMapPoints;
    {
      std::unique_lock<std::mutex> lockMapPoints(m_mutex_keyPoints);
      vSptrMapPoints = m_v_sptrMapPoints;
    }
    //for all map points in the key frame, check in which other key frames they are seen
    //increase counter for those key frames
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : vSptrMapPoints){
      if((sptrMapPoint) && !sptrMapPoint->isBad()){
        for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : sptrMapPoint->getObservations()){
          if(observation.first->m_int_keyFrameID!=m_int_keyFrameID){
            dicKeyFrameCounter[observation.first]++;
          }
        }
      }
    }
    if(dicKeyFrameCounter.empty()){
      return;
    }
    //add connection if the counter is larger than a threshold
    //add the the key frame with maximum counter, if no key frame counter is larger than the threshold
    int maxCount = 0;
    std::shared_ptr<KeyFrame> sptrKeyFrameOfMaxCount;
    std::vector<std::pair<int,std::shared_ptr<KeyFrame>>> vPairs;
    vPairs.reserve(dicKeyFrameCounter.size());
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &pair : dicKeyFrameCounter){
      if(pair.second>maxCount){
        maxCount = pair.second;
        sptrKeyFrameOfMaxCount = pair.first;
      }
      if(pair.second>15){
        vPairs.push_back(std::make_pair(pair.second,pair.first));
        pair.first->addConnection(shared_from_this(),pair.second);
      }
    }
    if(vPairs.empty()){
      vPairs.push_back(std::make_pair(maxCount,sptrKeyFrameOfMaxCount));
      sptrKeyFrameOfMaxCount->addConnection(shared_from_this(),maxCount);
    }
    std::sort(vPairs.begin(),vPairs.end(),[](const std::pair<int,std::shared_ptr<KeyFrame>> &a, const std::pair<int,std::shared_ptr<KeyFrame>> &b){return a.first>b.first;});
    std::list<std::shared_ptr<KeyFrame>> listKeyFrames;
    std::list<int> listWeights;
    for(const std::pair<int,std::shared_ptr<KeyFrame>> &pair : vPairs){
      listKeyFrames.push_back(pair.second);
      listWeights.push_back(pair.first);
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      m_dic_connectedKeyFrameWeights = dicKeyFrameCounter;
      m_v_orderedConnectedKeyFrames = std::vector<std::shared_ptr<KeyFrame>>(listKeyFrames.begin(),listKeyFrames.end());
      m_v_orderedWeights = std::vector<int>(listWeights.begin(),listWeights.end());
      if(m_b_isFirstConnection && m_int_keyFrameID!=0){
        m_sptr_parent = m_v_orderedConnectedKeyFrames.front();
        m_sptr_parent->addChild(shared_from_this());
        m_b_isFirstConnection = false;
      }
    }
  }
  void KeyFrame::orderConnectionsByWeight(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    std::vector<std::pair<int,std::shared_ptr<KeyFrame>>> vPairs;
    vPairs.reserve(m_dic_connectedKeyFrameWeights.size());
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &pair : m_dic_connectedKeyFrameWeights){
      vPairs.push_back(std::make_pair(pair.second,pair.first));
    }
    std::sort(vPairs.begin(),vPairs.end(),[](const std::pair<int,std::shared_ptr<KeyFrame>> &a, const std::pair<int,std::shared_ptr<KeyFrame>> &b){return a.first>b.first;});
    std::list<std::shared_ptr<KeyFrame>> listKeyFrames;
    std::list<int> listWeights;
    for(const std::pair<int,std::shared_ptr<KeyFrame>> &pair : vPairs){
      listKeyFrames.push_back(pair.second);
      listWeights.push_back(pair.first);
    }
    m_v_orderedConnectedKeyFrames = std::vector<std::shared_ptr<KeyFrame>>(listKeyFrames.begin(),listKeyFrames.end());
    m_v_orderedWeights = std::vector<int>(listWeights.begin(),listWeights.end());
  }
  std::set<std::shared_ptr<KeyFrame>> KeyFrame::getConnectedKeyFrames(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    std::set<std::shared_ptr<KeyFrame>> ret;
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &pair : m_dic_connectedKeyFrameWeights){
      ret.insert(pair.first);
    }
    return ret;
  }
  std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getOrderedConnectedKeyFrames(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return m_v_orderedConnectedKeyFrames;
  }
  std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getFirstNumOrderedConnectedKeyFrames(const int &_amount){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    if((int)m_v_orderedConnectedKeyFrames.size()<_amount)
        return m_v_orderedConnectedKeyFrames;
    else
        return std::vector<std::shared_ptr<KeyFrame>>(m_v_orderedConnectedKeyFrames.begin(),m_v_orderedConnectedKeyFrames.begin()+_amount);
  }
  std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getOrderedConnectedKeyFramesLargerThanWeight(const int &_weight){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    if(m_v_orderedConnectedKeyFrames.empty()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }
    std::vector<int>::iterator it = std::upper_bound(m_v_orderedWeights.begin(),m_v_orderedWeights.end(),_weight,[](const int &a, const int &b){return a>b;});
    if(it==m_v_orderedWeights.end()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }else {
      int amount = it - m_v_orderedWeights.begin();
      return std::vector<std::shared_ptr<KeyFrame>>(m_v_orderedConnectedKeyFrames.begin(),m_v_orderedConnectedKeyFrames.begin()+amount);
    }
  }
  int KeyFrame::getWeight(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    if(m_dic_connectedKeyFrameWeights.count(_sptrKeyFrame)){
      return m_dic_connectedKeyFrameWeights[_sptrKeyFrame];
    }else {
      return 0;
    }
  }
  void KeyFrame::addChild(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    m_set_sptrChildren.insert(_sptrKeyFrame);
  }
  void KeyFrame::eraseChild(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    m_set_sptrChildren.erase(_sptrKeyFrame);
  }
  void KeyFrame::changeParent(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    m_sptr_parent = _sptrKeyFrame;
    _sptrKeyFrame->addChild(shared_from_this());
  }
  std::set<std::shared_ptr<KeyFrame>> KeyFrame::getChildren(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return m_set_sptrChildren;
  }
  std::shared_ptr<KeyFrame> KeyFrame::getParent(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return m_sptr_parent;
  }
  bool KeyFrame::hasChild(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return (bool)(m_set_sptrChildren.count(_sptrKeyFrame));
  }
  void KeyFrame::addLoopEdge(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    m_b_isEraseExempted = true;
    m_set_sptrLoopEdges.insert(_sptrKeyFrame);
  }
  std::set<std::shared_ptr<KeyFrame>> KeyFrame::getLoopEdges(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return m_set_sptrLoopEdges;
  }
  void KeyFrame::addMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint, const int &_idx){
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    m_v_sptrMapPoints[_idx]=_sptrMapPoint;
  }
  void KeyFrame::eraseMatchedMapPoint(const int &_idx){
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    m_v_sptrMapPoints[_idx] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
  }
  void KeyFrame::eraseMatchedMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint){
    int idx = _sptrMapPoint->getIdxInKeyFrame(shared_from_this());
    if(idx>=0){
      std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
      m_v_sptrMapPoints[idx] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
    }
  }
  void KeyFrame::replaceMapPointMatch(std::shared_ptr<MapPoint> _sptrMapPoint, const int &_idx){
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    m_v_sptrMapPoints[_idx] = _sptrMapPoint;
  }
  std::set<std::shared_ptr<MapPoint>> KeyFrame::getMatchedMapPointsSet(){//getMapPoints
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    std::set<std::shared_ptr<MapPoint>> ret;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : m_v_sptrMapPoints){
      if(sptrMapPoint && !sptrMapPoint->isBad()){
        ret.insert(sptrMapPoint);
      }
    }
    return ret;
  }
  std::vector<std::shared_ptr<MapPoint>> KeyFrame::getMatchedMapPointsVec(){//getMatchedMapPoints
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    return m_v_sptrMapPoints;
  }
  int KeyFrame::trackedMapPointsNum(const int &_minObservationNum){
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    int pointsNum = 0;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : m_v_sptrMapPoints){
      if(sptrMapPoint && !sptrMapPoint->isBad()){
        if(_minObservationNum>0){
          if(sptrMapPoint->getObservationsNum()>=_minObservationNum){
            pointsNum++;
          }
        }else{
          pointsNum++;
        }
      }
    }
    return pointsNum;
  }
  std::shared_ptr<MapPoint> KeyFrame::getMapPoint(const int &_idx){
    std::unique_lock<std::mutex> lock(m_mutex_keyPoints);
    return m_v_sptrMapPoints[_idx];
  }
  void KeyFrame::setEraseExemption(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    m_b_isEraseExempted = true;
  }
  void KeyFrame::cancelEraseExemption(){
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      if(m_set_sptrLoopEdges.empty()){
        m_b_isEraseExempted = false;
      }
    }
    if(m_b_isEraseRequested){
      setBadFlag();
    }
  }
  void KeyFrame::setBadFlag(){
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      if(m_int_keyFrameID == 0){
        return;
      }else if(m_b_isEraseExempted){
        m_b_isEraseRequested = true;
        return;
      }
    }
    for(const std::pair<std::shared_ptr<KeyFrame>,int> &pair : m_dic_connectedKeyFrameWeights){
      pair.first->eraseConnection(shared_from_this());
    }
    for(std::shared_ptr<MapPoint> &sptrMapPoint : m_v_sptrMapPoints){
      if(sptrMapPoint){
        sptrMapPoint->eraseObservation(shared_from_this());
      }
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_connections);
      std::unique_lock<std::mutex> lock1(m_mutex_keyPoints);
      m_dic_connectedKeyFrameWeights.clear();
      m_v_orderedConnectedKeyFrames.clear();
      //update spanning tree
      std::set<std::shared_ptr<KeyFrame>> setParentCandidates;
      setParentCandidates.insert(m_sptr_parent);
      //assign a parent, the pair with highest covisibility weight, to a child
      //include the child as new parent candidate for the rest
      while(!m_set_sptrChildren.empty()){
        bool bContinue = false;
        int maxWeight = -1;
        std::shared_ptr<KeyFrame> sptrChild;
        std::shared_ptr<KeyFrame> sptrParent;
        for(const std::shared_ptr<KeyFrame> &child : m_set_sptrChildren){
          if(!child->isBad()){
            //check if a parent candidate is connected to the key frame
            std::vector<std::shared_ptr<KeyFrame>> vSptrConnectedKeyFrames = child->getOrderedConnectedKeyFrames();
            for(const std::shared_ptr<KeyFrame> &connectedKeyFrame : vSptrConnectedKeyFrames){
              for(const std::shared_ptr<KeyFrame> &parentCandidate : setParentCandidates){
                if(connectedKeyFrame->m_int_keyFrameID == parentCandidate->m_int_keyFrameID){
                  if(child->getWeight(connectedKeyFrame)>maxWeight){
                    sptrChild = child;
                    sptrParent = connectedKeyFrame;
                    maxWeight = getWeight(connectedKeyFrame);
                    bContinue = true;
                  }
                }
              }
            }
          }
        }
        if(bContinue){
          sptrChild->changeParent(sptrParent);
          setParentCandidates.insert(sptrChild);
          m_set_sptrChildren.erase(sptrChild);
        }else{
          break;
        }
      }
      //if a child has no covisibility connection with any parent candidate, assign the original parent to this key frame
      if(!m_set_sptrChildren.empty()){
        for(const std::shared_ptr<KeyFrame> &child : m_set_sptrChildren){
          child->changeParent(m_sptr_parent);
        }
      }
      m_sptr_parent->eraseChild(shared_from_this());
      m_cvMat_T_c2p = m_cvMat_T_c2w * m_sptr_parent->getInverseCameraPoseByTransform_w2c();
      m_b_isBad = true;
    }
    m_sptr_map->eraseKeyFrame(shared_from_this());
    m_sptr_keyFrameDatabase->erase(shared_from_this());
  }
  bool KeyFrame::isBad(){
    std::unique_lock<std::mutex> lock(m_mutex_connections);
    return m_b_isBad;
  }
}//namespace YDORBSLAM
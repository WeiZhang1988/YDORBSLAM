#include "loopClosing.hpp"
#include "sim3Solver.hpp"
#include "converter.hpp"
#include "optimizer.hpp"
#include "orbMatcher.hpp"

namespace YDORBSLAM{
  LoopClosing::LoopClosing(std::shared_ptr<Map> _sptr_map, std::shared_ptr<KeyFrameDatabase> _sptr_keyFrameDB, std::shared_ptr<DBoW3::Vocabulary> _sptr_voc, const bool &_b_isScaleFixed) : 
  m_sptr_map(_sptr_map),m_sptr_keyFrameDB(_sptr_keyFrameDB),m_sptr_voc(_sptr_voc),m_b_isScaleFixed(_b_isScaleFixed){}
  void LoopClosing::run(){
    m_b_isFinished = false;
    while(true){
      //check if there are key frames in the queue
      //detect loop candidates and check connection consistency
      //compute similarity transformation [sR|t] in the stereo/RGBD case s = 1
      if(checkNewKeyFrames() && detectLoop() && computeSim3()){
        //perform loop fusion and pos graph optimization
        correctLoop();
      }
      resetIfRequested();
      if(isFinished()){
        break;
      }
      usleep(5000);
    }
    setFinish();
  }
  void LoopClosing::insertKeyFrame(std::shared_ptr<KeyFrame> _sptr_keyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
    if(_sptr_keyFrame->m_int_keyFrameID!=0){
      m_list_keyFrameBufferQueue.push_back(_sptr_keyFrame);
    }
  }
  bool LoopClosing::detectLoop(){
    {
      std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
      m_sptr_currentLoopKeyFrame = m_list_keyFrameBufferQueue.front();
      m_list_keyFrameBufferQueue.pop_front();
      //avoid key frame being erased while bing processed by this thread
      m_sptr_currentLoopKeyFrame->setEraseExemption();
    }
    //if the map contains less than 10 key frames or less than 10 key frames have passed from last loop detection
    if(m_sptr_currentLoopKeyFrame->m_int_keyFrameID < m_int_lastLoopKeyFrameID + 10){
      m_sptr_keyFrameDB->add(m_sptr_currentLoopKeyFrame);
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
    //compute reference BoW similarity score
    //this is the lowest score to a connected key frame in the connected graph
    //we will impose loop candidates to have a higher similarity than this
    float minScore = 1.0f;
    for(const std::shared_ptr<KeyFrame>& connectedKeyFrame : m_sptr_currentLoopKeyFrame->getOrderedConnectedKeyFrames()){
      if(connectedKeyFrame && !connectedKeyFrame->isBad()){
        float score = m_sptr_voc->score(m_sptr_currentLoopKeyFrame->m_bow_wordVec, connectedKeyFrame->m_bow_wordVec);
        if(score < minScore){
          minScore = score;
        }
      }
    }
    //query the database imposing the minimum score
    std::vector<std::shared_ptr<KeyFrame>> candidateKeyFrames = m_sptr_keyFrameDB->detectLoopCandidates(m_sptr_currentLoopKeyFrame, minScore);
    //if there is no loop candidate, just add new key frame and return false
    if(candidateKeyFrames.empty()){
      m_sptr_keyFrameDB->add(m_sptr_currentLoopKeyFrame);
      m_v_lastConsistentGroups.clear();
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
    //for each loop candidate check consistency with previous loop candidates
    //each candidate expands a connection group (key frames connected to the loop candidate in the connection graph )
    //a group is consistent with previous group is they share at least a key frame
    //to accept it detection of a consistent loop in servaral consectutive key frames is neccessary
    m_v_consistentCandidates.clear();
    std::vector<KeyFrameAndNum> currentConsistentGroups;
    std::vector<bool> isLastGroupConsistent(m_v_lastConsistentGroups.size(),false);
    for(std::shared_ptr<KeyFrame>& candidateKeyFrame : candidateKeyFrames){
      std::set<std::shared_ptr<KeyFrame>> setCandidateGroup = candidateKeyFrame->getConnectedKeyFrames();
      setCandidateGroup.insert(candidateKeyFrame);
      bool isConsistencyEnough = false;
      bool isConsistentForSomeGroup = false;
      for(int i=0;i<m_v_lastConsistentGroups.size();i++){
        for(const std::shared_ptr<KeyFrame>& candidateKeyFrameInGroup : setCandidateGroup){
          if(m_v_lastConsistentGroups[i].first.count(candidateKeyFrameInGroup)){
            isConsistentForSomeGroup = true;
            if(!isLastGroupConsistent[i]){
              KeyFrameAndNum consistentGroup = std::make_pair(setCandidateGroup, m_v_lastConsistentGroups[i].second+1);
              currentConsistentGroups.push_back(consistentGroup);
              isLastGroupConsistent[i] = true;//this avoid to include the same group more than once
            }
            if(m_v_lastConsistentGroups[i].second+1>=m_int_connectionConsistencyThd && !isConsistencyEnough){
              m_v_consistentCandidates.push_back(candidateKeyFrame);
              isConsistencyEnough = true;//this avoid to include the same group more than once
            }
            break;
          }
        }
      }
      //if the group is not consistent with any previous group insert with consistency counter set to zero
      if(!isConsistentForSomeGroup){
        KeyFrameAndNum consistentGroup = std::make_pair(setCandidateGroup, 0);
        currentConsistentGroups.push_back(consistentGroup);
      }
    }
    //update connection consistent groups
    m_v_lastConsistentGroups = currentConsistentGroups;
    //stop here
  }
}
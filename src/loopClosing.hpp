#ifndef LOOPCLOSING_HPP
#define LOOPCLOSING_HPP

#include "keyFrame.hpp"
#include "localMapping.hpp"
#include "map.hpp"
#include "tracking.hpp"
#include "keyFrameDatabase.hpp"
#include "DBoW3/DBoW3.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/config.h>
#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include "stack_compatible_enable_shared_from_this.hpp"
#include "unistd.h"
#include <thread>
#include <mutex>
#include <map>
#include <memory>

namespace YDORBSLAM{
  class Tracking;
  class LocalMapping;
  class KeyFrameDatabase;
  class LoopClosing : public stack_compatible_enable_shared_from_this<LoopClosing>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //consistent group is a set of connected key frames of current key frame
    //and number of times of it being continuously found in groups of previous key frame
    typedef std::pair<std::set<std::shared_ptr<KeyFrame>>,int> KeyFrameAndNum;
    typedef std::map<std::shared_ptr<KeyFrame>,g2o::Sim3,std::less<std::shared_ptr<KeyFrame>>,Eigen::aligned_allocator<std::pair<const std::shared_ptr<KeyFrame>, g2o::Sim3>>> KeyFrameAndPose;
    LoopClosing(std::shared_ptr<Map> _sptr_map, std::shared_ptr<KeyFrameDatabase> _sptr_keyFrameDB, std::shared_ptr<DBoW3::Vocabulary> _sptr_voc, const bool &_b_isScaleFixed);
    inline void setTracker(std::shared_ptr<Tracking> _sptr_tracker){
      m_sptr_tracker = _sptr_tracker;
    }
    inline void setLocalMapper(std::shared_ptr<LocalMapping> _sptr_localMapper){
      m_sptr_localMapper = _sptr_localMapper;
    }
    //main function
    void run();
    void insertKeyFrame(std::shared_ptr<KeyFrame> _sptr_keyFrame);
    void requestReset();
    //this function will run in a separate thread
    void runGlobalBundleAdjustment(long int _int_loopKeyFrameID);
    inline bool isRunningGlobalBA(){
      std::unique_lock<std::mutex> lock(m_mutex_globalBA);
      return m_b_isRunningGlobalBA;
    }
    inline bool isGlobalBAFinished(){
      std::unique_lock<std::mutex> lock(m_mutex_globalBA);
      return m_b_isGlobalBAFinished;
    }
    inline void requestFinish(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      m_b_isFinishRequested = true;
    }
    inline bool isFinished(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      return m_b_isFinished;
    }
    protected:
    inline bool checkNewKeyFrames(){
      std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
      return(!m_list_keyFrameBufferQueue.empty());
    }
    bool detectLoop();
    bool computeSim3();
    void searchAndFuse(const KeyFrameAndPose &_correctedPosesMap);
    void correctLoop();
    void resetIfRequested();
    inline bool isFinishRequested(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      return m_b_isFinishRequested;
    }
    inline void setFinish(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      m_b_isFinished = true;
    }
    bool m_b_isResetRequested = false;
    bool m_b_isFinishRequested = false;
    bool m_b_isFinished = true;
    std::shared_ptr<Map> m_sptr_map;
    std::shared_ptr<Tracking> m_sptr_tracker;
    std::shared_ptr<KeyFrameDatabase> m_sptr_keyFrameDB;
    std::shared_ptr<DBoW3::Vocabulary> m_sptr_voc;
    std::shared_ptr<LocalMapping> m_sptr_localMapper;
    std::list<std::shared_ptr<KeyFrame>> m_list_keyFrameBufferQueue;
    int m_int_connectionConsistencyThd = 3;
    std::shared_ptr<KeyFrame> m_sptr_currentLoopKeyFrame;
    std::shared_ptr<KeyFrame> m_sptr_currentMatchedKeyFrame = std::shared_ptr<KeyFrame>(nullptr);
    std::vector<KeyFrameAndNum> m_v_lastConsistentGroups;
    std::vector<std::shared_ptr<KeyFrame>> m_v_consistentCandidates;
    std::vector<std::shared_ptr<KeyFrame>> m_v_connectedKeyFrames;
    std::vector<std::shared_ptr<MapPoint>> m_v_matchedMapPoints;
    std::vector<std::shared_ptr<MapPoint>> m_v_loopMapPoints;
    cv::Mat m_cvMat_sim3_c2w;
    g2o::Sim3 m_g2o_sim3_c2w;
    long int m_int_lastLoopKeyFrameID = 0;
    int m_int_fullBAIndex;
    bool m_b_isRunningGlobalBA = false, m_b_isGlobalBAFinished = true, m_b_isGlobalBAStopped = false, m_b_isScaleFixed;
    std::mutex m_mutex_reset, m_mutex_finish, m_mutex_loopQueue, m_mutex_globalBA;
    std::shared_ptr<std::thread> m_sptr_globalBAThread = std::shared_ptr<std::thread>(nullptr);
  };
}//namespace YDORBSLAM

#endif //LOOPCLOSING_HPP
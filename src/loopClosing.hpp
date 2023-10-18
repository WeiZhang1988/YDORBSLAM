/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_LOOPCLOSING_HPP
#define YDORBSLAM_LOOPCLOSING_HPP

#include "keyFrame.hpp"
#include "localMapping.hpp"
#include "map.hpp"
#include "tracking.hpp"
#include "keyFrameDatabase.hpp"
#include "DBoW3/DBoW3.h"
#include <Eigen/Core>
#include <Eigen/Dense>
//#include <g2o/config.h>
//#include <g2o/core/eigen_types.h>
//#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include "stack_compatible_enable_shared_from_this.hpp"
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
  // consistent group , a set of covisible keyframe and the continuous length of this group 
  typedef std::pair<std::set<std::shared_ptr<KeyFrame>>, int> setSptrKFAndLength;
  typedef std::map<std::shared_ptr<KeyFrame>,g2o::Sim3,std::less<std::shared_ptr<KeyFrame>>,\
  Eigen::aligned_allocator<std::pair<const std::shared_ptr<KeyFrame>, g2o::Sim3>>> sptrKeyFrameAndPose;
  LoopClosing(std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrameDatabase> _sptrDB, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, const bool& _bFixScale);
  inline void setTracker(std::shared_ptr<Tracking>> _sptrTracker){
    m_sptrTracker = _sptrTracker;
  }
  inline void setLocalMapper(std::shared_ptr<LocalMapping> _sptrLocalMapper){
    m_sptrLocalMapper = _sptrLocalMapper;
  }
  // Main function
  void run();
  void insertKeyFrame(std::shared_ptr<KeyFrame> _sptrKF);
  void requestReset();
  // This function will run in a separate thread
  void runGlobalBundleAdjustment(unsigned long _nLoopKF);
  inline bool isRunningGlobalBA(){
    std::unique_lock<std::mutex> lock(m_mutex_globalBA);
    return mbRunningGlobalBA;
  }
  inline bool isFinishedGlobalBA(){
    std::unique_lock<std::mutex> lock(m_mutex_globalBA);
    return mbFinishedGlobalBA;
  }   
  inline void requestFinish(){
    std::unique_lock<std::mutex> lock(m_mutex_finish);
    m_b_finishRequested = true;
  }
  inline bool isFinished(){
    std::unique_lock<std::mutex> lock(m_mutex_finish);
    return m_b_finished;
  }
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
  bool checkNewKeyFrames();
  bool detectLoop();
  bool computeSim3();
  void searchAndFuse(const sptrKeyFrameAndPose& _correctedPosesMap);
  void correctLoop();
  void resetIfRequested();
  bool m_b_resetRequested;
  std::mutex m_mutex_reset;
  inline bool checkFinish(){
    std::unique_lock<std::mutex> lock(m_mutex_finish);
    return m_b_finishRequested;
  }
  inline void setFinish(){
    std::unique_lock<std::mutex> lock(m_mutex_finish);
    m_b_finished = true;
  }
  bool m_b_finishRequested;
  bool m_b_finished;
  std::mutex m_mutex_finish;
  std::shared_ptr<Map> m_sptrMap;
  std::shared_ptr<Tracking> m_sptrTracker;
  std::shared_ptr<KeyFrameDatabase> m_sptrKeyFrameDB;
  std::shared_ptr<DBoW3::Vocabulary> m_sptrORBVocabulary;
  std::shared_ptr<LocalMapping> m_sptrLocalMapper;
  std::list<std::shared_ptr<KeyFrame>> m_l_sptrLoopKeyFrameBufferQueue;
  std::mutex m_mutex_loopQueue;
  // Loop detector parameters
  const int m_int_covisibilityConsistencyTh = 3;
  // Loop detector variables
  std::shared_ptr<KeyFrame> m_sptrCurrentLoopKF;
  std::shared_ptr<KeyFrame> m_sptrCurrentClosedLoopMatchedKF;
  std::vector<setSptrKFAndLength> m_v_lastConsistentGroups;
  // similar to NMS
  std::vector<std::shared_ptr<KeyFrame>> m_v_sptrEnoughConsistentCandidates;
  std::vector<std::shared_ptr<KeyFrame>> m_v_sptrCurrentConnectedKFs;
  std::vector<std::shared_ptr<MapPoint>> m_v_sptrCurrentMatchedPoints;
  std::vector<std::shared_ptr<MapPoint>> m_v_sptrLoopMapPoints;
  cv::Mat m_cvMatScw;
  g2o::Sim3 m_g2oScw;
  long unsigned int m_int_lastLoopKFid;
  // Variables related to Global Bundle Adjustment
  bool m_b_runningGlobalBA;
  bool m_b_finishedGlobalBA;
  bool m_b_stopGlobalBA;
  std::mutex m_mutex_globalBA;
  std::shared_ptr<std::thread> m_sptrThreadGlobalBA;
  // Fix scale in the stereo/RGB-D case
  bool m_b_fixScale;
  int m_int_fullBAIdx;
};
}//namespace YDORBSLAM

#endif //YDORBSLAM_LOOPCLOSING_HPP
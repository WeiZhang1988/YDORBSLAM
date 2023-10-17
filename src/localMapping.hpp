/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_LOCALMAPPING_HPP
#define YDORBSLAM_LOCALMAPPING_HPP

#include "keyFrame.hpp"
#include "mapPoint.hpp"
#include "map.hpp"
#include "loopClosing.hpp"
#include "tracking.hpp"
#include "keyFrameDatabase.hpp"
#include <memory>
#include <list>
#include <mutex>

namespace YDORBSLAM{
  class Tracking;
  class LoopClosing;
  class Map;
  class LocalMapping{
    public:
    LocalMapping(std::shared_ptr<Map> _sptr_map) : m_sptr_map(_sptr_map){}
    void setLoopCloser(std::shared_ptr<LoopClosing> _sptr_loopCloser);
    void setTracker(std::shared_ptr<Tracking> _sptr_tracker);
    //main function
    void run();
    void insertKeyFrame(std::shared_ptr<KeyFrame> _sptr_keyFrame);
    //tread synch
    void requestStop();
    void requestReset();
    bool stop();
    void release();
    inline bool isStopped(){
      std::unique_lock<std::mutex> lock(m_mutex_stop);
      return m_b_isStopped;
    }
    inline bool isStopRequested(){
      std::unique_lock<std::mutex> lock(m_mutex_stop);
      return m_b_isStopRequested;
    }
    inline bool isToAcceptKeyFrames(){
      std::unique_lock<std::mutex> lock(m_mutex_accept);
      return m_b_isToAcceptKeyFrames;
    }
    inline void setAcceptKeyFrames(bool _b_flag){
      std::unique_lock<std::mutex> lock(m_mutex_accept);
      m_b_isToAcceptKeyFrames = _b_flag;
    }
    bool setNotStop(bool _b_flag);
    inline void interruptBA(){
      m_b_isToAbortBA = true;
    }
    inline void requestFinish(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      m_b_isFinishRequested = true;
    }
    inline bool isFinished(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      return m_b_isFinished;
    }
    inline int getInQueueKeyFramesNum(){
      std::unique_lock<std::mutex> lock(m_mutex_newKeyFrames);
      return m_list_newKeyFrames.size();
    }
    protected:
    bool checkNewKeyFrames();
    void processNewKeyFrame();
    void createNewMapPoints();
    void cullMapPoint();
    void searchInNeighbors();
    void cullKeyFrame();
    cv::Mat computeFundamentalMatrix_first2second(std::shared_ptr<KeyFrame> _sptr_firstKeyFrame,std::shared_ptr<KeyFrame> _sptr_secondKeyFrame);
    cv::Mat skewSymmetricMatrix(const cv::Mat &_cvMat_vec);
    void resetIfRequested();
    inline bool isFinishRequested(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      return m_b_isFinishRequested;
    }
    inline void setFinished(){
      std::unique_lock<std::mutex> lock(m_mutex_finish);
      m_b_isFinished = true;    
      std::unique_lock<std::mutex> lock2(m_mutex_stop);
      m_b_isStopped = true;
    }
    bool m_b_isResetRequested     = false;
    bool m_b_isFinishRequested    = false;
    bool m_b_isFinished           = true;
    bool m_b_isToAbortBA          = false;
    bool m_b_isStopped            = false;
    bool m_b_isStopRequested      = false;
    bool m_b_isNotStoped          = false;
    bool m_b_isToAcceptKeyFrames  = true;
    std::shared_ptr<Map> m_sptr_map;
    std::shared_ptr<LoopClosing> m_sptr_loopcloser;
    std::shared_ptr<Tracking> m_sptr_tracker;
    std::shared_ptr<KeyFrame> m_sptr_currentKeyFrame;
    std::list<std::shared_ptr<KeyFrame>> m_list_newKeyFrames;
    std::list<std::shared_ptr<MapPoint>> m_list_recentAddedMapPoints;
    std::mutex m_mutex_reset, m_mutex_finish, m_mutex_newKeyFrames, m_mutex_stop, m_mutex_accept;
  };
}//namespace YDORBSLAM

#endif // YDORBSLAM_LOCALMAPPING_HPP
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
#include "map.hpp"
#include "loopClosing.hpp"
#include "tracking.hpp"
#include "keyFrameDatabase.hpp"
#include <mutex>
#include <memory>

namespace YDORBSLAM{
class Tracking;
class LoopClosing;
class Map;
class LocalMapping{
    public:
    LocalMapping(std::shared_ptr<Map> _sptrMap);
    inline void setLoopCloser(std::shared_ptr<LoopClosing> _sptrLoopCloser){
        m_sptrLoopCloser = _sptrLoopCloser;
    }
    inline void setTracker(std::shared_ptr<Tracking> _sptrTRacker){
        m_sptrTracker = _sptrTracker;
    }
    // Main function
    void run();
    void insertKeyFrame(std::shared_ptr<KeyFrame> _sptrKF);
    // Thread Synch
    void requestStop();
    void requestReset();
    bool stop();
    void release();
    bool setNotStop(bool _flag);
    inline bool isStopped(){
        std::unique_lock<std::mutex> lock(m_mutex_stop);
        return m_b_stopped;
    }
    inline bool stopRequested(){
        std::unique_lock<std::mutex> lock(m_mutex_stop);
        return m_b_stopRequested;
    }
    inline bool acceptKeyFrames(){
        std::unique_lock<std::mutex> lock(m_mutex_accept);
        return m_b_acceptKeyFrames;
    }
    inline void setAcceptKeyFrames(bool _flag){
        std::unique_lock<std::mutex> lock(m_mutex_accept);
        m_b_acceptKeyFrames = _flag;
    }
    inline void interruptBA(){
        m_b_abortBA = true;
    }
    inline void requestFinish(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        m_b_finishRequested = true;
    }
    inline bool isFinished(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        return m_b_finished;
    }
    inline int keyframesInQueue(){
        std::unique_lock<std::mutex> lock(m_mutex_newKFs);
        return m_l_newKeyFrames.size();
    }

    protected:
    bool checkNewKeyFrames();
    void processNewKeyFrame();
    void createNewMapPoints();
    void mapPointCulling();
    void searchInNeighbors();
    cv::Mat computeF12(std::shared_ptr<KeyFrame>& _sptrKF1, std::shared_ptr<KeyFrame>& _sptrKF2);
    void keyFrameCulling();
    cv::Mat skewSymmetricMatrix(const cv::Mat& _v);
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
        std::unique_lock<std::mutex> lock2(m_mutex_stop);
        m_b_stopped = true;
    }
    bool m_b_finishRequested;
    bool m_b_finished;
    std::mutex m_mutex_finish;
    std::shared_ptr<Map> m_sptrMap;
    std::shared_ptr<LoopClosing> m_sptrLoopCloser;
    std::shared_ptr<Tracking> m_sptrTracker;
    std::list<std::shared_ptr<KeyFrame>> m_l_newKeyFrames;
    std::shared_ptr<KeyFrame> m_sptrCurrentKeyFrame;
    std::list<std::shared_ptr<MapPoint>> m_l_sptrRecentAddedMapPoints;
    bool m_b_abortBA;
    std::mutex m_mutex_newKFs;
    bool m_b_stopped;
    bool m_b_stopRequested;
    bool m_b_notStop;
    std::mutex m_mutex_stop;
    bool m_b_acceptKeyFrames;
    std::mutex m_mutex_accept;
};

} //namespace YDORBSLAM

#endif // YDORBSLAM_LOCALMAPPING_HPP
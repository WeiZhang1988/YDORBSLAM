/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_VIEWER_HPP
#define YDORBSLAM_VIEWER_HPP

#include "frameDrawer.hpp"
#include "mapDrawer.hpp"
#include "tracking.hpp"
#include "system.hpp"
#include <mutex>

namespace YDORBSLAM{
class Tracking;
class FrameDrawer;
class MapDrawer;
class System;
class Viewer{
    public:
    Viewer(std::shared_ptr<System> _sptrSystem, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Tracking> _sptrTracking, const std::string& _strConfigurationPath);
    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void run();
    inline void requestFinish(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        m_b_finishRequested = true;
    }
    inline void requestStop(){
        std::unique_lock<std::mutex> lock(m_mutex_stop);
        if(!m_b_stopped){
            m_b_stopRequested = true;
        }
    }
    inline bool isFinished(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        return m_b_finished;
    }
    inline bool isStopped(){
        std::unique_lock<std::mutex> lock(m_mutex_stop);
        return m_b_stopped;
    }
    inline void release(){
        std::unique_lock<std::mutex> lock(m_mutex_stop);
        m_b_stopped = false;
    }
    private:
    bool stop();
    inline bool checkFinish(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        return m_b_finishRequested;
    }
    inline void setFinish(){
        std::unique_lock<std::mutex> lock(m_mutex_finish);
        m_b_finished = true;
    }
    std::shared_ptr<System> m_sptr_system;
    std::shared_ptr<FrameDrawer> m_sptr_frameDrawer;
    std::shared_ptr<MapDrawer> m_sptr_mapDrawer;
    std::shared_ptr<Tracking> m_sptr_tracker;
    // 1/fps in ms
    double m_db_period;
    float m_flt_imageWidth, m_flt_imageHeight;
    float m_flt_viewpointX, m_flt_viewpointY, m_flt_viewpointZ, m_flt_viewpointF;
    bool m_b_finishRequested;
    bool m_b_finished;
    bool m_b_stopRequested;
    bool m_b_stopped;
    std::mutex m_mutex_finish;
    std::mutex m_mutex_stop;
};
} // namespace YDORBSLAM

#endif // YDORBSLAM_VIEWER_HPP
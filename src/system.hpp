/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_SYSTEM_HPP
#define YDORBSLAM_SYSTEM_HPP

#include <string>
#include <thread>
#include <memory>
#include <opencv2/core/core.hpp>

#include "DBoW3/DBoW3.h"
#include "tracking.hpp"
#include "frameDrawer.hpp"
#include "mapDrawer.hpp"
#include "map.hpp"
#include "keyFrame.hpp"
#include "localMapping.hpp"
#include "loopClosing.hpp"
#include "keyFrameDatabase.hpp"
#include "viewer.hpp"
#include "enumclass.hpp"
#include "stack_compatible_enable_shared_from_this.hpp"

namespace YDORBSLAM{
class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;
class System : public stack_compatible_enable_shared_from_this<System>{
    public:
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const std::string& _strVocFile, const std::string& _strConfigurationPath, const Sensor& _enumSensor, const bool& _bUseViewer = true);
    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat trackStereo(const cv::Mat& _cvMatLeftImage, const cv::Mat& _cvMatRightImage, const double& _dbTimeStamp);
    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat trackRGBD(const cv::Mat& _cvMatImage, const cv::Mat& _cvMatDepth, const double& _dbTimeStamp);
    // This stops local mapping thread (map building) and performs only camera tracking.
    void activateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void deactivateLocalizationMode();
    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool mapChanged();
    // Reset the system (clear map)
    void reset();
    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void shutdown();
    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void saveTrajectoryTUM(const std::string& _filename);
    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void saveKeyFrameTrajectoryTUM(const std::string& _filename);
    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int getTrackingState();
    std::vector<std::shared_ptr<MapPoint>> getTrackedMapPoints();
    std::vector<cv::KeyPoint> getTrackedKeyPoints();
    private:
    // Input sensor
    Sensor m_enum_sensor;
    // ORB vocabulary used for place recognition and feature matching.
    std::shared_ptr<DBoW3::Vocabulary> m_sptr_vocabulary;
    // KeyFrame database for place recognition (relocalization and loop detection).
    std::shared_ptr<KeyFrameDatabase> m_sptr_keyFrameDatabase;
    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    std::shared_ptr<Map> m_sptr_map;
    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    std::shared_ptr<Tracking> m_sptr_tracker;
    // Local Mapper. It manages the local map and performs local bundle adjustment.
    std::shared_ptr<LocalMapping> m_sptr_localMapper;
    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    std::shared_ptr<LoopClosing> m_sptr_loopCloser;
    // The viewer draws the map and the current camera pose. It uses Pangolin.
    std::shared_ptr<Viewer> m_sptr_viewer;
    std::shared_ptr<FrameDrawer> m_sptr_frameDrawer;
    std::shared_ptr<MapDrawer> m_sptr_mapDrawer;
    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::shared_ptr<std::thread> m_sptr_threadLocalMapping;
    std::shared_ptr<std::thread> m_sptr_threadLoopClosing;
    std::shared_ptr<std::thread> m_sptr_threadViewer;
    // Reset flag
    std::mutex m_mutex_reset;
    bool m_b_reset;
    // Change mode flags
    std::mutex m_mutex_mode;
    bool m_b_activateLocalizationMode;
    bool m_b_deactivateLocalizationMode;
    // Tracking state
    int m_int_trackingState;
    std::vector<std::shared_ptr<MapPoint>> m_v_sptrTrackedMapPoints;
    std::vector<cv::KeyPoint> m_v_cvPointTrackedKeyPoints;
    std::mutex m_mutex_state;
};
} //namespace YDORBSLAM

#endif // YDORBSLAM_SYSTEM_HPP
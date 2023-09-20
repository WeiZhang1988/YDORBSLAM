/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_TRACKING_HPP
#define YDORBSLAM_TRACKING_HPP

#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"
#include "map.hpp"
#include "frame.hpp"
#include "keyFrameDatabase.hpp"
#include "orbExtractor.hpp"
#include "system.hpp"
#include "Viewer.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "stack_compatible_enable_shared_from_this.hpp"
#include <string>
#include <memory>
#include <mutex>

namespace YDORBSLAM{
class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;
class Tracking : public stack_compatible_enable_shared_from_this<Tracking>{
  public:
    Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
    std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, const string &_strConfigurationPath, const int sensor);
  
    void setLocalMapper(std::shared_ptr<LocalMapping> _sptrLocalMapper);
    void setLoopClosing(std::shared_ptr<LoopClosing> _sptrLoopClosing);
    void setViewer(std::shared_ptr<Viewer> _sptrViewer);

    //preprocess the input and call track(), extract key points and perform stereo matching.
    cv::Mat grabImageStereo(const cv::Mat &_leftImage, const cv::Mat &_rightImage, const double &_timeStamp);
    cv::Mat grabImageRGBD(const cv::Mat &_rgbImage,const cv::Mat &_depthImage, const double &_timeStamp);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::predictScaleLevel to take into account focal lenght
    void changeIntrinsics(const string &_strConfigurationPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void onlyLocalization(const bool &_flag);

    void reset();

    enum class eTrackingState{
      NOT_CONFIGURATION=-1,
      NOT_INPUT_IMAGES=0,
      NOT_INITIALIZED=1,
      TRACKING_SUCCESS=2,
      TRACKING_FAIL=3
    };

    eTrackingState m_enum_currentState;
    eTrackingState m_enum_lastState;

    // Input sensor
    int m_int_sensor;
    // Current Frame
    Frame m_currentFrame;
    cv::Mat m_cvMat_grayImage;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<cv::Mat> m_l_cvMatRelativeFramePoses;
    std::list<std::shared_ptr<KeyFrame>> m_l_sptrRefKF;
    std::list<double> m_l_dframeTimes;
    std::list<bool> m_l_btrackFail;

    // True if local mapping is deactivated and we are performing only localization
    bool m_b_onlyLocalization;

  protected:
    // Main tracking function. It is independent of the input sensor.
    void Track();
    // Map initialization for stereo and RGB-D
    void stereoInitialization();
    // Check if there are MapPoints replaced in lastFrame
    void checkReplacedMPInLastFrame();
    bool trackWithReferenceKeyFrame();
    // create 100 temporal MapPoints in lastFrame only in Localization Mode
    void updateLastFrame();
    bool trackWithMotionModel();
    bool trackLocalMap();
    void searchLocalPoints();
    // relocalization in localmap
    void updateLocalMap();
    void updateLocalPoints();
    void updateLocalKeyFrames();
    bool relocalization();
    // product KeyFrame
    bool needNewKeyFrame();
    void createNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool m_b_fewMatched;
    //Other Thread Pointers
    std::shared_ptr<LocalMapping> m_sptr_localMapper;
    std::shared_ptr<LoopClosing> m_sptr_loopClosing;
    //ORB
    std::shared_ptr<OrbExtractor> m_sptr_leftOrbExtractor, m_sptr_rightOrbExtractor;
    //BoW
    std::shared_ptr<DBoW3::Vocabulary> m_sptr_orbVocabulary;
    std::shared_ptr<KeyFrameDatabase> m_sptr_keyFrameDB;
    //Local Map
    std::shared_prt<KeyFrame> m_sptr_referenceKF;
    std::vector<std::shared_prt<KeyFrame>> m_v_sptrLocalKeyFrames;
    std::vector<std::shared_prt<MapPoint>> m_v_sptrLocalMapPoints;
    //system
    std::shared_ptr<System> m_sptr_system;
    //Drawers
    std::shared_ptr<Viewer> m_sptr_viewer;
    std::shared_ptr<FrameDrawer> m_sptr_frameDrawer;
    std::shared_ptr<MapDrawer> m_sptr_mapDrawer;
    //Map
    std::shared_ptr<Map> m_sptr_map;
    //Calibration matrix
    cv::Mat m_cvMat_intParMat;
    cv::Mat m_cvMat_imageDistCoef;
    float m_flt_baseLineTimesFx;
    //New KeyFrame rules (according to fps)
    int m_int_minFrequency;
    int m_int_maxFrequency;
    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float m_flt_depthThd;
    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float m_flt_depthMapFactor;
    //Current matches MapPoints in frame
    int m_int_currentMatchedMapPoints;
    //Last Frame, KeyFrame and Relocalisation Info
    Frame m_lastFrame;
    std::shared_ptr<KeyFrame> m_sptr_lastKeyFrame;
    unsigned int m_int_lastKeyFrameId;
    unsigned int m_int_lastRelocFrameId;

    //Motion Model
    cv::Mat m_cvMat_velocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool m_b_isRGB;

    std::list<std::shared_ptr<MapPoint>> m_l_sptrTemporalMapPoints;
  };

}//namespace YDORBSLAM

#endif //YDORBSLAM_TRACKING_HPP
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
#include <string>
#include <vector>
#include <list>
#include <memory>
#include <mutex>

namespace YDORBSLAM{
  class Viewer;
  class FrameDrawer;
  class Map;
  class LocalMapping;
  class LoopClosing;
  class System;
  class Tracking{
    public:
    Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
    std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, const int _sensor, const string &_strSettingPath);
    //preprocess the input and call track(), extract key points and perform stereo matching.
    cv::Mat grabImageStereo(const cv::Mat &_leftImageRect, const cv::Mat &_rightImageRect, const double &_timestamp);
    cv::Mat grabImageRGBD(const cv::Mat &_imageRGB, const cv::Mat &_imageDepth, const double &_timestamp);
    void setLocalMapper(std::shared_ptr<LocalMapping> _sptrLocalMapper);
    void setLoopClosing(std::shared_ptr<LoopClosing> _sptrLoopClosing);
    void setViewer(std::shared_ptr<Viewer> _sptrViewer);
    void changeCalibration(const std::string &_strSettingPath);
    void informOnlyTracking(const bool &_flag);
    enum class TrackingState{
      SYSTEM_NOT_READY,
      NO_IMAGE_YET,
      NOT_INITIALIZED,
      OK,
      LOST
    };
    TrackingState m_ts_state = TrackingState::NO_IMAGE_YET;
    TrackingState m_ts_lastProcessedState;
    int m_int_sensor;
    Frame m_frame_currentFrame;
    cv::Mat m_cvMat_grayImage;
    //lists to recover the full camera trajectory at the end of execution
    //basically reference key frames are stored for each fream and its relative transformation
    std::list<cv::Mat> m_list_relativeFramePoses;
    std::list<std::shared_ptr<KeyFrame>> m_list_referenceKeyFrames;
    std::list<double> m_list_frameTimes;
    std::list<bool> m_list_lost;
    //true if local mapping is deactivated and only localization is performed
    bool m_b_isTrackingOnly = false;
    void reset();
    protected:
    //main tracking function and it is independent of the input sensor
    void track();
    //map initialization for stereo and RGBD
    void initializeStereo();
    //processing functions
    void checkReplacementInLastFrame();
    bool trackReferenceKeyFrame();
    void updateLastFrame();
    bool trackWithMotionModel();
    bool relocalize();
    void updateLocalMap();
    void updateLocalPoints();
    void updateLocalKeyFrames();
    bool trackLocalMap();
    void searchLocalPoints();
    bool needNewKeyFrame();
    void createNewKeyFrame();
    //in localization only mode, this flat is true if there is no match to points in the map.
    //tracking continues if there is not enough match with temporal points.
    //in this case, visual odometry is used.
    //the system do relocalization to recover "zero-drift" localization to the map.
    bool m_b_isDoingVisualOdometry = false;
    //other thread pointers
    std::shared_ptr<LocalMapping> m_sptr_localMapper;
    std::shared_ptr<LoopClosing> m_sptr_loopClosing;
    //Orb
    std::shared_ptr<OrbExtractor> m_sptr_leftOrbExtractor, m_sptr_rightOrbExtractor;
    std::shared_ptr<OrbExtractor> m_sptr_initOrbExtractor;
    //Bow
    std::shared_ptr<DBoW3::Vocabulary> m_sptr_orbVocabulary;
    std::shared_ptr<KeyFrameDatabase> m_sptr_keyFrameDataBase;
    //local map
    std::shared_ptr<KeyFrame> m_sptr_referenceKeyFrame;
    std::vector<std::shared_ptr<KeyFrame>> m_v_localKeyFrames;
    std::vector<std::shared_ptr<MapPoint>> m_v_localMapPoints;
    //system
    std::shared_ptr<System> m_sptr_system;
    //drawers
    std::shared_ptr<Viewer> m_sptr_viewer = std::shared_ptr<Viewer>(nullptr);
    std::shared_ptr<FrameDrawer> m_sptr_frameDrawer;
    std::shared_ptr<MapDrawer> m_sptr_mapDrawer;
    //map
    std::shared_ptr<Map> m_sptr_map;
    //calibration matrix
    cv::Mat m_cvMat_leftCamIntParMat, m_cvMat_rightCamIntParMat;
    cv::Mat m_cvMat_leftImageDistCoef, m_cvMat_rightImageDistCoef;
    float m_flt_baseLineTimesFx;
    //new key frame rules according to fps
    int m_int_minFramesNum;
    int m_int_maxFramesNum;
    //threshold for close/far point
    //points seen as close by the stereo/rgbd sensor are considered reliable
    //and inserted from just one frame.
    //for points require a match in two key frames.
    float m_flt_depthThd;
    //for rgbd inputs only. for some datasets(e.g. TUM) the depth map values are scaled
    float m_flt_depthMapFactor;
    //current match number in frame
    int m_int_matchInliersNum;
    //last frame, key frame and relocalization info
    std::shared_ptr<KeyFrame> m_sptr_lastKeyFrame;
    Frame m_frame_lastFrame;
    long int m_int_lastKeyFrameID;
    long int m_int_lastRelocalizedFrameID = 0;
    //motion model
    cv::Mat m_cvMat_velocity;
    //color order (true rgb, false bgr, ignored if grayscale)
    bool m_b_isRGB;
    //temporal points
    std::list<std::shared_ptr<MapPoint>> m_list_temperalPoints;
  };

}//namespace YDORBSLAM

#endif //YDORBSLAM_TRACKING_HPP
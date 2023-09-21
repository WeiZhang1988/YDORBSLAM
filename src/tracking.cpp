#include "tracking.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "optimizer.hpp"
#include "pnpSolver.hpp"

namespace YDORBSLAM{
  Tracking::Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
  std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, System::Sensor &_sensor, const string &_strSettingPath) : \
  m_sptr_system(_sptrSys), m_sptr_orbVocabulary(_sptrVoc), m_sptr_frameDrawer(_sptrFrameDrawer), m_sptr_mapDrawer(_sptrMapDrawer), m_sptr_map(_sptrMap), \
  m_sptr_keyFrameDataBase(_sptrKeyFrameDatabase), m_sys_sensor(_sensor){
    //load camera parameters from setting file
    cv::FileStorage settingFile(_strSettingPath, cv::FileStorage::READ);
    m_cvMat_intParMat.at<float>(0,0) = settingFile["Camera.fx"];
    m_cvMat_intParMat.at<float>(1,1) = settingFile["Camera.fy"];
    m_cvMat_intParMat.at<float>(0,2) = settingFile["Camera.cx"];
    m_cvMat_intParMat.at<float>(1,2) = settingFile["Camera.cy"];
    m_cvMat_leftImageDistCoef.at<float>(0) = settingFile["LeftCamera.k1"];
    m_cvMat_leftImageDistCoef.at<float>(1) = settingFile["LeftCamera.k2"];
    m_cvMat_leftImageDistCoef.at<float>(2) = settingFile["LeftCamera.p1"];
    m_cvMat_leftImageDistCoef.at<float>(3) = settingFile["LeftCamera.p2"];
    m_cvMat_leftImageDistCoef.at<float>(4) = settingFile["LeftCamera.k3"];
    m_cvMat_rightImageDistCoef.at<float>(0) = settingFile["RightCamera.k1"];
    m_cvMat_rightImageDistCoef.at<float>(1) = settingFile["RightCamera.k2"];
    m_cvMat_rightImageDistCoef.at<float>(2) = settingFile["RightCamera.p1"];
    m_cvMat_rightImageDistCoef.at<float>(3) = settingFile["RightCamera.p2"];
    m_cvMat_rightImageDistCoef.at<float>(4) = settingFile["RightCamera.k3"];
    m_flt_baseLineTimesFx = settingFile["Camera.bf"];
    m_int_maxFramesNum = settingFile["Camera.fps"]>0?settingFile["Camera.fps"]:30;
    std::cout << endl << "Camera Parameters: " << std::endl;
    std::cout << "- fx: " << m_cvMat_intParMat.at<float>(0,0) << std::endl;
    std::cout << "- fy: " << m_cvMat_intParMat.at<float>(1,1) << std::endl;
    std::cout << "- cx: " << m_cvMat_intParMat.at<float>(0,2) << std::endl;
    std::cout << "- cy: " << m_cvMat_intParMat.at<float>(1,2) << std::endl;
    std::cout << "- left k1: " << m_cvMat_leftImageDistCoef.at<float>(0) << std::endl;
    std::cout << "- left k2: " << m_cvMat_leftImageDistCoef.at<float>(1) << std::endl;
    std::cout << "- left p1: " << m_cvMat_leftImageDistCoef.at<float>(2) << std::endl;
    std::cout << "- left p2: " << m_cvMat_leftImageDistCoef.at<float>(3) << std::endl;
    std::cout << "- left k3: " << m_cvMat_leftImageDistCoef.at<float>(4) << std::endl;
    std::cout << "- right k1: " << m_cvMat_rightImageDistCoef.at<float>(0) << std::endl;
    std::cout << "- right k2: " << m_cvMat_rightImageDistCoef.at<float>(1) << std::endl;
    std::cout << "- right p1: " << m_cvMat_rightImageDistCoef.at<float>(2) << std::endl;
    std::cout << "- right p2: " << m_cvMat_rightImageDistCoef.at<float>(3) << std::endl;
    std::cout << "- right k3: " << m_cvMat_rightImageDistCoef.at<float>(4) << std::endl;
    std::cout << "- fps: " << m_int_maxFramesNum << std::endl;
    std::cout << m_b_isRGB?"- color order: RGB (ignored if grayscale)":"- color order: BGR (ignored if grayscale)" << std::endl;
    int keyPointsNum   = settingFile["ORBextractor.nFeatures"];
    float scaleFactor  = settingFile["ORBextractor.scaleFactor"];
    int levelsNum      = settingFile["ORBextractor.nLevels"];
    int initFastThd    = settingFile["ORBextractor.iniThFAST"];
    int minFastThd     = settingFile["ORBextractor.minThFAST"];
    m_sptr_leftOrbExtractor  = std::make_shared<OrbExtractor>(keyPointsNum,scaleFactor,levelsNum,initFastThd,minFastThd);
    m_sptr_rightOrbExtractor = std::make_shared<OrbExtractor>(keyPointsNum,scaleFactor,levelsNum,initFastThd,minFastThd);
    std::cout << std::endl << "ORB Extractor Parameters: " << std::endl;
    std::cout << "- Number of Features: " << keyPointsNum << std::endl;
    std::cout << "- Scale Factor: " << scaleFactor << std::endl;
    std::cout << "- Scale Levels: " << levelsNum << std::endl;
    std::cout << "- Initial Fast Threshold: " << initFastThd << std::endl;
    std::cout << "- Minimum Fast Threshold: " << minFastThd << std::endl;
    if(m_sys_sensor==System::Sensor::STEREO || m_sys_sensor==System::Sensor::RGBD){
      m_flt_depthThd = m_flt_baselineTimesFx * (float)settingFile["ThDepth"]/settingFile["Camera.fx"];
      std::cout << std::endl << "Depth Threshold (Close/Far Points): " << m_flt_depthThd << std::endl;
      if(m_sys_sensor==System::Sensor::RGBD){
        m_flt_depthMapFactor = settingFile["DepthMapFactor"];
        m_flt_depthMapFactor = fabs(m_flt_depthMapFactor)<1e-5?1.0f:1.0f/m_flt_depthMapFactor;
      }
    }
  }
  cv::Mat Tracking::grabImageStereo(const cv::Mat &_leftImageRect, const cv::Mat &_rightImageRect, const double &_timestamp){
    m_cvMat_grayImage = _leftImageRect;
    cv::Mat rightGrayImage = _rightImageRect;
    if(m_b_isRGB){
      if(m_cvMat_grayImage.channels()==3){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGB2GRAY);
        cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_RGB2GRAY);
      }else if(m_cvMat_grayImage.channels()==4){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGBA2GRAY);
        cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_RGBA2GRAY);
      }
    }else{
      if(m_cvMat_grayImage.channels()==3){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGR2GRAY);
        cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_BGR2GRAY);
      }else if(m_cvMat_grayImage.channels()==4){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGRA2GRAY);
        cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_BGRA2GRAY);
      }
    }
    m_frame_currentFrame = Frame(m_cvMat_grayImage,rightGrayImage,_timeStamp,m_cvMat_intParMat,m_cvMat_leftImageDistCoef,m_cvMat_rightImageDistCoef,m_flt_baseLineTimesFx,m_flt_depthThd,m_sptr_leftOrbExtractor,m_sptr_rightOrbExtractor,m_sptr_orbVocabulary);
    track();
    return m_frame_currentFrame.m_cvMat_T_c2w.clone();
  }
  cv::Mat Tracking::grabImageRGBD(const cv::Mat &_rgbImage, const cv::Mat &_depthImage, const double &_timestamp){
    m_cvMat_grayImage = _rgbImage;
    cv::Mat depthImage = _depthImage;
    if(m_b_isRGB){
      if(m_cvMat_grayImage.channels()==3){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGB2GRAY);
      }else if(m_cvMat_grayImage.channels()==4){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGBA2GRAY);
      }
    }else{
      if(m_cvMat_grayImage.channels()==3){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGR2GRAY);
      }else if(m_cvMat_grayImage.channels()==4){
        cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGRA2GRAY);
      }
    }
    if((fabs(m_flt_depthMapFactor-1.0f)>1e-5) || depthImage.type()!=CV_32F){
      depthImage.convertTo(depthImage,CV_32F,m_flt_depthMapFactor);
    }
    m_frame_currentFrame = Frame(m_cvMat_grayImage,depthImage,_timeStamp,m_cvMat_intParMat,m_cvMat_imageDistCoef,m_flt_baseLineTimesFx,m_flt_depthThd,m_sptr_leftOrbExtractor,m_sptr_orbVocabulary);
    track();
    return m_frame_currentFrame.m_cvMat_T_c2w.clone();
  }
  void Tracking::setLocalMapper(std::shared_ptr<LocalMapping> _sptrLocalMapper){
    m_sptr_localMapper = _sptrLocalMapper;
  }
  void Tracking::setLoopClosing(std::shared_ptr<LoopClosing> _sptrLoopClosing){
    m_sptr_loopClosing = _sptrLoopClosing;
  }
  void Tracking::setViewer(std::shared_ptr<Viewer> _sptrViewer){
      m_sptr_viewer = _sptrViewer;
  }
  void Tracking::changeIntParMat(const std::string &_strSettingPath){
    cv::FileStorage settingFile(_strSettingPath, cv::FileStorage::READ);
    m_cvMat_intParMat.at<float>(0,0) = settingFile["Camera.fx"];
    m_cvMat_intParMat.at<float>(1,1) = settingFile["Camera.fy"];
    m_cvMat_intParMat.at<float>(0,2) = settingFile["Camera.cx"];
    m_cvMat_intParMat.at<float>(1,2) = settingFile["Camera.cy"];
    m_cvMat_leftImageDistCoef.at<float>(0) = settingFile["LeftCamera.k1"];
    m_cvMat_leftImageDistCoef.at<float>(1) = settingFile["LeftCamera.k2"];
    m_cvMat_leftImageDistCoef.at<float>(2) = settingFile["LeftCamera.p1"];
    m_cvMat_leftImageDistCoef.at<float>(3) = settingFile["LeftCamera.p2"];
    m_cvMat_leftImageDistCoef.at<float>(4) = settingFile["LeftCamera.k3"];
    m_cvMat_rightImageDistCoef.at<float>(0) = settingFile["RightCamera.k1"];
    m_cvMat_rightImageDistCoef.at<float>(1) = settingFile["RightCamera.k2"];
    m_cvMat_rightImageDistCoef.at<float>(2) = settingFile["RightCamera.p1"];
    m_cvMat_rightImageDistCoef.at<float>(3) = settingFile["RightCamera.p2"];
    m_cvMat_rightImageDistCoef.at<float>(4) = settingFile["RightCamera.k3"];
    m_flt_baseLineTimesFx = settingFile["Camera.bf"];
    Frame::m_b_isComputeInit = true;
  }
  void Tracking::informOnlyTracking(const bool &_flag){
    m_b_isTrackingOnly = _flag;
  }
  void Tracking::reset(){
    std::cout<< "System Reseting" << std::endl;
    if(m_sptr_viewer){
      m_sptr_viewer->requestStop();
      while(!m_sptr_viewer->isStopped()){
        std::usleep(3000);
      }
    }
    //reset local mapping, loop closing, and BoW
    std::cout << "Reseting Local Mapper...";
    m_sptr_localMapper->requestReset();
    std::cout << " done" << std::endl;
    std::cout << "Reseting Loop Closing...";
    m_sptr_loopClosing->requestReset();
    std::cout << " done" << std::endl;
    std::cout << "Reseting Database...";
    m_sptr_keyFrameDataBase->clear();
    std::cout << " done" << std::endl;
    //clear map including map points and key frames
    m_sptr_map->clear();
    KeyFrame::m_int_reservedKeyFrameID = 0;
    Frame::m_int_reservedID = 0;
    m_ts_state = TrackingState::NO_IMAGE_YET;
    m_list_relFramePoses.clear();
    m_list_refKeyFrames.clear();
    m_list_frameTimes.clear();
    m_list_isLost.clear();
    if(m_sptr_viewer){
      m_sptr_viewer->release();
    }
  }
  void Tracking::track(){
    if(m_ts_state == TrackingState::NO_IMAGE_YET){
      m_ts_state = TrackingState::NOT_INITIALIZED;
    }
    m_ts_lastProcessedState = m_ts_state;
    std::unique_lock<std::mutex> lock(m_sptr_map->m_mutex_updateMap);
    if(m_ts_state == TrackingState::NOT_INITIALIZED){
      if(m_sys_sensor==System::Sensor::STEREO || m_sys_sensor==System::Sensor::RGBD){
        initializeStereo();
      }
      m_sptr_frameDrawer->update(shared_from_this());
      if(m_ts_state!=TrackingState::OK){
        return;
      }
    }else{
      bool isOK = false;
      if(m_b_isTrackingOnly){
        //localize only mode. local mapping is deactivated
        if(m_ts_state == TrackingState::LOST){
          isOK = relocalize();
        }else{
          if(m_b_isDoingVisualOdometry){
            //in last frame mainly "visual odometry" points are tracked
            //one camera pose is computed from motion model and the other one is from relocalization
            //if relocalization succeeds the solution is chosen, otherwise "visual odometry" solotion remains
            bool isTrackingWithMotionModelOK = false;
            bool isRelocalizationOK = false;
            std::vector<std::shared_ptr<MapPoint>> vSptrMapPointsByMotionModel;
            std::vector<bool> vIsOutliersByMotionModel;
            cv::Mat T_c2w_byMotionModel;
            if(!m_cvMat_velocity.empty()){
              isTrackingWithMotionModelOK = trackWithMotionModel();
              vSptrMapPointsByMotionModel = m_frame_currentFrame.m_v_sptrMapPoints;
              vIsOutliersByMotionModel    = m_frame_currentFrame.m_v_isOutliers;
              T_c2w_byMotionModel         = m_frame_currentFrame.m_cvMat_T_c2w.clone();
              isRelocalizationOK = relocalize();
              if(isTrackingWithMotionModelOK && !isRelocalizationOK){
                m_frame_currentFrame.setCameraPoseByTransform_c2w(T_c2w_byMotionModel);
                m_frame_currentFrame.m_v_sptrMapPoints = vSptrMapPointsByMotionModel;
                m_frame_currentFrame.m_v_isOutliers    = vIsOutliersByMotionModel;
                if(m_b_isDoingVisualOdometry){
                  int i_for=0;
                  for(std::shared_ptr<MapPoint> &sptrMapPoint : m_frame_currentFrame.m_v_sptrMapPoints){
                    if(sptrMapPoint && !m_currentFrame.m_v_isOutliers[i_for]){
                      sptrMapPoint->increaseFound();
                    }
                    i_for++;
                  }
                }
              }
            }else{
              isRelocalizationOK = relocalize();
            }
            if(isRelocalizationOK){
              m_b_isDoingVisualOdometry = false;
            }
            isOK = isTrackingWithMotionModelOK || isRelocalizationOK;
          }else{
            if(!m_cvMat_velocity.empty()){
              isOK = trackWithMotionModel();
            }else{
              isOK = trackReferenceKeyFrame();
            }
          }
        }
      }else{
        if(m_ts_state==TrackingState::OK){
          checkReplacementInLastFrame();
          if(!m_cvMat_velocity.empty() && m_frame_currentFrame.m_int_ID>=m_int_lastRelocalizedFrameID+2 && trackWithMotionModel()){
            isOK = true;
          }else{
            isOK = trackReferenceKeyFrame();
          }
        }else{
          isOK = relocalize();
        }
      }
      m_frame_currentFrame.m_sptr_refKeyFrame = m_sptr_refKeyFrame;
      //if there is an initial estimation of the camera pose and matching, track the local map
      if(isOK && (!m_b_isTrackingOnly || !m_b_isDoingVisualOdometry)){
        isOK = trackLocalMap();
      }
      if(isOK){
        m_ts_state = TrackingState::OK;
      }else{
        m_ts_state = TrackingState::LOST;
      }
      m_sptr_frameDrawer->update(shared_from_this());
      //if tracking is good, check if to insert a key frame
      if(isOK){
        if(m_frame_lastFrame.m_cvMat_T_c2w.empty()){
          m_cvMat_velocity = cv::Mat();
        }else{
          cv::Mat lastFrame_T_w2c = cv::Mat::eye(4,4,CV_32F);
          m_frame_lastFrame.getRotation_w2c().copyTo(lastFrame_T_w2c.rowRange(0,3).colRange(0,3));
          m_frame_lastFrame.getCameraOriginInWorld().copyTo(lastFrame_T_w2c.rowRange(0,3).col(3));
          m_cvMat_velocity = m_frame_currentFrame.m_cvMat_T_c2w * lastFrame_T_w2c;
        }
        m_sptr_mapDrawer->setCurrentCameraPose(m_frame_currentFrame.m_cvMat_T_c2w);
        //clean current matched map points
        for(int i=0;i<m_frame_currentFrame.m_int_keyPointsNum;i++){
          if(m_frame_currentFrame.m_v_sptrMapPoints[i] && m_frame_currentFrame.m_v_sptrMapPoints[i]->getObservationsNum()<1){
            m_currentFrame.m_v_isOutliers[i] = false;
            m_frame_currentFrame.m_v_sptrMapPoints[i] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
          }
        }
        //delete temporal map points, the for loop may not be needed.
        for(std::shared_ptr<MapPoint> &tempMapPoint : m_list_tempMapPoints){
          sptrMapPoint.reset();
        }
        m_list_tempMapPoints.clear();
        //check whether need to insert a new key frame
        if(needNewKeyFrame()){
          createNewKeyFrame();
        }
        //the outliers by the Huber function can pass to the new key frame, so that bundle adjustment will decide if they are real outliers
        //next frame does not need to estimate the position of outliers so they are discarded in the frame
        for(int i=0;i<m_frame_currentFrame.m_int_keyPointsNum;i++){
          if(m_frame_currentFrame.m_v_sptrMapPoints[i] && m_frame_currentFrame.m_v_isOutliers[i]){
            m_frame_currentFrame.m_v_sptrMapPoints[i] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
          }
        }
      }
      //reset if the camera get lost soon after initialization
      if(m_ts_state == TrackingState::LOST && m_sptr_map->getKeyFramesNum()<=5){
        std::cout<< "Track lost soon after initialisation, reseting..." << std::endl;
        m_sptr_system->reset();
        return;
      }
      if(!m_frame_currentFrame.m_sptr_refKeyFrame){
        m_frame_currentFrame.m_sptr_refKeyFrame = m_sptr_refKeyFrame;
      }
      m_frame_lastFrame = Frame(m_frame_currentFrame);
    }
    //store frame pose information to retrieve the complete camera trajectory afterwards
    if(!m_frame_currentFrame.m_cvMat_T_c2w.empty()){
      cv::Mat T_crr2ref = m_frame_currentFrame.m_cvMat_T_c2w * m_frame_currentFrame.m_sptr_refKeyFrame->getInverseCameraPoseByTransform_w2c();
      m_list_relFramePoses.push_back(T_crr2ref);
      m_list_refKeyFrames.push_back(m_sptr_refKeyFrame);
      m_list_frameTimes.push_back(m_frame_currentFrame.m_d_timeStamp);
      m_list_isLost.push_back(m_ts_state == TrackingState::LOST)
    }else{
      //this happens when tracking is lost
      m_list_relFramePoses.push_back(m_list_relFramePoses.back());
      m_list_refKeyFrames.push_back(m_list_refKeyFrames.back());
      m_list_frameTimes.push_back(m_list_frameTimes.back());
      m_list_isLost.push_back(m_ts_state == TrackingState::LOST)
    }
  }
  void Tracking::initializeStereo(){
    if(m_frame_currentFrame.m_int_keyPointsNum > 500){
      //set frame pose as the origin
      m_frame_currentFrame.setCameraPoseByTransform_c2w(cv::Mat::eye(4,4,CV_32F));
      //create key frame
      std::shared_ptr<KeyFrame> sptrInitKeyFrame = std::make_shared<KeyFrame>(m_frame_currentFrame,m_sptr_map,m_sptr_keyFrameDataBase);
      //insert key frame in the map
      m_sptr_map->addKeyFrame(sptrInitKeyFrame);
      //create map points and associate to key frame
      for(int i=0;i<m_frame_currentFrame.m_int_keyPointsNum;i++){
        if(m_frame_currentFrame.m_v_depth[i]>0){
          std::shared_ptr<MapPoint> sptrNewMapPoint = std::make_shared<MapPoint>(m_frame_currentFrame.inverseProject(i),m_sptr_map,sptrInitKeyFrame);
          sptrNewMapPoint->addObservation(sptrInitKeyFrame,i);
          sptrInitKeyFrame->addMapPoint(sptrNewMapPoint,i);
          sptrNewMapPoint->computeDistinctiveDescriptors();
          sptrNewMapPoint->updateNormalAndDepth();
          m_sptr_map->addMapPoint(sptrNewMapPoint);
          m_frame_currentFrame.m_v_sptrMapPoints[i] = sptrNewMapPoint;
        }
      }
      std::cout << "New map created with " << m_sptr_map->getMapPointsNum() << " points" << std::endl;
      m_sptr_localMapper->insertKeyFrame(sptrInitKeyFrame);
      m_frame_lastFrame = Frame(m_frame_currentFrame);
      m_int_lastKeyFrameID = m_frame_currentFrame.m_int_ID;
      m_sptr_lastKeyFrame = sptrInitKeyFrame;
      m_v_localKeyFrames.push_back(sptrInitKeyFrame);
      m_v_localMapPoints = m_sptr_map->getAllMapPoints();
      m_sptr_refKeyFrame = sptrInitKeyFrame;
      m_sptr_map->setReferenceMapPoints(m_v_localMapPoints);
      m_sptr_map->m_v_sptrOriginalKeyFrames.push_back(sptrInitKeyFrame);
      m_sptr_mapDrawer->setCurrentCameraPose(m_frame_currentFrame.m_cvMat_T_c2w);
      m_ts_state = TrackingState::OK;
    }
  }
  void Tracking::checkReplacementInLastFrame(){
    for(std::shared_ptr<MapPoint> &sptrMapPoint : m_frame_lastFrame.m_v_sptrMapPoints){
      if(sptrMapPoint && sptrMapPoint->getReplacement()){
        sptrMapPoint = sptrMapPoint->getReplacement();
      }
    }
  }
  bool Tracking::trackReferenceKeyFrame(){
    //compute bag of words vector
    m_frame_currentFrame.computeBoW();
    //firstly, orb matching with reference key frame
    //if enough matches are found, PnP solver will be setup
    OrbMatcher matcher(0.7,true);
    std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPoint;
    int refKeyFrame2currKeyFrameMatchNum = matcher.searchByBowInKeyFrameAndFrame(m_sptr_refKeyFrame,m_frame_currentFrame,vSptrMatchedMapPoint);
    if(refKeyFrame2currKeyFrameMatchNum>=15){
      m_frame_currentFrame.m_v_sptrMapPoints = vSptrMatchedMapPoint;
      m_frame_currentFrame.setCameraPoseByTransform_c2w(m_frame_lastFrame.m_cvMat_T_c2w);
      Optimizer::optimizePose(m_frame_currentFrame);
      //discard outliers
      int matchedMapPointsNum = 0;
      for(int i=0;i<m_frame_currentFrame.m_int_keyPointsNum;i++){
        if(m_frame_currentFrame.m_v_sptrMapPoints[i]){
          if(m_frame_currentFrame.m_v_isOutliers[i]){
            m_frame_currentFrame.m_v_sptrMapPoints[i]->m_b_isTrackInView = false;
            m_frame_currentFrame.m_v_sptrMapPoints[i]->m_int_lastSeenInFrameID = m_frame_currentFrame.m_int_ID;
            m_frame_currentFrame.m_v_isOutliers[i] = false;
            m_frame_currentFrame.m_v_sptrMapPoints[i] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
            refKeyFrame2currKeyFrameMatchNum--;
          }else if(m_frame_currentFrame.m_v_sptrMapPoints[i]->getObservationsNum()>0){
            matchedMapPointsNum++;
          }
        }
      }
      return matchedMapPointsNum>=10;
    }else{
      return false;
    }
  }
  void Tracking::updateLastFrame(){
    //update pose according to reference key frame
    m_frame_lastFrame.setCameraPoseByTransform_c2w(m_list_relFramePoses.back()*m_frame_lastFrame.m_sptr_refKeyFrame->getCameraPoseByTransrom_c2w());
    if(m_int_lastKeyFrameID==m_frame_lastFrame.m_int_ID || !m_b_isTrackingOnly || m_sys_sensor==System::Sensor::MONOCULAR){
      return;
    }
    //create visual odometry map points
    //points are sorted according to their depth from stereo/RGBD sensor
    std::vector<std::pair<float,int>> vDepthIdx;
    vDepthIdx.reserve(m_frame_lastFrame.m_int_keyPointsNum);
    for(int i=0;i<m_frame_lastFrame.m_int_keyPointsNum;i++){
      if(m_frame_lastFrame.m_v_depth[i]>0){
        vDepthIdx.push_back(std::make_pair(m_frame_lastFrame.m_v_depth[i],i));
      }
    }
    if(!vDepthIdx.empty()){
      std::sort(vDepthIdx.begin(),vDepthIdx.end());
      //insert all points that are closer than m_flt_depthThd
      //if num of points that are closer than m_flt_depthThd is less than 100, 100 closest points are added
      int pointsNum = 0;
      for(const std::pair<float,int> &pair : vDepthIdx){
        if(!m_frame_lastFrame.m_v_sptrMapPoints[pair.second] || m_frame_lastFrame.m_v_sptrMapPoints[pair.second]->getObservationsNum()<1){
          std::shared_ptr<MapPoint> sptrNewMapPoint = std::make_shared<MapPoint>(m_frame_lastFrame.inverseProject(pair.second),m_sptr_map,m_frame_lastFrame,pair.second);
          m_frame_lastFrame.m_v_sptrMapPoints[pair.second]=sptrNewMapPoint;
          m_list_tempMapPoints.push_back(sptrNewMapPoint);
          pointsNum++;
        }else{
          pointsNum++;
        }
        if(pair.first > m_flt_depthThd && pointsNum>100){
          break;
        }
      } 
    }
  }
  bool Tracking::trackWithMotionModel(){
    OrbMatcher matcher(0.9,true);
    //update last frame pose according to its reference key frame
    //create visual odometry points if in localization mode
    updateLastFrame();
    m_frame_currentFrame.setCameraPoseByTransform_c2w(m_cvMat_velocity*m_frame_lastFrame.m_cvMat_T_c2w);
    std::fill(m_frame_currentFrame.m_v_sptrMapPoints.begin(),m_frame_currentFrame.m_v_sptrMapPoints.end(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
    //project points seen in previous frame
    int thd = 7;
    if(m_sys_sensor==System::Sensor::STEREO){
      thd = 7;
    }else{
      thd = 15;
    }
    int lastframe2currentFrameMatchNum = matcher.searchByProjectionInLastAndCurrentFrame(m_frame_currentFrame,m_frame_lastFrame,thd);
    //if match num is few, wider window search is used
    if(lastframe2currentFrameMatchNum<20){
      std::fill(m_frame_currentFrame.m_v_sptrMapPoints.begin(),m_frame_currentFrame.m_v_sptrMapPoints.end(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
      lastframe2currentFrameMatchNum = matcher.searchByProjectionInLastAndCurrentFrame(m_frame_currentFrame,m_frame_lastFrame,2*thd);
    }
    if(lastframe2currentFrameMatchNum<20){
      return false;
    }
    //optimize frame pose with all matches
    Optimizer::optimizePose(m_frame_currentFrame);
    //discard outliers
    int matchedMapPointsNum = 0;
    for(int i=0;i<m_frame_currentFrame.m_int_keyPointsNum;i++){
      if(m_frame_currentFrame.m_v_sptrMapPoints[i]){
        if(m_frame_currentFrame.m_v_isOutliers[i]){
          m_frame_currentFrame.m_v_sptrMapPoints[i]->m_b_isTrackInView = false;
          m_frame_currentFrame.m_v_sptrMapPoints[i]->m_int_lastSeenInFrameID = m_frame_currentFrame.m_int_ID;
          m_frame_currentFrame.m_v_isOutliers[i] = false;
          m_frame_currentFrame.m_v_sptrMapPoints[i] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
          lastframe2currentFrameMatchNum--;
        }else if(m_frame_currentFrame.m_v_sptrMapPoints[i]->getObservationsNum()>0){
          matchedMapPointsNum++;
        }
      }
    }
    if(m_b_isTrackingOnly){
      m_b_isDoingVisualOdometry = matchedMapPointsNum < 10;
      return lastframe2currentFrameMatchNum > 20;
    }else{
      return matchedMapPointsNum >= 10;
    }
  }
  void Tracking::updateLocalMap(){
    //this is for visualization
    m_sptr_map->setReferenceMapPoints(m_v_localMapPoints);
    //update
    updateLocalKeyFrames();
    updateLocalPoints();
  }
  void Tracking::updateLocalPoints(){
    m_v_localMapPoints.clear();
    for(const std::shared_prt<KeyFrame> &sptrKeyFrame : m_v_localKeyFrames){
      for(const std::shared_prt<MapPoint> &sptrMapPoint : sptrKeyFrame->getMatchedMapPointsVec()){
        if(sptrMapPoint && sptrMapPoint->m_int_trackRefForFrameID!=m_frame_currentFrame.m_int_ID && !sptrMapPoint->isBad()){
          m_v_localMapPoints.push_back(sptrMapPoint);
          sptrMapPoint->m_int_trackRefForFrameID = m_frame_currentFrame.m_int_ID;
        }
      }
    }
  }
  void Tracking::updateLocalKeyFrames(){
    //each map point votes for the key frame in which it is observed
    std::map<std::shared_ptr<KeyFrame>, int> dicKeyFrameCounter;
    for(std::shared_ptr<MapPoint> &sptrMapPoint : m_frame_currentFrame.m_v_sptrMapPoints){
      if(sptrMapPoint){
        if(!sptrMapPoint->isBad()){
          for(const std::pair<const std::shared_ptr<KeyFrame>, int> &pair : sptrMapPoint->getObservations()){
            dicKeyFrameCounter[pair.first]++;
          }
        }else{
          sptrMapPoint = std::shared_ptr<MapPoint>(nullptr);
        }
      }
    }
    if(dicKeyFrameCounter.empty()){
      return;
    }
    int maxObservationNum = 0;
    std::shared_ptr<KeyFrame> mostObservationKeyFrame = static_cast<std::shared_ptr<KeyFrame>>(nullptr);
    m_v_localKeyFrames.clear();
    m_v_localKeyFrames.reserve(3*dicKeyFrameCounter.size());
    //all key frames that observe a map point are included in the local map
    //check which key frame contains most map points
    for(const std::pair<const std::shared_ptr<KeyFrame>, int> &pair : dicKeyFrameCounter){
      if(pair.first && !pair.first->isBad()){
        if(pair.second > maxObservationNum){
          maxObservationNum = pair.second;
          mostObservationKeyFrame = pair.first;
        }
        m_v_localKeyFrames.push_back(pair.first);
        pair.first->m_int_trackRefForFrameID = m_frame_currentFrame.m_int_ID;
      }
    }
    //include some not-yet-included key frames that are neighbors to already-included key frames
    for(const std::shared_prt<KeyFrame>& sptrKeyFrame : m_v_localKeyFrames){
      //limit the number of key frames
      if(m_v_localKeyFrames.size() > 80){
        break;
      }
      for(const std::shared_prt<KeyFrame> &connectedkeyFrame : sptrKeyFrame->getOrderedConnectedKeyFramesLargerThanWeight(10)){
        if(connectedkeyFrame && !connectedkeyFrame->isBad() && connectedkeyFrame->m_int_trackRefForFrameID!=m_frame_currentFrame.m_int_ID){
          m_v_localKeyFrames.push_back(connectedkeyFrame);
          connectedkeyFrame->m_int_trackRefForFrameID = m_frame_currentFrame.m_int_ID;
          break;
        }
      }
      for(const std::shared_prt<KeyFrame> &childKeyFrame : sptrKeyFrame->getChildren()){
        if(childKeyFrame && !childKeyFrame->isBad() && childKeyFrame->m_int_trackRefForFrameID!=m_frame_currentFrame.m_int_ID){
          m_v_localKeyFrames.push_back(childKeyFrame);
          childKeyFrame->m_int_trackRefForFrameID = m_frame_currentFrame.m_int_ID;
          break;
        }
      }
      if(sptrKeyFrame->getParent() && sptrKeyFrame->getParent()->m_int_trackRefForFrameID!=m_frame_currentFrame.m_int_ID){
        m_v_localKeyFrames.push_back(sptrKeyFrame->getParent());
        sptrKeyFrame->getParent()->m_int_trackRefForFrameID=m_frame_currentFrame.m_int_ID;
      }
    }
    if(mostObservationKeyFrame){
      m_sptr_refKeyFrame = mostObservationKeyFrame;
      m_frame_currentFrame.m_sptr_refKeyFrame = m_sptr_refKeyFrame;
    }
  }
  void Tracking::searchLocalPoints(){
    //do not search already matched map points
    for(std::shared_ptr<MapPoint> &sptrMapPoint : m_frame_currentFrame.m_v_sptrMapPoints){
      if(sptrMapPoint){
        if(sptrMapPoint->isBad()){
          sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
        }else{
          sptrMapPoint->increaseVisible();
          sptrMapPoint->m_int_lastSeenInFrameID = m_frame_currentFrame.m_int_ID;
          sptrMapPoint->m_b_isTrackInView = false;
        }
      }
    }
    int matchedMapPointsNum = 0;
    //project points in frame and check its visibility
    for(std::shared_ptr<MapPoint> &sptrMapPoint : m_v_localMapPoints){
      if(sptrMapPoint && sptrMapPoint->m_int_lastSeenInFrameID!=m_frame_currentFrame.m_int_ID && \
      !sptrMapPoint->isBad() && m_frame_currentFrame.isInCameraFrustum(sptrMapPoint,0.5)){
        sptrMapPoint->increaseVisible();
        matchedMapPointsNum++;
      }
    }
    if(matchedMapPointsNum>0){
      OrbMatcher matcher(0.8);
      int thd = 1;
      if(m_sys_sensor==System::Sensor::RGBD){
        thd = 3;
      }
      //if the camera is relocalized recently, perform a coarser search
      if(m_frame_currentFrame.m_int_ID<m_int_lastRelocalizedFrameID+2){
        thd = 5;
      }
      matcher.searchByProjectionInFrameAndMapPoint(m_frame_currentFrame,m_v_localMapPoints,thd);
    }
  }
  bool Tracking::trackLocalMap(){
    //stop here
  }
}//namespace YDORBSLAM

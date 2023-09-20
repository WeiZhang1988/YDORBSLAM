/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "tracking.hpp"
#include <opencv2/opencv.hpp>

namespace YDORBSLAM
{

Tracking::Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, const string &_strConfigurationPath, const int _sensor):\
m_enum_currentState(eTrackingState::NOT_INPUT_IMAGES), m_int_sensor(_sensor), m_b_onlyLocalization(false), m_b_fewMatched(false), m_sptr_orbVocabulary(_sptrVoc),\
m_sptr_keyFrameDB(_sptrKeyFrameDatabase), m_sptr_system(_sptrSys), m_sptr_viewer(nullptr), m_sptr_frameDrawer(_sptrFrameDrawer), m_sptr_mapDrawer(_sptrMapDrawer), m_sptr_map(_sptrMap), m_int_LastRelocFrameId(0){
    cv::FileStorage fileConfigs(_strConfigurationPath, cv::FileStorage::READ);
    cv::Mat camIntParMat = cv::Mat::eye(3,3,CV_32F);
    camIntParMat.at<float>(0,0) = fileConfigs["Camera.fx"];
    camIntParMat.at<float>(1,1) = fileConfigs["Camera.fy"];
    camIntParMat.at<float>(0,2) = fileConfigs["Camera.cx"];
    camIntParMat.at<float>(1,2) = fileConfigs["Camera.cy"];
    camIntParMat.copyTo(m_cvMat_intParMat);

    cv::Mat imageDistCoef(4,1,CV_32F);
    imageDistCoef.at<float>(0) = fileConfigs["Camera.k1"];
    imageDistCoef.at<float>(1) = fileConfigs["Camera.k2"];
    imageDistCoef.at<float>(2) = fileConfigs["Camera.p1"];
    imageDistCoef.at<float>(3) = fileConfigs["Camera.p2"];
    if(fileConfigs["Camera.k3"] != 0){
        imageDistCoef.resize(5);
        imageDistCoef.at<float>(4) = fileConfigs["Camera.k3"];
    }
    imageDistCoef.copyTo(m_cvMat_imageDistCoef);

    m_flt_baseLineTimesFx = fileConfigs["Camera.bf"];
    // Max/Min Frames to insert keyframes and to check relocalisation
    m_int_minFrequency = 0;
    if(fileConfigs["Camera.fps"] == 0){
        m_int_maxFrequency = 30;
    }else{
        m_int_maxFrequency = fileConfigs["Camera.fps"];
    }

    std::cout << endl << "Camera Parameters: " << std::endl;
    std::cout << "- fx: " << camIntParMat.at<float>(0,0) << std::endl;
    std::cout << "- fy: " << camIntParMat.at<float>(1,1) << std::endl;
    std::cout << "- cx: " << camIntParMat.at<float>(0,2) << std::endl;
    std::cout << "- cy: " << camIntParMat.at<float>(1,2) << std::endl;
    std::cout << "- k1: " << imageDistCoef.at<float>(0) << std::endl;
    std::cout << "- k2: " << imageDistCoef.at<float>(1) << std::endl;
    if(imageDistCoef.rows==5){
        std::cout << "- k3: " << imageDistCoef.at<float>(4) << std::endl;
    }
    std::cout << "- p1: " << imageDistCoef.at<float>(2) << std::endl;
    std::cout << "- p2: " << imageDistCoef.at<float>(3) << std::endl;
    std::cout << "- fps: " << m_int_maxFrequency << std::endl;
    m_b_isRGB = fileConfigs["Camera.RGB"];
    if(m_b_isRGB){
        std::cout << "- color order: RGB (ignored if grayscale)" << std::endl;
    }else{
        std::cout << "- color order: BGR (ignored if grayscale)" << std::endl;
    }

    int keyPointsNum = fileConfigs["ORBextractor.nFeatures"];
    float scaleFactor = fileConfigs["ORBextractor.scaleFactor"];
    int levelsNum = fileConfigs["ORBextractor.nLevels"];
    int initFastThd = fileConfigs["ORBextractor.iniThFAST"];
    int minFastThd = fileConfigs["ORBextractor.minThFAST"];

    m_sptr_leftOrbExtractor = make_shared<OrbExtractor>(keyPointsNum,scaleFactor,levelsNum,initFastThd,minFastThd);
    m_sptr_rightOrbExtractor = make_shared<OrbExtractor>(keyPointsNum,scaleFactor,levelsNum,initFastThd,minFastThd);
    
    std::cout << std::endl << "ORB Extractor Parameters: " << std::endl;
    std::cout << "- Number of Features: " << keyPointsNum << std::endl;
    std::cout << "- Scale Factor: " << scaleFactor << std::endl;
    std::cout << "- Scale Levels: " << levelsNum << std::endl;
    std::cout << "- Initial Fast Threshold: " << initFastThd << std::endl;
    std::cout << "- Minimum Fast Threshold: " << minFastThd << std::endl;

    if(_sensor==System::STEREO || _sensor==System::RGBD){
        m_flt_depthThd = m_flt_baseLineTimesFx * (float)fileConfigs["ThDepth"]/fileConfigs["Camera.fx"];
        std::cout << std::endl << "Depth Threshold (Close/Far Points): " << m_flt_depthThd << std::endl;
    }
    if(_sensor==System::RGBD){
        m_flt_depthMapFactor = fileConfigs["DepthMapFactor"];
        if(fabs(m_flt_depthMapFactor)<1e-5){
            m_flt_depthMapFactor = 1.0f;
        }else{
            m_flt_depthMapFactor = 1.0f/m_flt_depthMapFactor;
        }
    }

    void Tracking::setLocalMapper(std::shared_ptr<LocalMapping> _sptrLocalMapper)
    {
        m_sptr_localMapper=_sptrLocalMapper;
    }

    void Tracking::setLoopClosing(std::shared_ptr<LoopClosing> _sptrLoopClosing)
    {
        m_sptr_loopClosing = _sptrLoopClosing;
    }

    void Tracking::setViewer(std::shared_ptr<Viewer> _sptrViewer)
    {
        m_sptr_viewer = _sptrViewer;
    }

    cv::Mat Tracking::grabImageStereo(const cv::Mat &_leftImage, const cv::Mat &_rightImage, const double &_timeStamp){
        m_cvMat_grayImage = _leftImage;
        cv::Mat rightGrayImage = _rightImage;
        if(m_cvMat_grayImage.channels()==3 || m_cvMat_grayImage.channels()==4){
            if(m_b_isRGB){
                cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGB2GRAY);
                cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_RGB2GRAY);
            }else{
                cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGR2GRAY);
                cv::cvtColor(rightGrayImage,rightGrayImage,COLOR_BGR2GRAY);
            }
        }
        m_currentFrame = Frame(m_cvMat_grayImage,rightGrayImage,_timeStamp,m_cvMat_intParMat,m_cvMat_imageDistCoef,m_cvMat_intParMat,m_cvMat_imageDistCoef,\
        m_flt_baseLineTimesFx,m_flt_depthThd,m_sptr_leftOrbExtractor,m_sptr_rightOrbExtractor,m_sptr_orbVocabulary);
        Track();
        return m_currentFrame.m_cvMat_T_c2w.clone();
    }
    
    cv::Mat Tracking::grabImageRGBD(const cv::Mat &_rgbImage,const cv::Mat &_depthImage, const double &_timeStamp){
        m_cvMat_grayImage = _rgbImage;
        cv:Mat depthImage = _depthImage;
        if(m_cvMat_grayImage.channels()==3 || m_cvMat_grayImage.channels()==4){
            if(m_b_isRGB){
                cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_RGB2GRAY);
            }else{
                cv::cvtColor(m_cvMat_grayImage,m_cvMat_grayImage,COLOR_BGR2GRAY);
            }
        }
        if((fabs(m_flt_depthMapFactor-1.0f)>1e-5) || depthImage.type()!=CV_32F){
            depthImage.convertTo(depthImage,CV_32F,m_flt_depthMapFactor)
        }
        m_currentFrame = Frame(m_cvMat_grayImage,depthImage,_timeStamp,m_cvMat_intParMat,m_cvMat_imageDistCoef,\
        m_flt_baseLineTimesFx,m_flt_depthThd,m_sptr_leftOrbExtractor,m_sptr_orbVocabulary);
        Track();
        return m_currentFrame.m_cvMat_T_c2w.clone();
    }

    void Tracking::Track(){
        if(m_enum_currentState == eTrackingState::NOT_INPUT_IMAGES){
            m_enum_currentState = eTrackingState::NOT_INITIALIZED;
        }
        m_enum_lastState = m_enum_currentState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(m_sptr_map->m_mutex_map);
        if(m_enum_currentState == eTrackingState::NOT_INITIALIZED){
            if(m_int_sensor==System::STEREO || m_int_sensor==System::RGBD){
                stereoInitialization();
            }
            m_sptr_frameDrawer->update(shared_from_this()); //no define
            if(m_enum_currentState != eTrackingState::TRACKING_SUCCESS){
                return;
            }
        }else{
            // System is initialized. Track Frame.
            bool btrackSuccess;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if(!m_b_onlyLocalization){
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only localization" mode.
                if(m_enum_currentState == eTrackingState::TRACKING_SUCCESS){
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    checkReplacedMPInLastFrame();
                    if(!m_cvMat_velocity.empty() && m_currentFrame.m_int_ID>=m_int_lastRelocFrameId+2 && TrackWithMotionModel()){
                        btrackSuccess = true;
                    }else{
                        btrackSuccess = trackWithReferenceKeyFrame();
                    }
                }else{
                    btrackSuccess = relocalization();
                }
            }else{
                // Localization Mode: Local Mapping is deactivated
                if(m_enum_currentState == eTrackingState::TRACKING_FAIL){
                    btrackSuccess = relocalization();
                }else{
                    if(!m_b_fewMatched){
                        // In last frame we tracked enough MapPoints in the map
                        if(!m_cvMat_velocity.empty()){
                            btrackSuccess = TrackWithMotionModel();
                        }else{
                            btrackSuccess = trackWithReferenceKeyFrame();
                        }
                    }else{
                        // In last frame we tracked mainly "visual odometry" points.
                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution,
                        // otherwise we retain the "visual odometry" solution.
                        if(!m_cvMat_velocity.empty()){
                            if(TrackWithMotionModel() && !relocalization()){
                                m_currentFrame.setCameraPoseByTransform_c2w(m_currentFrame.m_cvMat_T_c2w.clone());
                                //???
                                if(m_b_fewMatched){
                                    int ifor = 0;
                                    for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
                                        if(sptrMapPoint && !m_currentFrame.m_v_isOutliers[ifor]){
                                            sptrMapPoint->increaseFound();
                                        }
                                        ifor++;
                                    }
                                }
                            }
                            else if(relocalization()){
                                m_b_fewMatched = false;
                            }
                        }
                        btrackSuccess = (relocalization() || TrackWithMotionModel());
                    }
                }
            }
            m_currentFrame.m_sptr_refKeyFrame = m_sptr_referenceKF;
            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(!m_b_onlyLocalization){
                if(btrackSuccess){
                    btrackSuccess = trackLocalMap();
                }
            }else{
                // m_b_fewMatched true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform trackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if(btrackSuccess && !m_b_fewMatched){
                    btrackSuccess = trackLocalMap();
                }
            }
            if(btrackSuccess){
                m_enum_currentState == eTrackingState::TRACKING_SUCCESS;
            }else{
                m_enum_currentState == eTrackingState::TRACKING_FAIL;
            }
            // Update drawer
            m_sptr_frameDrawer->update(shared_from_this()); //no define
            // If tracking were good, check if we insert a keyframe
            if(btrackSuccess){
                // Update motion model
                if(!m_lastFrame.m_cvMat_T_c2w.empty()){
                    cv::Mat cvMatLastTc2w = cv::Mat::eye(4,4,CV_32F);
                    m_lastFrame.getRotation_w2c().copyTo(cvMatLastTc2w.rowRange(0,3).colRange(0,3));
                    m_lastFrame.getCameraOriginInWorld().copyTo(cvMatLastTc2w.rowRange(0,3).col(3));
                    m_cvMat_velocity = m_currentFrame.m_cvMat_T_c2w * cvMatLastTc2w;
                }else{
                    m_cvMat_velocity = cv::Mat();
                }
                m_sptr_mapDrawer->SetCurrentCameraPose(m_currentFrame.m_cvMat_T_c2w); //no define
                // Clean current matched MapPoints
                int ifor = 0;
                for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
                    if(sptrMapPoint && sptrMapPoint->getObservationsNum() < 1){
                        m_currentFrame.m_v_isOutliers[ifor] = false;
                        sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    }
                    ifor++;
                }
                // Delete temporal MapPoints
                for(std::shared_ptr<MapPoint>& tempMapPoint : m_l_sptrTemporalMapPoints){
                    delete tempMapPoint;
                }
                m_l_sptrTemporalMapPoints.clear();
                // Check if we need to insert a new keyframe
                if(needNewKeyFrame()){
                    createNewKeyFrame();
                }
                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for(int i = 0; i < m_currentFrame.m_int_keyPointsNum; i++){
                    if(m_currentFrame.m_v_sptrMapPoints[i] && m_currentFrame.m_v_isOutliers[i]){
                        m_currentFrame.m_v_sptrMapPoints[i] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    }
                }
            }
            // Reset if the camera get lost soon after initialization
            if(m_enum_currentState == eTrackingState::TRACKING_FAIL && m_sptr_map->getKeyFramesNum()<=5){
                std::cout<< "Track lost soon after initialisation, reseting..." << std::endl;
                m_sptr_system->reset(); //no define
                return;
            }
            if(!m_currentFrame.m_sptr_refKeyFrame){
                m_currentFrame.m_sptr_refKeyFrame = m_sptr_referenceKF;
            }
            m_lastFrame = Frame(m_currentFrame);
        }
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(!m_currentFrame.m_cvMat_T_c2w.empty()){
            cv::Mat cvMatTcf2refkf = m_currentFrame.m_cvMat_T_c2w * m_currentFrame.m_sptr_refKeyFrame->getInverseCameraPoseByTransform_w2c();
            m_l_cvMatRelativeFramePoses.push_back(cvMatTcf2refkf);
            m_l_sptrRefKF.push_back(m_sptr_referenceKF);
            m_l_dframeTimes.push_back(m_currentFrame.m_d_timeStamp);
            m_l_btrackFail.push_back(m_enum_currentState == eTrackingState::TRACKING_FAIL);
        }else{
            // This can happen if tracking is fail
            m_l_cvMatRelativeFramePoses.push_back(m_l_cvMatRelativeFramePoses.back());
            m_l_sptrRefKF.push_back(m_l_sptrRefKF.back());
            m_l_dframeTimes.push_back(m_l_dframeTimes.back());
            m_l_btrackFail.push_back(m_enum_currentState == eTrackingState::TRACKING_FAIL);
        }
    }

    void Tracking::stereoInitialization(){
        if(m_currentFrame.m_int_keyPointsNum > 500){
            // Set Frame pose to the origin
            m_currentFrame.setCameraPoseByTransform_c2w(cv::Mat::eye(4,4,CV_32F));
            // Create KeyFrame
            std::shared_ptr<KeyFrame> sptrIniKF = std::make_shared<KeyFrame>(m_currentFrame,m_sptr_map,m_sptr_keyFrameDB);
            // Insert KeyFrame in the map
            m_sptr_map->addKeyFrame(sptrIniKF);
            // Create MapPoints and asscoiate to KeyFrame
            for(int i = 0; i < m_currentFrame.m_int_keyPointsNum; i++){
                if(m_currentFrame.m_v_depth[i] > 0){
                    std::shared_ptr<MapPoint> sptrNewMP = std::make_shared<MapPoint>(m_currentFrame.inverseProject(i),m_sptr_map,sptrIniKF);
                    sptrNewMP->addObservation(sptrIniKF,i);
                    sptrIniKF->addMapPoint(sptrNewMP,i);
                    sptrNewMP->computeDistinctiveDescriptors();
                    sptrNewMP->updateNormalAndDepth();
                    m_sptr_map->addMapPoint(sptrNewMP);
                    m_currentFrame.m_v_sptrMapPoints[i] = sptrNewMP;
                }
            }
            std::cout << "New map created with " << m_sptr_map->getMapPointsNum() << " points" << std::endl;
            m_sptr_localMapper->InsertKeyFrame(sptrIniKF); //no define
            m_lastFrame = Frame(m_currentFrame);
            m_int_lastKeyFrameId = m_currentFrame.m_int_ID;
            m_sptr_lastKeyFrame = sptrIniKF;
            m_v_sptrLocalKeyFrames.push_back(sptrIniKF);
            m_v_sptrLocalMapPoints = m_sptr_map->getAllMapPoints();
            m_sptr_referenceKF = sptrIniKF;
            m_sptr_map->setReferenceMapPoints(m_v_sptrLocalMapPoints);
            m_sptr_map->m_v_sptrOriginalKeyFrames.push_back(sptrIniKF);
            m_sptr_mapDrawer->SetCurrentCameraPose(m_currentFrame.m_cvMat_T_c2w); //no define
            m_enum_currentState = eTrackingState::TRACKING_SUCCESS;
        }
    }

    void Tracking::checkReplacedMPInLastFrame(){
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_lastFrame.m_v_sptrMapPoints){
            if(sptrMapPoint && sptrMapPoint->getReplacement()){
                sptrMapPoint = sptrMapPoint->getReplacement();
            }
        }
    }

    bool Tracking::trackWithReferenceKeyFrame(){
        // Compute Bag of Words vector
        m_currentFrame.computeBoW();
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        OrbMatcher matcher(0.7,true);
        std::vector<std::shared_ptr<MapPoint>> vsptrMatchedMapPoint;
        int nRefKF2CFmatches = matcher.searchByBowInKeyFrameAndFrame(m_sptr_referenceKF,m_currentFrame,vsptrMatchedMapPoint);
        if(nRefKF2CFmatches < 15){
            return false;
        }
        m_currentFrame.m_v_sptrMapPoints = vsptrMatchedMapPoint;
        m_currentFrame.setCameraPoseByTransform_c2w(m_lastFrame.m_cvMat_T_c2w);
        Optimizer::optimizePose(m_currentFrame);
        // Discard outliers
        int nTrackmatches = 0;
        int ifor = 0;
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
            if(sptrMapPoint){
                if(m_currentFrame.m_v_isOutliers[ifor]){
                    sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    m_currentFrame.m_v_isOutliers[ifor] = false;
                    sptrMapPoint->m_b_isTrackInView = false;
                    sptrMapPoint->m_int_lastSeenInFrameID = m_currentFrame.m_int_ID;
                    nlF2cFmatches--;
                }
                else if(sptrMapPoint->getObservationsNum() > 0){
                    nTrackmatches++;
                }
            }
            ifor++;
        }
        return nTrackmatches >= 10;
    }

    void Tracking::updateLastFrame(){
        // Update pose according to reference keyframe
        m_lastFrame.setCameraPoseByTransform_c2w(m_l_cvMatRelativeFramePoses.back() * m_lastFrame.m_sptr_refKeyFrame->getCameraPoseByTransrom_c2w());
        if(m_int_lastKeyFrameId==m_lastFrame.m_int_ID || m_int_sensor==System::MONOCULAR || !m_b_onlyLocalization){
            return;
        }
        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        std::vector<std::pair<float,int> > vpairDepthIdx;
        vpairDepthIdx.reserve(m_lastFrame.m_int_keyPointsNum);
        int ifor = 0;
        for(const float& z : m_lastFrame.m_v_depth){
            if(z > 0){
                vpairDepthIdx.push_back(std::make_pair(z,ifor));
            }
            ifor++;
        }
        if(vpairDepthIdx.empty()){
            return;
        }
        sort(vpairDepthIdx.begin(),vpairDepthIdx.end());
        // We insert all close points (depth < m_flt_depthThd)
        // If less than 100 close points, we insert the 100 closest ones.
        int ntemporalMapPointNUMS = 0;
        for(const std::pair<float,int>& pDepthIdx : vpairDepthIdx){
            std::shared_ptr<MapPoint> sptrMP = m_lastFrame.m_v_sptrMapPoints[pDepthIdx.second];
            if(!sptrMP || sptrMP->getObservationsNum()<1){
                std::shared_ptr<MapPoint> sptrNewMP = std::make_shared<MapPoint>(m_lastFrame.inverseProject(pDepthIdx.second),m_sptr_map,m_lastFrame,pDepthIdx.second);
                m_lastFrame.m_v_sptrMapPoints[pDepthIdx.second] = sptrNewMP;
                m_l_sptrTemporalMapPoints.push_back(sptrNewMP);
                ntemporalMapPointNUMS++;
            }else{
                ntemporalMapPointNUMS++;
            }
            if(pDepthIdx.first > m_flt_depthThd && ntemporalMapPointNUMS > 100){
                break;
            }
        }
    }

    bool Tracking::trackWithMotionModel(){
        OrbMatcher matcher(0.9,true);
        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        updateLastFrame();
        m_currentFrame.setCameraPoseByTransform_c2w(m_cvMat_velocity*m_lastFrame.m_cvMat_T_c2w);
        fill(m_currentFrame.m_v_sptrMapPoints.begin(),m_currentFrame.m_v_sptrMapPoints.end(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
        // Project points seen in previous frame
        int th;
        if(m_int_sensor==System::STEREO){
            th = 7;
        }else{
            th = 15;
        }
        int nlF2cFmatches = matcher.SearchByProjection(m_currentFrame,m_lastFrame,th);
        // If few matches, uses a wider window search
        if(nlF2cFmatches < 20){
            fill(m_currentFrame.m_v_sptrMapPoints.begin(),m_currentFrame.m_v_sptrMapPoints.end(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
            nlF2cFmatches = matcher.SearchByProjection(m_currentFrame,m_lastFrame,2*th);
        }
        if(nlF2cFmatches < 20){
            return false;
        }
        // Optimize frame pose with all matches
        Optimizer::optimizePose(m_currentFrame);
        // Discard outliers
        int nTrackmatches = 0;
        int ifor = 0;
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
            if(sptrMapPoint){
                if(m_currentFrame.m_v_isOutliers[ifor]){
                    sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    m_currentFrame.m_v_isOutliers[ifor] = false;
                    sptrMapPoint->m_b_isTrackInView = false;
                    sptrMapPoint->m_int_lastSeenInFrameID = m_currentFrame.m_int_ID;
                    nlF2cFmatches--;
                }
                else if(sptrMapPoint->getObservationsNum() > 0){
                    nTrackmatches++;
                }
            }
            ifor++;
        }
        if(m_b_onlyLocalization){
            m_b_fewMatched = nTrackmatches < 10;
            return nlF2cFmatches > 20;
        }
        return nTrackmatches >= 10;
    }

    bool Tracking::trackLocalMap(){
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        updateLocalMap();
        searchLocalPoints();
        // Optimize Pose
        Optimizer::optimizePose(m_currentFrame);
        m_int_currentMatchedMapPoints = 0;
        // Update MapPoints Statistics
        int ifor = 0;
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
            if(sptrMapPoint){
                if(!m_currentFrame.m_v_isOutliers[ifor]){
                    sptrMapPoint->increaseFound();
                    if(!m_b_onlyLocalization){
                        if(sptrMapPoint->getObservationsNum() > 0){
                            m_int_currentMatchedMapPoints++;
                        }
                    }else{
                        m_int_currentMatchedMapPoints++;
                    }
                }
                else if(m_int_sensor==System::STEREO){
                    sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                }
            }
            ifor++;
        }
        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if(m_currentFrame.m_int_ID<m_int_lastRelocFrameId+m_int_maxFrequency && m_int_currentMatchedMapPoints<50){
            return false;
        }
        if(m_int_currentMatchedMapPoints < 30){
            return false;
        }else{
            return true;
        }
    }

    void Tracking::searchLocalPoints(){
        // Do not search map points already matched
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
            if(sptrMapPoint){
                if(sptrMapPoint->isBad()){
                    sptrMapPoint = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                }else{
                    sptrMapPoint->increaseVisible();
                    sptrMapPoint->m_int_lastSeenInFrameID = m_currentFrame.m_int_ID;
                    sptrMapPoint->m_b_isTrackInView = false;
                }
            }
        }
        int nMatched = 0;
        // Project points in frame and check its visibility
        for(std::shared_ptr<MapPoint>& sptrLocalMapPoint : m_v_sptrLocalMapPoints){
            if(sptrLocalMapPoint->m_int_lastSeenInFrameID != m_currentFrame.m_int_ID &&\
            !sptrLocalMapPoint->isBad() && m_currentFrame.isInCameraFrustum(sptrLocalMapPoint,0.5)){
                // Project (this fills MapPoint variables for matching)
                sptrLocalMapPoint->increaseVisible();
                nMatched++;
            }
        }
        if(nMatched>0){
            OrbMatcher matcher(0.8);
            int th = 1;
            if(m_int_sensor==System::RGBD){
                th = 3;
            }
            // If the camera has been relocalised recently, perform a coarser search
            if(m_currentFrame.m_int_ID < m_int_lastRelocFrameId + 2){
                th = 5;
            }
            matcher.searchByProjectionInFrameAndMapPoint(m_currentFrame,m_v_sptrLocalMapPoints,th);
        }
    }

    void Tracking::updateLocalMap(){
        // This is for visualization
        m_sptr_map->setReferenceMapPoints(m_v_sptrLocalMapPoints);

        // Update
        updateLocalKeyFrames();
        updateLocalPoints();
    }

    void Tracking::updateLocalKeyFrames(){
        // Each map point vote for the keyframes in which it has been observed
        std::map<std::shared_ptr<KeyFrame>, int> mapKeyFrameCounter;
        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
            if(sptrMapPoint){
                if(!sptrMapPoint->isBad()){
                    for(const std::pair<const std::shared_ptr<KeyFrame>, int>& pObservation : sptrMapPoint->getObservations()){
                        mapKeyFrameCounter[pObservation.first]++;
                    }
                }else{
                    sptrMapPoint = nullptr;
                }
            }
        }
        if(mapKeyFrameCounter.empty()){
            return;
        }
        int maxObervationNums = 0;
        std::shared_ptr<KeyFrame> maxObervationSptrKF = static_cast<std::shared_ptr<KeyFrame>>(nullptr);
        m_v_sptrLocalKeyFrames.clear();
        m_v_sptrLocalKeyFrames.reserve(3*mapKeyFrameCounter.size());
        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for(const std::pair<const std::shared_ptr<KeyFrame>, int>& keyframeCouter : mapKeyFrameCounter){
            if(!keyframeCouter.first->isBad()){
                if(keyframeCouter.second > maxObervationNums){
                    maxObervationNums = keyframeCouter.second;
                    maxObervationSptrKF = keyframeCouter.first;
                }
                m_v_sptrLocalKeyFrames.push_back(keyframeCouter.first);
                keyframeCouter.first->m_int_trackRefForFrameID = m_currentFrame.m_int_ID;
            }
        }
        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for(const std::shared_prt<KeyFrame>& sptrLocalKeyFrame : m_v_sptrLocalKeyFrames){
            // Limit the number of keyframes
            if(m_v_sptrLocalKeyFrames.size() > 80){
                break;
            }
            for(const std::shared_prt<KeyFrame>& weightKeyFrame : sptrLocalKeyFrame->getOrderedConnectedKeyFramesLargerThanWeight(10)){
                if(!weightKeyFrame->isBad() && weightKeyFrame->m_int_trackRefForFrameID!=m_currentFrame.m_int_ID){
                    m_v_sptrLocalKeyFrames.push_back(weightKeyFrame);
                    weightKeyFrame->m_int_trackRefForFrameID = m_currentFrame.m_int_ID;
                    break;
                }
            }
            for(const std::shared_prt<KeyFrame>& childKeyFrame : sptrLocalKeyFrame->getChildren()){
                if(!childKeyFrame->isBad() && childKeyFrame->m_int_trackRefForFrameID!=m_currentFrame.m_int_ID){
                    m_v_sptrLocalKeyFrames.push_back(childKeyFrame);
                    childKeyFrame->m_int_trackRefForFrameID = m_currentFrame.m_int_ID;
                    break;
                }
            }
            if(sptrLocalKeyFrame->getParent() && sptrLocalKeyFrame->getParent()->m_int_trackRefForFrameID!=m_currentFrame.m_int_ID){
                m_v_sptrLocalKeyFrames.push_back(sptrLocalKeyFrame->getParent());
                sptrLocalKeyFrame->getParent()->m_int_trackRefForFrameID = m_currentFrame.m_int_ID;
                break;
            }
        }
        if(maxObervationSptrKF){
            m_sptr_referenceKF = maxObervationSptrKF;
            m_currentFrame.m_sptr_refKeyFrame = m_sptr_referenceKF;
        }
    }

    void Tracking::updateLocalPoints(){
        m_v_sptrLocalMapPoints.clear();
        for(const std::shared_prt<KeyFrame>& sptrLocalKeyFrame : m_v_sptrLocalKeyFrames){
            for(const std::shared_prt<MapPoint>& sptrMapPoint : sptrLocalKeyFrame->getMatchedMapPointsVec()){
                if(sptrMapPoint && sptrMapPoint->m_int_trackRefForFrameID!=m_currentFrame.m_int_ID && !pMP->isBad()){
                    m_v_sptrLocalMapPoints.push_back(sptrMapPoint);
                    sptrMapPoint->m_int_trackRefForFrameID=m_currentFrame.m_int_ID;
                }
            }
        }
    }

    bool Tracking::relocalization(){
        // Compute Bag of Words Vector
        m_currentFrame.computeBoW();
        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        std::vector<std::shared_prt<KeyFrame>> vsptrCandidateKFs = m_sptr_keyFrameDB->detectRelocalizationCandidates(m_currentFrame);
        if(vsptrCandidateKFs.empty()){
            return false;
        }
        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        OrbMatcher matcher(0.75,true);
        std::vector<std::shared_ptr<PnPsolver>> vsptrPnPsolvers; //no define
        vsptrPnPsolvers.resize(vsptrCandidateKFs.size());
        std::vector<std::vector<std::shared_ptr<MapPoint>>> vvsptrMatchedMapPoints;
        vvsptrMatchedMapPoints.resize(vsptrCandidateKFs.size());
        std::vector<bool> vbDiscarded;
        vbDiscarded.resize(vsptrCandidateKFs.size());
        int nfinalCandidates = 0;
        for(int i = 0; i < vsptrCandidateKFs.size(); i++){
            if(vsptrCandidateKFs[i]->isBad()){
                vbDiscarded[i] = true;
            }else{
                int nKF2CFmatches = matcher.searchByBowInKeyFrameAndFrame(vsptrCandidateKFs[i],m_currentFrame,vvsptrMatchedMapPoints[i]);
                if(nKF2CFmatches < 15){
                    vbDiscarded[i] = true;
                    continue;
                }else{
                    std::shared_ptr<PnPsolver> pnpSolver = std::make_shared<PnPsolver>(m_currentFrame,vvsptrMatchedMapPoints[i]); //no define
                    pnpSolver->setRansacParameters(0.99,10,300,4,0.5,5.991); //no define
                    vsptrPnPsolvers[i] = pnpSolver;
                    nfinalCandidates++;
                }
            }
        }
        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        OrbMatcher matcher2(0.9,true);
        int nBAinlierNums = 0;
        while(nfinalCandidates > 0 && nBAinlierNums < 50){
            for(int i = 0; i < vsptrCandidateKFs.size(); i++){
                if(vbDiscarded[i]){
                    continue;
                }
                // Perform 5 Ransac Iterations
                std::vector<bool> vbIsInlier;
                int nInlierNums;
                bool bIsOverIterationNum;
                std::shared_ptr<PnPsolver> pSolver = vsptrPnPsolvers[i];
                cv::Mat cvMatTc2w = pSolver->iterate(5,bIsOverIterationNum,vbIsInlier,nInlierNums); //no define
                //If Ransac over max iteration nums, discard keyframe
                if(bIsOverIterationNum){
                    vbDiscarded[i] = true;
                    nfinalCandidates--;
                }
                // Optimize a Camera Pose that is computed
                if(cvMatTc2w.empty()){
                    continue;
                }
                cvMatTc2w.copyTo(m_currentFrame.m_cvMat_T_c2w);
                std::set<std::shared_ptr<MapPoint>> sFoundsptrMapPoint;
                for(int j=0; j<vbIsInlier.size(); j++){
                    if(vbIsInlier[j]){
                        m_currentFrame.m_v_sptrMapPoints[j] = vvsptrMatchedMapPoints[i][j];
                        sFoundsptrMapPoint.insert(vvsptrMatchedMapPoints[i][j]);
                    }else{
                        m_currentFrame.m_v_sptrMapPoints[j] = nullptr;
                    }
                }
                ninlierNums = Optimizer::optimizePose(m_currentFrame);
                if(nBAinlierNums < 10){
                    continue;
                }
                for(int iout=0; iout<m_currentFrame.m_int_keyPointsNum; iout++){
                    if(m_currentFrame.m_v_isOutliers[iout]){
                        m_currentFrame.m_v_sptrMapPoints[iout] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    }
                }
                // If few inliers, search by projection in a coarse window and optimize again
                int nKF2CFaddNums = matcher2.searchByProjectionInKeyFrameAndCurrentFrame(m_currentFrame,vsptrCandidateKFs[i],sFoundsptrMapPoint,10,100);
                if(nBAinlierNums<50 && nBAinlierNums+nKF2CFaddNums>=50){
                    nBAinlierNums = Optimizer::optimizePose(m_currentFrame);
                    // If many inliers but still not enough, search by projection again in a narrower window
                    // the camera has been already optimized with many points
                    if(nBAinlierNums>30 && nBAinlierNums<50){
                        sFoundsptrMapPoint.clear();
                        for(std::shared_ptr<MapPoint>& sptrMapPoint : m_currentFrame.m_v_sptrMapPoints){
                            if(sptrMapPoint){
                                sFoundsptrMapPoint.insert(sptrMapPoint);
                            }
                        }
                        nKF2CFaddNums = matcher2.searchByProjectionInKeyFrameAndCurrentFrame(m_currentFrame,vsptrCandidateKFs[i],sFoundsptrMapPoint,3,64);
                        // Final optimization
                        if(nBAinlierNums+nKF2CFaddNums>=50){
                            nBAinlierNums = Optimizer::optimizePose(m_currentFrame);
                            for(int iout=0; iout<m_currentFrame.m_int_keyPointsNum; iout++){
                                if(m_currentFrame.m_v_isOutliers[iout]){
                                    m_currentFrame.m_v_sptrMapPoints[iout] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                                }
                            }
                        }
                    }
                }
                // If the pose is supported by enough inliers stop ransacs and continue
                if(nBAinlierNums>=50){
                    break;
                }
            }
        }
        if(nBAinlierNums<50){
            return false;
        }else{
            m_int_lastRelocFrameId = m_currentFrame.m_int_ID;
            return true;
        }
    }

    bool Tracking::needNewKeyFrame(){
        // If Only Localization,
        // If Local Mapping is freezed by a Loop Closure,
        // If current frame is near the last relocalization,
        // do not product keyframes
        bool bisRelocFrameNeabor = m_currentFrame.m_int_ID<m_int_lastRelocFrameId+m_int_maxFrequency;
        if(m_b_onlyLocalization || m_sptr_localMapper->isStopped() || m_sptr_localMapper->stopRequested() \
        || (bisRelocFrameNeabor && m_sptr_map->getKeyFramesNum()>m_int_maxFrequency)){
            return false;
        } //no define
        // Check how many "close"(0<m_v_depth<m_flt_depthThd) points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose= 0;
        if(m_int_sensor!=System::MONOCULAR){
            for(int i=0; i<m_currentFrame.m_int_keyPointsNum; i++){
                if(m_currentFrame.m_v_depth[i]>0 && m_currentFrame.m_v_depth<m_flt_depthThd){
                    if(m_currentFrame.m_v_sptrMapPoints[i] && !m_currentFrame.m_v_isOutliers[i]){
                        nTrackedClose++;
                    }else{
                        nNonTrackedClose++;
                    }
                }
            }
        }
        // Few Matched between MapPoints and close KeyPoints, and many close KeyPoints no matches MapPoints
        bool bneedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);
        // Thresholds
        float thRefMatchRatio = 0.75f;
        if(m_sptr_map->getKeyFramesNum() < 2){
            thRefMatchRatio = 0.4f;
        }
        if(m_int_sensor==System::MONOCULAR){
            thRefMatchRatio = 0.9f;
        }
        // Local Mapping could accept keyframes?
        bool blocalMappingIdle = m_sptr_localMapper->AcceptKeyFrames(); //no define
        // Tracked MapPoints in the reference keyframe
        int nrefMatches = m_sptr_referenceKF->trackedMapPointsNum(3);
        if(m_sptr_map->getKeyFramesNum() <= 2){
            nrefMatches = m_sptr_referenceKF->trackedMapPointsNum(2);
        }
        // Condition bnotNearLastKeyFrame: The current frame is not near the last KeyFrame
        const bool bnotNearLastKeyFrame = m_currentFrame.m_int_ID >= m_int_lastKeyFrameId+m_int_maxFrequency;
        // Condition bnextFrameNeedLocalMapping: The current frame is near the last KeyFrame and Local Mapping is idle
        const bool bnextFrameNeedLocalMapping = (mCurrentFrame.mnId>=m_int_lastKeyFrameId+m_int_minFrequency && blocalMappingIdle);
        // Condition blowSimilarityWithRefKF: Fewer tracked points than reference keyframe, or Current Frame tracked few MapPoints
        const bool blowSimilarityWithRefKF =  m_int_sensor!=System::MONOCULAR && (m_int_currentMatchedMapPoints<nrefMatches*0.25 || bneedToInsertClose);
        // Condition bcurrentFrameMatchedEnough: Few tracked points compared to reference keyframe, and Matched MapPoints NUM more than 15;
        const bool bcurrentFrameMatchedEnough =  ((m_int_currentMatchedMapPoints<nrefMatches*thRefMatchRatio || bneedToInsertClose) && m_int_currentMatchedMapPoints>15);
        if((bnotNearLastKeyFrame||bnextFrameNeedLocalMapping||blowSimilarityWithRefKF)&&bcurrentFrameMatchedEnough){
            // If the LocalMapping accepts keyframes, insert keyframe
            // Otherwise send a signal to interrupt BA
            if(blocalMappingIdle){
                return true;
            }else{
                m_sptr_localMapper->InterruptBA(); //no define
                if(m_int_sensor==System::MONOCULAR){
                    if(m_sptr_localMapper->KeyframesInQueue()<3){
                        return true;
                    }else{
                        return false;
                    }
                }else{
                    return false;
                }
            }
        }else{
            return false;
        }
    }

    void Tracking::createNewKeyFrame(){
        if(!m_sptr_localMapper->SetNotStop(true)){
            return;
        } //no define
        std::shared_ptr<KeyFrame> sptrNewKF = std::make_shared<KeyFrame>(m_currentFrame,m_sptr_map,m_sptr_keyFrameDB);
        m_sptr_referenceKF = sptrNewKF;
        m_currentFrame.m_sptr_refKeyFrame = sptrNewKF;
        if(m_int_sensor!=System::MONOCULAR){
            m_currentFrame.updatePoseMatrices();
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            std::vector<std::pair<float,int>> vDepthIdx;
            vDepthIdx.reserve(m_currentFrame.m_int_keyPointsNum);
            int ifor = 0;
            for(float& z : m_currentFrame.m_v_depth){
                if(z > 0){
                    vDepthIdx.push_back(std::make_pair(z,ifor));
                }
                ifor++;
            }
            if(!vDepthIdx.empty()){
                sort(vDepthIdx.begin(),vDepthIdx.end());
                int nNewPoints = 0;
                for(const pair<float,int>& depthIdx : vDepthIdx){
                    if(m_currentFrame.m_v_sptrMapPoints[depthIdx.second]->getObservationsNum()<1){
                        m_currentFrame.m_v_sptrMapPoints[depthIdx.second] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
                    }
                    if(!m_currentFrame.m_v_sptrMapPoints[depthIdx.second] \
                    || m_currentFrame.m_v_sptrMapPoints[depthIdx.second]->getObservationsNum()<1){
                        std::shared_ptr<MapPoint> sptrNewMP = std::make_shared<MapPoint>(m_currentFrame.inverseProject(depthIdx.second),m_sptr_map,sptrNewKF);
                        sptrNewMP->addObservation(sptrNewKF,depthIdx.second);
                        sptrNewKF->addMapPoint(sptrNewMP,depthIdx.second);
                        sptrNewMP->computeDistinctiveDescriptors();
                        sptrNewMP->updateNormalAndDepth();
                        m_sptr_map->addMapPoint(sptrNewMP);
                        m_currentFrame.m_v_sptrMapPoints[i] = sptrNewMP;
                        nNewPoints++;
                    }else{
                        nNewPoints++;
                    }
                    if(depthIdx.first>m_flt_depthThd && nNewPoints>100){
                        break;
                    }
                }
            }
        }
        m_sptr_localMapper->InsertKeyFrame(sptrNewKF); //no define
        m_sptr_localMapper->SetNotStop(false); //no define
        m_int_lastKeyFrameId = m_currentFrame.m_int_ID;
        m_sptr_lastKeyFrame = sptrNewKF;
    }

    void Tracking::reset(){
        std::cout<< "System Reseting" << std::endl;
        if(m_sptr_viewer){
            m_sptr_viewer->RequestStop(); //no define
            while(!m_sptr_viewer->isStopped()){
                std::usleep(3000);
            }
        }
        // Reset Local Mapping
        std::cout << "Reseting Local Mapper...";
        m_sptr_localMapper->RequestReset(); //no define
        std::cout << " done" << std::endl;
        // Reset Loop Closing
        std::cout << "Reseting Loop Closing...";
        m_sptr_loopClosing->RequestReset(); //no define
        std::cout << " done" << std::endl;
        // Clear BoW Database
        std::cout << "Reseting Database...";
        m_sptr_keyFrameDB->clear(); //no define
        std::cout << " done" << std::endl;

        // Clear Map (this erase MapPoints and KeyFrames)
        m_sptr_map->clear();
        KeyFrame::m_int_reservedKeyFrameID = 0;
        Frame::m_int_reservedID = 0;
        m_enum_currentState = eTrackingState::NOT_INPUT_IMAGES;
        m_l_cvMatRelativeFramePoses.clear();
        m_l_sptrRefKF.clear();
        m_l_dframeTimes.clear();
        m_l_btrackFail.clear();
        if(m_sptr_viewer){
            m_sptr_viewer->Release(); //no define
        }
    }

    void Tracking::changeIntrinsics(const string &_strConfigurationPath){
        cv::FileStorage fileConfigs(_strConfigurationPath, cv::FileStorage::READ);
        cv::Mat camIntParMat = cv::Mat::eye(3,3,CV_32F);
        camIntParMat.at<float>(0,0) = fileConfigs["Camera.fx"];
        camIntParMat.at<float>(1,1) = fileConfigs["Camera.fy"];
        camIntParMat.at<float>(0,2) = fileConfigs["Camera.cx"];
        camIntParMat.at<float>(1,2) = fileConfigs["Camera.cy"];
        camIntParMat.copyTo(m_cvMat_intParMat);

        cv::Mat imageDistCoef(4,1,CV_32F);
        imageDistCoef.at<float>(0) = fileConfigs["Camera.k1"];
        imageDistCoef.at<float>(1) = fileConfigs["Camera.k2"];
        imageDistCoef.at<float>(2) = fileConfigs["Camera.p1"];
        imageDistCoef.at<float>(3) = fileConfigs["Camera.p2"];
        if(fileConfigs["Camera.k3"] != 0){
            imageDistCoef.resize(5);
            imageDistCoef.at<float>(4) = fileConfigs["Camera.k3"];
        }
        imageDistCoef.copyTo(m_cvMat_imageDistCoef);

        m_flt_baseLineTimesFx = fileConfigs["Camera.bf"];
        Frame::m_b_isComputeInit = true;
    }

    void Tracking::onlyLocalization(const bool &_flag){
        m_b_onlyLocalization = _flag;
    }
}

} //namespace YDORBSLAM
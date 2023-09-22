/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/
#include "system.hpp"
#include "converter.hpp"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
#include <memory>

namespace YDORBSLAM{
System::System(const std::string& _strVocFile, const std::string& _strConfigurationPath, const eSensor& _enumSensor, const bool& _bUseViewer):\
                m_enum_sensor(_enumSensor),m_sptr_viewer(static_cast<std::shared_ptr<Viewer>>(nullptr)),m_b_reset(false),\
                m_b_activateLocalizationMode(false),m_b_deactivateLocalizationMode(false){
    // Output welcome message
    std::cout << "Input sensor was set to: ";
    if(m_enum_sensor == STEREO){
        std::cout << "Stereo" << std::endl;
    }
    else if(m_enum_sensor == RGBD){
        std::cout << "RGB-D" << std::endl;
    }
    //Check settings file
    cv::FileStorage fileConfigs(_strConfigurationPath.c_str(), cv::FileStorage::READ); //?
    if(!fileConfigs.isOpened()){
        std::cerr << "Failed to open settings file at: " << _strConfigurationPath << std::endl;
        exit(-1);
    }
    //Load ORB Vocabulary
    std::cout << std::endl << "Loading ORB Vocabulary. This could take a while..." << std::endl;
    m_sptr_vocabulary = std::make_shared<std::shared_ptr<DBoW3::Vocabulary>>();
    m_sptr_vocabulary->load(_strVocFile);
    std::cout << "Vocabulary loaded!" << std::endl << std::endl;
    //Create KeyFrame Database
    m_sptr_keyFrameDatabase = std::make_shared<KeyFrameDatabase>(*m_sptr_vocabulary);
    //Create the Map
    m_sptr_map = std::make_shared<Map>();
    //Create Drawers. These are used by the Viewer
    m_sptr_frameDrawer = std::make_shared<FrameDrawer>(m_sptr_map);
    m_sptr_mapDrawer = std::make_shared<MapDrawer>(m_sptr_map, _strConfigurationPath);
    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    m_sptr_tracker = std::make_shared<Tracking>(shared_from_this(),m_sptr_vocabulary,m_sptr_frameDrawer,m_sptr_mapDrawer,\
                                                m_sptr_map,m_sptr_keyFrameDatabase,_strConfigurationPath,static_cast<int>(m_enum_sensor));
    //Initialize the Local Mapping thread and launch
    m_sptr_localMapper = std::make_shared<LocalMapping>(m_sptr_map);
    m_sptr_threadLocalMapping = std::make_shared<thread>(&YDORBSLAM::LocalMapping::run,m_sptr_localMapper);
    //Initialize the Loop Closing thread and launch
    m_sptr_loopCloser = std::make_shared<LoopClosing>(m_sptr_map,m_sptr_keyFrameDatabase,m_sptr_vocabulary);
    m_sptr_threadLoopClosing = std::make_shared<thread>(&YDORBSLAM::LoopClosing::run,m_sptr_loopCloser);
    //Initialize the Viewer thread and launch
    if(_bUseViewer){
        m_sptr_viewer = std::make_shared<Viewer>(shared_from_this(),m_sptr_frameDrawer,m_sptr_mapDrawer,m_sptr_tracker,_strConfigurationPath);
        m_sptr_threadViewer = std::make_shared<thread>(&Viewer::run,m_sptr_viewer);
        m_sptr_tracker->setViewer(m_sptr_viewer);
    }
    //Set pointers between threads
    m_sptr_tracker->setLocalMapper(m_sptr_localMapper);
    m_sptr_tracker->setLoopClosing(m_sptr_loopCloser);
    m_sptr_localMapper->setTracker(m_sptr_tracker);
    m_sptr_localMapper->setLoopCloser(m_sptr_loopCloser);
    m_sptr_loopCloser->setTracker(m_sptr_tracker);
    m_sptr_loopCloser->setLocalMapper(m_sptr_localMapper);
}

cv::Mat System::trackStereo(const cv::Mat& _cvMatLeftImage, const cv::Mat& _cvMatRightImage, const double& _dbTimeStamp){
    if(m_enum_sensor != STEREO){
        std::cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << std::endl;
        exit(-1);
    }
    // Check mode change
    {
        std::unique_lock<std::mutex> lock(m_mutex_mode);
        if(m_b_activateLocalizationMode){
            m_sptr_localMapper->requestStop();
            // Wait until Local Mapping has effectively stopped
            while(!m_sptr_localMapper->isStopped()){
                usleep(1000);
            }
            m_sptr_tracker->onlyLocalization(true);
            m_b_activateLocalizationMode = false;
        }
        else if(m_b_deactivateLocalizationMode){
            m_sptr_tracker->onlyLocalization(false);
            m_sptr_localMapper->release();
            m_b_deactivateLocalizationMode = false;
        }
    }
    // Check reset
    {
        std::unique_lock<std::mutex> lock(m_mutex_reset);
        if(m_b_reset){
            m_sptr_tracker->reset();
            m_b_reset = false;
        }
    }
    cv::Mat cvMatTc2w = m_sptr_tracker->grabImageStereo(_cvMatLeftImage,_cvMatRightImage,_dbTimeStamp);
    std::unique_lock<std::mutex> lock2(m_mutex_state);
    m_int_trackingState = (int)m_sptr_tracker->m_enum_currentState;
    m_v_sptrTrackedMapPoints = m_sptr_tracker->m_currentFrame.m_v_sptrMapPoints;
    m_v_cvPointTrackedKeyPoints = m_sptr_tracker->m_currentFrame.m_v_keyPoints;
    return cvMatTc2w;
}

cv::Mat System::trackRGBD(const cv::Mat& _cvMatImage, const cv::Mat& _cvMatDepth, const double& _dbTimeStamp){
    if(m_enum_sensor != RGBD){
        std::cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << std::endl;
        exit(-1);
    }
    // Check mode change
    {
        std::unique_lock<std::mutex> lock(m_mutex_mode);
        if(m_b_activateLocalizationMode){
            m_sptr_localMapper->requestStop();
            // Wait until Local Mapping has effectively stopped
            while(!m_sptr_localMapper->isStopped()){
                usleep(1000);
            }
            m_sptr_tracker->onlyLocalization(true);
            m_b_activateLocalizationMode = false;
        }
        else if(m_b_deactivateLocalizationMode){
            m_sptr_tracker->onlyLocalization(false);
            m_sptr_localMapper->release();
            m_b_deactivateLocalizationMode = false;
        }
    }
    // Check reset
    {
        std::unique_lock<std::mutex> lock(m_mutex_reset);
        if(m_b_reset){
            m_sptr_tracker->reset();
            m_b_reset = false;
        }
    }
    cv::Mat cvMatTc2w = m_sptr_tracker->grabImageRGBD(_cvMatImage,_cvMatDepth,_dbTimeStamp);
    std::unique_lock<std::mutex> lock2(m_mutex_state);
    m_int_trackingState = (int)m_sptr_tracker->m_enum_currentState;
    m_v_sptrTrackedMapPoints = m_sptr_tracker->m_currentFrame.m_v_sptrMapPoints;
    m_v_cvPointTrackedKeyPoints = m_sptr_tracker->m_currentFrame.m_v_keyPoints;
    return cvMatTc2w;
}

void System::activateLocalizationMode(){
    std::unique_lock<std::mutex> lock(m_mutex_mode);
    m_b_activateLocalizationMode = true;
}

void System::deactivateLocalizationMode(){
    std::unique_lock<std::mutex> lock(m_mutex_mode);
    m_b_deactivateLocalizationMode = true;
}

bool System::mapChanged(){
    static int timeThreshold = 0;
    int loopUpdateAndGBAtimes = mpMap->getLastBigChangeIdx();
    if(timeThreshold < loopUpdateAndGBAtimes){
        timeThreshold = loopUpdateAndGBAtimes;
        return true;
    }else{
        return false;
    }
}

void System::reset(){
    std::unique_lock<std::mutex> lock(m_mutex_reset);
    m_b_reset = true;
}

void System::shutdown(){
    m_sptr_localMapper->requestFinish();
    m_sptr_loopCloser->requestFinish();
    if(m_sptr_viewer){
        m_sptr_viewer->requestFinish();
        while(!m_sptr_viewer->isFinished())
            usleep(5000);
    }
    // Wait until all thread have effectively stopped
    while(!m_sptr_localMapper->isFinished() || !m_sptr_loopCloser->isFinished() || m_sptr_loopCloser->isRunningGBA()){
        usleep(5000);
    }
    if(m_sptr_viewer){
        pangolin::BindToContext("YDORBSLAM: Map Viewer");
    }
}

void System::saveTrajectoryTUM(const string& _filename){
    std::cout << std::endl << "Saving camera trajectory to " << _filename << " ..." << std::endl;
    std::vector<std::shared_ptr<KeyFrame>> vsptrAllKFs = m_sptr_map->getAllKeyFrames();
    sort(vsptrAllKFs.begin(),vsptrAllKFs.end(),[](std::shared_ptr<KeyFrame> sptrKF1, std::shared_ptr<KeyFrame> sptrKF2)\
    {return sptrKF1->m_int_keyFrameID < sptrKF2->m_int_keyFrameID});
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Tw_firstKF = vsptrAllKFs[0]->getInverseCameraPoseByTransform_w2c();
    ofstream trajectoryFile;
    trajectoryFile.open(_filename.c_str());
    trajectoryFile << fixed;
    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbF).
    std::list<std::shared_ptr<KeyFrame>>::iterator lRKFit = m_sptr_tracker->m_l_sptrRefKF.begin();
    std::list<double>::iterator lTit = m_sptr_tracker->m_l_dframeTimes.begin();
    std::list<bool>::iterator lbFit = m_sptr_tracker->m_l_btrackFail.begin();
    for(cv::Mat& TcF_refKF : m_sptr_tracker->m_l_cvMatRelativeFramePoses){
        if(*lbFit){
            continue;
        }
        cv::Mat TrefKF_firstKF = cv::Mat::eye(4,4,CV_32F);
        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while((*lRKFit)->isBad()){
            TrefKF_firstKF = TrefKF_firstKF * (*lRKFit)->m_cvMat_T_c2p;
            (*lRKFit) = (*lRKFit)->getParent();
        }
        TrefKF_firstKF = TrefKF_firstKF * (*lRKFit)->getCameraPoseByTransform_c2w() * Tw_firstKF;
        cv::Mat TcF_firstKF = TcF_refKF * TrefKF_firstKF;
        cv::Mat RfirstKF_cF = TcF_firstKF.rowRange(0,3).colRange(0,3).t();
        cv::Mat tfirstKF_cF = -RfirstKF_cF*TcF_firstKF.rowRange(0,3).col(3);
        std::vector<float> q = Converter::rotation_cvMat_eigenQuat(RfirstKF_cF);
        trajectoryFile << setprecision(6) << *lT << " " <<  setprecision(9) << tfirstKF_cF.at<float>(0) << " " << tfirstKF_cF.at<float>(1) \
        << " " << tfirstKF_cF.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
        lRKFit++;
        lTit++;
        lbFit++;
    }
    trajectoryFile.close();
    std::cout << std::endl << "trajectory saved!" << std::endl;
}

void System::saveKeyFrameTrajectoryTUM(const string& _filename){
    std::cout << std::endl << "Saving keyframe trajectory to " << _filename << " ..." << std::endl;
    std::vector<std::shared_ptr<KeyFrame>> vsptrAllKFs = m_sptr_map->getAllKeyFrames();
    sort(vsptrAllKFs.begin(),vsptrAllKFs.end(),[](std::shared_ptr<KeyFrame> sptrKF1, std::shared_ptr<KeyFrame> sptrKF2)\
    {return sptrKF1->m_int_keyFrameID < sptrKF2->m_int_keyFrameID});
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // cv::Mat Two = vpKFs[0]->GetPoseInverse();
    ofstream KFTrajectoryFile;
    KFTrajectoryFile.open(_filename.c_str());
    KFTrajectoryFile << fixed;

    for(std::shared_ptr<KeyFrame>& sptrKF : vsptrAllKFs){
        // pKF->SetPose(pKF->GetPose()*Two);
        if(!sptrKF->isBad()){
            cv::Mat Rkf2w = sptrKF->getRotation_c2w();
            cv::Mat Okf = sptrKF->getCameraOriginInWorld();
            std::vector<float> q = Converter::rotation_cvMat_eigenQuat(Rkf2w);
            KFTrajectoryFile << setprecision(6) << sptrKF->m_d_timeStamp << setprecision(7) << " " << Okf.at<float>(0) \
            << " " << Okf.at<float>(1) << " " << Okf.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
    }
    KFTrajectoryFile.close();
    std::cout << std::endl << "keyframe trajectory saved!" << std::endl;
}

int System::getTrackingState(){
    std::unique_lock<std::mutex> lock(m_mutex_state);
    return m_int_trackingState;
}

std::vector<std::shared_ptr<MapPoint>> System::getTrackedMapPoints(){
    std::unique_lock<std::mutex> lock(m_mutex_state);
    return m_v_sptrTrackedMapPoints;
}

std::vector<cv::KeyPoint> System::getTrackedKeyPoints(){
    std::unique_lock<std::mutex> lock(m_mutex_state);
    return m_v_cvPointTrackedKeyPoints;
}

} //namespace YDORBSLAM
/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "frameDrawer.hpp"
#include "tracking.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <sstream>
#include <mutex>

namespace YDORBSLAM{
FrameDrawer::FrameDrawer(std::shared_ptr<Map> _sptrMap):m_sptrMap(_sptrMap){
    m_tracking_state = TrackingState::SYSTEM_NOT_READY;
    m_cvMat_image = cv::Mat(480,640,CV_8UC3,cv::Scalar(0,0,0));
}
cv::Mat FrameDrawer::drawFrame(){
    cv::Mat cvMatImage;
    std::vector<cv::KeyPoint> vIniKeyPoints; // Initialization: KeyPoints in reference frame
    std::vector<int> vnMatchedKeyPointID; // Initialization: correspondeces id with reference keypoints
    std::vector<cv::KeyPoint> vCurrentKeyPoints; // KeyPoints in current frame
    std::vector<bool> vbNonObservation, vbOwnObservation; // Tracked MapPoints in current frame
    TrackingState state; // Tracking state

    //Copy variables within scoped mutex
    {
        std::unique_lock<std::mutex> lock(m_copyMutex);
        state = m_tracking_state;
        if(m_tracking_state == TrackingState::SYSTEM_NOT_READY){
            m_tracking_state == TrackingState::NO_IMAGE_YET;
        }
        m_cvMat_image.copyTo(cvMatImage);
        if(m_tracking_state == TrackingState::NOT_INITIALIZED){
            vCurrentKeyPoints = m_v_cvMatCurrentKeyPoints;
        }
        else if(m_tracking_state == TrackingState::OK){
            vCurrentKeyPoints = m_v_cvMatCurrentKeyPoints;
            vbNonObservation = m_v_bNonObservation;
            vbOwnObservation = m_v_bOwnObservation;
        }
        else if(m_tracking_state == TrackingState::LOST){
            vCurrentKeyPoints = m_v_cvMatCurrentKeyPoints;
        }
    } // destroy scoped mutex -> release mutex
    if(cvMatImage.channels()<3){
        cvtColor(cvMatImage,cvMatImage,cv::COLOR_GRAY2BGR);
    } //this should be always true
    //Draw
    if(m_tracking_state == TrackingState::NOT_INITIALIZED){
        unsigned int ifor = 0;
        for(int& KeyPointID : vnMatchedKeyPointID){
            if(KeyPointID >= 0){
                cv::line(cvMatImage,vIniKeyPoints[ifor].pt,vCurrentKeyPoints[KeyPointID].pt,cv::Scalar(0,255,0));
            }
            ifor++;
        }
    }
    else if(m_tracking_state == TrackingState::OK){
        m_int_ownObservationMPNums = 0;
        m_int_nonObservationMPNums = 0;
        const float r = 5;
        for(int i=0; i<vCurrentKeyPoints.size(); i++){
            if(vbNonObservation[i] || vbOwnObservation[i]){
                // match window(2r*2r)
                cv::Point2f pt1,pt2;
                pt1.x = vCurrentKeyPoints[i].pt.x - r;
                pt1.y = vCurrentKeyPoints[i].pt.y - r;
                pt2.x = vCurrentKeyPoints[i].pt.x + r;
                pt2.y = vCurrentKeyPoints[i].pt.y + r;
                // This is a match to a MapPoint in the map
                if(vbOwnObservation[i]){
                    cv::rectangle(cvMatImage,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(cvMatImage,vCurrentKeyPoints[i].pt,2,cv::Scalar(0,255,0),-1);
                    m_int_ownObservationMPNums++;
                }else{
                    cv::rectangle(cvMatImage,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(cvMatImage,vCurrentKeyPoints[i].pt,2,cv::Scalar(255,0,0),-1);
                    m_int_nonObservationMPNums++;
                } // This is match to a "visual odometry" MapPoint created in the last frame
            }
        }
    }
    cv::Mat imWithInfo;
    drawTextInfo(cvMatImage,state,imWithInfo);
    return imWithInfo;
}
void FrameDrawer::drawTextInfo(cv::Mat &_image, TrackingState _nState, cv::Mat &_imageText){
    std::stringstream s;
    if(_nState == TrackingState::NO_IMAGE_YET){
        s << " WAITING FOR IMAGES";
    }
    else if(_nState == TrackingState::NOT_INITIALIZED){
        s << " TRYING TO INITIALIZE ";
    }
    else if(_nState == TrackingState::OK){
        if(!m_b_onlyLocalization){
            s << "SLAM MODE |  ";
        }else{
            s << "LOCALIZATION | ";
        }
        s << "KFs: " << m_sptrMap->getKeyFramesNum() << ", MPs: " << m_sptrMap->getMapPointsNum() << ", Matches: " << m_int_ownObservationMPNums;
        if(m_int_nonObservationMPNums > 0){
            s << ", + nonObservation matches: " << m_int_nonObservationMPNums;
        }
    }
    else if(_nState == TrackingState::LOST){
        s << " TRACK FAIL. TRYING TO RELOCALIZE ";
    }
    else if(_nState == TrackingState::SYSTEM_NOT_READY){
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);
    _imageText = cv::Mat(_image.rows+textSize.height+10,_image.cols,_image.type());
    _image.copyTo(_imageText.rowRange(0,_image.rows).colRange(0,_image.cols));
    _imageText.rowRange(_image.rows,_imageText.rows) = cv::Mat::zeros(textSize.height+10,_image.cols,_image.type());
    cv::putText(_imageText,s.str(),cv::Point(5,_imageText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
}
void FrameDrawer::update(std::shared_ptr<Tracking> _sptrTracker){
    std::unique_lock<std::mutex> lock(m_copyMutex);
    _sptrTracker->m_cvMat_grayImage.copyTo(m_cvMat_image);
    m_v_cvMatCurrentKeyPoints = _sptrTracker->m_frame_currentFrame.m_v_keyPoints;
    m_int_CurrentKeyPointNUM = m_v_cvMatCurrentKeyPoints.size();
    m_v_bNonObservation = std::vector<bool>(m_int_CurrentKeyPointNUM,false);
    m_v_bOwnObservation = std::vector<bool>(m_int_CurrentKeyPointNUM,false);
    m_b_onlyLocalization = _sptrTracker->m_b_isTrackingOnly;

    if(_sptrTracker->m_ts_lastProcessedState == TrackingState::OK){
        for(int i=0; i<m_int_CurrentKeyPointNUM; i++){
            std::shared_ptr<MapPoint> sptrMP = _sptrTracker->m_frame_currentFrame.m_v_sptrMapPoints[i];
            if(sptrMP && !_sptrTracker->m_frame_currentFrame.m_v_isOutliers[i]){
                if(sptrMP->getObservationsNum() > 0){
                    m_v_bOwnObservation[i] = true;
                }else{
                    m_v_bNonObservation[i] = true;
                }
            }
        }
    }
    m_tracking_state = _sptrTracker->m_ts_lastProcessedState;
}

} //namespace ORB_SLAM
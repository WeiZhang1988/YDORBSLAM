/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "localMapping.hpp"
#include "loopClosing.hpp"
#include "orbMatcher.hpp"
#include "optimizer.hpp"
#include <mutex>
#include <memory>
#include <unistd.h>

namespace YDORBSLAM{

LocalMapping::LocalMapping(std::shared_ptr<Map> _sptrMap):mbResetRequested(false), mbFinishRequested(false), mbFinished(true), \
m_sptrMap(_sptrMap),mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true){}

void LocalMapping::run(){
    m_b_finished = false;
    while(1){
        // Tracking will see that Local Mapping is busy
        setAcceptKeyFrames(false);
        // Check if there are keyframes in the queue
        if(checkNewKeyFrames()){
            // BoW conversion and insertion in Map
            processNewKeyFrame();
            // Check recent MapPoints
            mapPointCulling();
            // Triangulate new MapPoints
            createNewMapPoints();
            if(!checkNewKeyFrames()){
                // Find more matches in neighbor keyframes and fuse point duplications
                searchInNeighbors();
            }
            m_b_abortBA = false;
            if(!checkNewKeyFrames() && !stopRequested()){
                // Local BA
                if(m_sptrMap->getKeyFramesNum() > 2){
                    Optimizer::localBundleAdjust(m_sptrCurrentKeyFrame,m_sptrMap,m_b_abortBA);
                }
                // Check redundant local Keyframes
                keyFrameCulling();
            }
            m_sptrLoopCloser->insertKeyFrame(m_sptrCurrentKeyFrame);
        }
        else if(stop()){
            // Safe area to stop
            while(isStopped() && !checkFinish()){
                usleep(3000);
            }
            if(checkFinish()){
                break;
            }
        }
        resetIfRequested();
        // Tracking will see that Local Mapping is busy
        setAcceptKeyFrames(true);
        if(checkFinish()){
            break;
        }
        usleep(3000);
    }
    setFinish();
}

void LocalMapping::insertKeyFrame(std::shared_ptr<KeyFrame> _sptrKF){
    std::unique_lock<std::mutex> lock(m_mutex_newKFs);
    m_l_newKeyFrames.push_back(_sptrKF);
    m_b_abortBA = true;
}

bool LocalMapping::checkNewKeyFrames(){
    std::unique_lock<std::mutex> lock(m_mutex_newKFs);
    return(!m_l_newKeyFrames.empty());
}

void LocalMapping::processNewKeyFrame(){
    {
        std::unique_lock<std::mutex> lock(m_mutex_newKFs);
        m_sptrCurrentKeyFrame = m_l_newKeyFrames.front();
        m_l_newKeyFrames.pop_front();
    }
    // Compute Bags of Words structures
    m_sptrCurrentKeyFrame->computeBoW();
    // Associate MapPoints to the new keyframe and update normal and descriptor
    for(const std::shared_ptr<MapPoint>& sptrMapPoint : m_sptrCurrentKeyFrame->getMatchedMapPointsVec()){
        if(sptrMapPoint && !sptrMapPoint->isBad()){
            if(!sptrMapPoint->isInKeyFrame(m_sptrCurrentKeyFrame)){
                sptrMapPoint->addObservation(m_sptrCurrentKeyFrame, i);
                sptrMapPoint->updateNormalAndDepth();
                sptrMapPoint->computeDistinctiveDescriptors();
            }else{
                m_l_sptrRecentAddedMapPoints.push_back(sptrMapPoint);
            } // this can only happen for new stereo points inserted by the Tracking
        }
    }
    // Update links in the Covisibility Graph
    m_sptrCurrentKeyFrame->updateConnections();
    // Insert Keyframe in Map
    m_sptrMap->addKeyFrame(m_sptrCurrentKeyFrame);
}

void LocalMapping::createNewMapPoints(){
    // Retrieve neighbor(covisible) keyframes in covisibility graph
    const std::vector<std::shared_ptr<KeyFrame>> vsptrCovisibleKFs = m_sptrCurrentKeyFrame->getFirstNumOrderedConnectedKeyFrames(10);
    OrbMatcher matcher(0.6,false);
    cv::Mat currentKFRc2w = m_sptrCurrentKeyFrame->getRotation_c2w();
    cv::Mat currentKFRw2c = m_sptrCurrentKeyFrame->getRotation_w2c();
    cv::Mat currentKFtc2w = m_sptrCurrentKeyFrame->getTranslation_c2w();
    cv::Mat currentKFTc2w(3,4,CV_32F);
    currentKFRw2c.copyTo(currentKFTc2w.colRange(0,3));
    currentKFtc2w.copyTo(currentKFTc2w.col(3));
    cv::Mat currentKFOw = m_sptrCurrentKeyFrame->getCameraOriginInWorld();
    const float& currentKFfx = m_sptrCurrentKeyFrame->m_flt_fx;
    const float& currentKFfy = m_sptrCurrentKeyFrame->m_flt_fy;
    const float& currentKFcx = m_sptrCurrentKeyFrame->m_flt_cx;
    const float& currentKFcy = m_sptrCurrentKeyFrame->m_flt_cy;
    const float& currentKFinvFx = m_sptrCurrentKeyFrame->m_flt_invFx;
    const float& currentKFinvFy = m_sptrCurrentKeyFrame->m_flt_invFy;
    const float ratioFactor = 1.5f * m_sptrCurrentKeyFrame->m_flt_scaleFactor;
    // Search matches with epipolar restriction and triangulate
    for(const std::shared_ptr<KeyFrame>& covisibleKF : vsptrCovisibleKFs){
        if(covisibleKF != vsptrCovisibleKFs.front() && checkNewKeyFrames()){
            return;
        }
        // Check first that baseline is not too short
        cv::Mat covisibleKFOw = covisibleKF->getCameraOriginInWorld();
        const float baseLine = cv::norm(covisibleKFOw - currentKFOw);
        if(baseLine < covisibleKF->m_flt_baseLine){
            continue;
        }
        // Compute Fundamental Matrix
        cv::Mat Fcurrent2covisible = ComputeF12(m_sptrCurrentKeyFrame,covisibleKF);
        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t,size_t> > vpairMatchedIndices;
        matcher.searchForTriangulation(m_sptrCurrentKeyFrame,covisibleKF,Fcurrent2covisible,vpairMatchedIndices,false);
        cv::Mat covisibleKFRc2w = covisibleKF->getRotation_c2w();
        cv::Mat covisibleKFRw2c = covisibleKF->getRotation_w2c();
        cv::Mat covisibleKFtc2w = covisibleKF->getTranslation_c2w();
        cv::Mat covisibleKFTc2w(3,4,CV_32F);
        covisibleKFRw2c.copyTo(covisibleKFTc2w.colRange(0,3));
        covisibleKFtc2w.copyTo(covisibleKFTc2w.col(3));
        const float& covisibleKFfx = covisibleKF->m_flt_fx;
        const float& covisibleKFfy = covisibleKF->m_flt_fy;
        const float& covisibleKFcx = covisibleKF->m_flt_cx;
        const float& covisibleKFcy = covisibleKF->m_flt_cy;
        const float& covisibleKFinvFx = covisibleKF->m_flt_invFx;
        const float& covisibleKFinvFy = covisibleKF->m_flt_invFy;
        // Triangulate each match
        for(const std::pair<size_t,size_t>& matchedIndices : vpairMatchedIndices){
            const cv::KeyPoint& currentKFkp = m_sptrCurrentKeyFrame->m_v_keyPoints[matchedIndices.first];
            const float currentKFkp_rX = m_sptrCurrentKeyFrame->m_v_rightXcords[matchedIndices.first];
            const cv::KeyPoint& covisibleKFkp = covisibleKF->m_v_keyPoints[matchedIndices.second];
            const float covisibleKFkp_rX = covisibleKF->m_v_rightXcords[matchedIndices.second];
            // Check parallax between rays
            cv::Mat Ocurrentkp_directionV = (cv::Mat_<float>(3,1) << (currentKFkp.pt.x-currentKFcx)*currentKFinvFx, (currentKFkp.pt.y-currentKFcy)*currentKFinvFy, 1.0);
            cv::Mat Ocovisiblekp_directionV = (cv::Mat_<float>(3,1) << (covisibleKFkp.pt.x-covisibleKFcx)*covisibleKFinvFx, (covisibleKFkp.pt.y-covisibleKFcy)*covisibleKFinvFy, 1.0);
            cv::Mat WcurrentRayDirection = currentKFRw2c * Ocurrentkp_directionV;
            cv::Mat WcovisibleRayDirection = covisibleKFRw2c * Ocovisiblekp_directionV;
            const float cosParallaxRays = WcurrentRayDirection.dot(WcovisibleRayDirection)/(cv::norm(WcurrentRayDirection)*cv::norm(WcovisibleRayDirection));
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cos(2*atan2(m_sptrCurrentKeyFrame->m_flt_baseLine/2,m_sptrCurrentKeyFrame->m_v_depth[matchedIndices.first]));
            float cosParallaxStereo2 = cosParallaxStereo;
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
            cv::Mat inverseProject3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0){
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = Ocurrentkp_directionV.at<float>(0)*currentKFTc2w.row(2)-currentKFTc2w.row(0);
                A.row(1) = Ocurrentkp_directionV.at<float>(1)*currentKFTc2w.row(2)-currentKFTc2w.row(1);
                A.row(2) = Ocovisiblekp_directionV.at<float>(0)*covisibleKFTc2w.row(2)-covisibleKFTc2w.row(0);
                A.row(3) = Ocovisiblekp_directionV.at<float>(1)*covisibleKFTc2w.row(2)-covisibleKFTc2w.row(1);
                cv::Mat u,w,vt; //A = u * w * vt
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
                inverseProject3D = vt.row(3).t();
                if(inverseProject3D.at<float>(3) == 0){
                    continue;
                }
                // Euclidean coordinates
                inverseProject3D = inverseProject3D.rowRange(0,3)/inverseProject3D.at<float>(3);
            }
            else if(cosParallaxStereo1 < cosParallaxStereo2){
                inverseProject3D = m_sptrCurrentKeyFrame->inverseProject(matchedIndices.first);
            }
            else if(cosParallaxStereo2 < cosParallaxStereo1){
                inverseProject3D = covisibleKF->inverseProject(matchedIndices.second);
            }
            else{
                continue; //No stereo and very low parallax
            }
            //Check triangulation in front of cameras
            float z1 = currentKFRc2w.row(2).dot(inverseProject3D.t())+currentKFtc2w.at<float>(2);
            float z2 = covisibleKFRw2c.row(2).dot(inverseProject3D.t())+covisibleKFtc2w.at<float>(2);
            if(z1 <= 0 || z2 <= 0){
                continue;
            }
            //Check reprojection error in first keyframe
            const float& sigmaSquare1 = m_sptrCurrentKeyFrame->m_v_scaleFactorSquares[currentKFkp.octave];
            const float x1 = currentKFRc2w.row(0).dot(inverseProject3D.t())+currentKFtc2w.at<float>(0);
            const float y1 = currentKFRc2w.row(1).dot(inverseProject3D.t())+currentKFtc2w.at<float>(1);
            const float invz1 = 1.0f/z1; 
            float u1 = currentKFfx * x1 * invz1 + currentKFcx;
            float u1_r = u1 - m_sptrCurrentKeyFrame->m_flt_baseLineTimesFx * invz1;
            float v1 = currentKFfy * y1 * invz1 + currentKFcy;
            cv::Mat Err1 = (cv::Mat_<float>(3,1) << (u1-currentKFkp.pt.x), (v1-currentKFkp.pt.y), (u1_r-currentKFkp_rX));
            if(pow(cv::norm(Err1),2) > 7.8*sigmaSquare1){
                continue;
            }
            //Check reprojection error in second keyframe
            const float& sigmaSquare2 = covisibleKF->m_v_scaleFactorSquares[covisibleKFkp.octave];
            const float x1 = covisibleKFRc2w.row(0).dot(inverseProject3D.t())+covisibleKFtc2w.at<float>(0);
            const float y1 = covisibleKFRc2w.row(1).dot(inverseProject3D.t())+covisibleKFtc2w.at<float>(1);
            const float invz2 = 1.0f/z2; 
            float u2 = covisibleKFfx * x2 * invz2 + covisibleKFcx;
            float u2_r = u2 - covisibleKF->m_flt_baseLineTimesFx * invz2;
            float v2 = covisibleKFfy * y2 * invz2 + covisibleKFcy;
            cv::Mat Err2 = (cv::Mat_<float>(3,1) << (u2-covisibleKFkp.pt.x), (v2-covisibleKFkp.pt.y), (u2_r-covisibleKFkp_rX));
            if(pow(cv::norm(Err2),2) > 7.8*sigmaSquare2){
                continue;
            }
            //Check scale consistency
            float dist1 = cv::norm(inverseProject3D - currentKFOw);
            float dist2 = cv::norm(inverseProject3D - covisibleKFOw);
            if(dist1 == 0 || dist2 == 0){
                continue;
            }
            const float ratioDist = dist2/dist1;
            const float ratioOctave = m_sptrCurrentKeyFrame->m_v_scaleFactors[currentKFkp.octave]/covisibleKF->m_v_scaleFactors[covisibleKFkp.octave];
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor){
                continue;
            }
            // Triangulation is succesfull
            std::shared_ptr<MapPoint> sptrMP = std::make_shared<MapPoint>(inverseProject3D,m_sptrCurrentKeyFrame,m_sptrMap);
            sptrMP->addObservation(m_sptrCurrentKeyFrame, matchedIndices.first);
            sptrMP->addObservation(covisibleKF, matchedIndices.second);
            m_sptrCurrentKeyFrame->addMapPoint(sptrMP, matchedIndices.first);
            covisibleKF->addMapPoint(sptrMP, matchedIndices.second);
            sptrMP->computeDistinctiveDescriptors();
            sptrMP->updateNormalAndDepth();
            m_sptrMap->addMapPoint(sptrMP);
            m_l_sptrRecentAddedMapPoints.push_back(m_sptrMap);
        }
    }
}

void LocalMapping::mapPointCulling(){
    // Check Recent Added MapPoints
    std::list<std::shared_ptr<MapPoint>>::iterator lit = m_l_sptrRecentAddedMapPoints.begin();
    const unsigned long int currentKFid = m_sptrCurrentKeyFrame->m_int_keyFrameID;
    while(lit != m_l_sptrRecentAddedMapPoints.end()){
        std::shared_ptr<MapPoint> sptrMP = *lit;
        if(sptrMP->isBad()){
            lit = m_l_sptrRecentAddedMapPoints.erase(lit);
        }
        else if(sptrMP->getFoundRatio() < 0.25f){
            sptrMP->setBadFlag();
            lit = m_l_sptrRecentAddedMapPoints.erase(lit);
        }
        else if(((int)currentKFid-(int)sptrMP->m_int_firstKeyFrameID) >= 2 && sptrMP->getObservationsNum() <= 3){
            sptrMP->setBadFlag();
            lit = m_l_sptrRecentAddedMapPoints.erase(lit);
        }
        else if(((int)currentKFid-(int)sptrMP->m_int_firstKeyFrameID) >= 3){
            lit = m_l_sptrRecentAddedMapPoints.erase(lit);
        }
        else{
            lit++;
        }
    }
}

void LocalMapping::searchInNeighbors(){
    // Retrieve neighbor keyframes
    std::vector<std::shared_ptr<KeyFrame>> vsptrTargetKFs;
    for(const std::shared_ptr<KeyFrame>& covisibleKF : m_sptrCurrentKeyFrame->getFirstNumOrderedConnectedKeyFrames(10)){
        if(!covisibleKF->isBad() && covisibleKF->m_int_fuseTargetForKeyFrameID != m_sptrCurrentKeyFrame->m_int_keyFrameID){
            vsptrTargetKFs.push_back(covisibleKF);
            covisibleKF->m_int_fuseTargetForKeyFrameID = m_sptrCurrentKeyFrame->m_int_keyFrameID;
            for(const std::shared_ptr<KeyFrame>& secondCovisibleKF : m_sptrCurrentKeyFrame->getFirstNumOrderedConnectedKeyFrames(5)){
                if(!secondCovisibleKF->isBad() && secondCovisibleKF->m_int_fuseTargetForKeyFrameID != m_sptrCurrentKeyFrame->m_int_keyFrameID && \
                secondCovisibleKF->m_int_keyFrameID != m_sptrCurrentKeyFrame->m_int_keyFrameID){
                    vsptrTargetKFs.push_back(secondCovisibleKF);
                }
            }
        }
    }
    // Search matches by projection from current KF in target KFs
    OrbMatcher matcher;
    const std::vector<std::shared_ptr<MapPoint>>& vsptrCurrentMP = m_sptrCurrentKeyFrame->getMatchedMapPointsVec();
    for(std::shared_ptr<KeyFrame>& project2TargetKF : vsptrTargetKFs){
        matcher.FuseByProjection(project2TargetKF, vsptrCurrentMP);
    }
    // Search matches by projection from target KFs in current KF
    std::vector<std::shared_ptr<MapPoint>> vsptrFuseCandidateMP;
    vsptrFuseCandidateMP.reserve(vsptrTargetKFs.size() * vsptrCurrentMP.size());
    for(std::shared_ptr<KeyFrame>& targetKF2Project : vsptrTargetKFs){
        for(std::shared_ptr<MapPoint>& sptrTargetMP : targetKF2Project->getMatchedMapPointsVec()){
            if(sptrTargetMP && !sptrTargetMP->isBad() && sptrTargetMP->m_int_fuseCandidateForKeyFrameID!=m_sptrCurrentKeyFrame->m_int_keyFrameID){
                sptrTargetMP->m_int_fuseCandidateForKeyFrameID = m_sptrCurrentKeyFrame->m_int_keyFrameID;
                vsptrFuseCandidateMP.push_back(sptrTargetMP);
            }
        }
    }
    matcher.FuseByProjection(m_sptrCurrentKeyFrame, vsptrFuseCandidateMP);
    // Update points
    for(std::shared_ptr<MapPoint>& updateCurrentMP : m_sptrCurrentKeyFrame->getMatchedMapPointsVec()){
        if(updateCurrentMP && !updateCurrentMP->isBad()){
            updateCurrentMP->computeDistinctiveDescriptors();
            updateCurrentMP->updateNormalAndDepth();
        }
    }
    // Update connections in covisibility graph
    m_sptrCurrentKeyFrame->updateConnections();
}

cv::Mat LocalMapping::computeF12(std::shared_ptr<KeyFrame>& _sptrKF1, std::shared_ptr<KeyFrame>& _sptrKF2){
    cv::Mat R1w = _sptrKF1->getRotation_c2w();
    cv::Mat t1w = _sptrKF1->getTranslation_c2w();
    cv::Mat R2w = _sptrKF2->getRotation_c2w();
    cv::Mat t2w = _sptrKF2->getTranslation_c2w();
    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    const cv::Mat &K1 = _sptrKF1->m_cvMat_intParMat;
    const cv::Mat &K2 = _sptrKF1->m_cvMat_intParMat;
    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::requestStop(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    m_b_stopRequested = true;
    std::unique_lock<std::mutex> lock2(m_mutex_newKFs);
    m_b_abortBA = true;
}

void LocalMapping::requestReset(){
    {
        std::unique_lock<std::mutex> lock(m_mutex_reset);
        m_b_resetRequested = true;
    }
    while(1){
        {
           std::unique_lock<std::mutex> lock2(m_mutex_reset);
            if(!m_b_resetRequested){
                break;
            }
        }
        usleep(3000);
    }
}

bool LocalMapping::stop(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    if(m_b_stopRequested && !m_b_notStop)
    {
        m_b_stopped = true;
        std::cout << "Local Mapping STOP" << std::endl;
        return true;
    }
    return false;
}

void LocalMapping::release(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    std::unique_lock<std::mutex> lock2(m_mutex_finish);
    if(m_b_finished)
        return;
    m_b_stopped = false;
    m_b_stopRequested = false;
    for(std::shared_ptr<KeyFrame>& newKF : m_l_newKeyFrames){
        newKF.reset();
    }
    m_l_newKeyFrames.clear();
    std::cout << "Local Mapping RELEASE" << std::endl;
}

bool LocalMapping::setNotStop(bool _flag){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    if(_flag && m_b_stopped){
        return false;
    }
    m_b_notStop = _flag;
    return true;
}

void LocalMapping::keyFrameCulling(){
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    int ifor = 0;
    for(std::shared_ptr<KeyFrame>& vsptrLocalKF : m_sptrCurrentKeyFrame->getOrderedConnectedKeyFrames()){
        if(vsptrLocalKF->m_int_keyFrameID == 0){
            continue;
        }
        const int thObs = 3;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for(const std::shared_ptr<MapPoint>& sptrLocalMP : vsptrLocalKF->getMatchedMapPointsVec()){
            if(sptrLocalMP && !sptrLocalMP->isBad()){
                nMPs++;
                if(sptrLocalMP->getObservationsNum() > thObs){
                    int nObs = 0;
                    const int& scaleLevel = vsptrLocalKF->m_v_keyPoints[ifor].octave;
                    for(const std::pair<std::shared_ptr<KeyFrame>, size_t>& localMPObs : sptrLocalMP->getObservations()){
                        const int& scaleLeveli = vsptrLocalKF->m_v_keyPoints[localMPObs.second].octave;
                        if(localMPObs.first != vsptrLocalKF && scaleLeveli <= scaleLevel+1){
                            nObs++;
                            if(nObs >= thObs){
                                break;
                            }
                        }
                    }
                    if(nObs >= thObs){
                        nRedundantObservations++;
                    }
                }
            }
        }
        if(nRedundantObservations > 0.9*nMPs){
            vsptrLocalKF->setBadFlag();
        }
        ifor++;
    }
}

cv::Mat LocalMapping::skewSymmetricMatrix(const cv::Mat& _v){
    return (cv::Mat_<float>(3,3) << 0, -_v.at<float>(2), _v.at<float>(1),\
                                    _v.at<float>(2), 0, -_v.at<float>(0),\
                                    -_v.at<float>(1), _v.at<float>(0), 0);
}

void LocalMapping::resetIfRequested(){
    std::unique_lock<std::mutex> lock(m_mutex_reset);
    if(m_b_resetRequested)
    {
        m_l_newKeyFrames.clear();
        m_l_sptrRecentAddedMapPoints.clear();
        m_b_resetRequested=false;
    }
}

} //namespace YDORBSLAM
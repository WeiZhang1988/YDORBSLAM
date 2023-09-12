#include "orbMatcher.hpp"
#include <limits.h>
#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"

namespace YDORBSLAM{
  const int OrbMatcher::m_int_highThd = 100;
  const int OrbMatcher::m_int_lowThd = 50;
  const int OrbMatcher::m_int_histLen = 30;

  int OrbMatcher::computeDescriptorsDistance(const cv::Mat &_firstDescriptor, const cv::Mat &_secondDescriptor){
    const int *pFirstDescriptor  = _firstDescriptor.ptr<int32_t>();
    const int *pSecondDescriptor = _secondDescriptor.ptr<int32_t>();
    int dist=0;
    for(int i=0; i<8; i++, pFirstDescriptor++, pSecondDescriptor++){
      unsigned  int v = *pFirstDescriptor ^ *pSecondDescriptor;
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
  }
  int OrbMatcher::searchByProjectionInFrameAndMapPoint(Frame &_frame, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, const float _thd){
    int matchNum = 0;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : _vSptrMapPoints){
      if(sptrMapPoint->m_b_isTrackInView && !sptrMapPoint->isBad()){
        const int &trackScaleLevel = sptrMapPoint->m_int_trackScaleLevel;
        //the size of the window depends on viewing direction
        float radius = _thd * getRadiusByViewCos(sptrMapPoint->m_flt_trackViewCos);
        const std::vector<int> vIndices = _frame.getKeyPointsInArea(sptrMapPoint->m_flt_trackProjX,sptrMapPoint->m_flt_trackProjY,radius*_frame.m_v_scaleFactors[trackScaleLevel],trackScaleLevel-1,trackScaleLevel);
        if(!vIndices.empty()){
          int bestDist        = 256;
          int bestLevel       = -1;
          int secondBestDist  = 256;
          int secondBestLevel = -1;
          int bestIdx         = -1;
          //get best and second best matches with near key points
          for(const int &idx : vIndices){
            if((!_frame.m_v_sptrMapPoints[idx] || _frame.m_v_sptrMapPoints[idx]->getObservationsNum()<=0) && \
            (_frame.m_v_rightXcords[idx]<=0 || fabs(sptrMapPoint->m_flt_trackProjRightX-_frame.m_v_rightXcords[idx])<=radius*_frame.m_v_scaleFactors[trackScaleLevel])){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_frame.m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                secondBestDist  = bestDist;
                bestDist        = dist;
                secondBestLevel = bestLevel;
                bestLevel       = _frame.m_v_keyPoints[idx].octave;
                bestIdx         = idx;
              }else if(dist<secondBestDist){
                secondBestLevel = _frame.m_v_keyPoints[idx].octave;
                secondBestDist  = dist;
              }
            }
          }
          //if the best and the second best matches are in the same scale level, apply ratio to second match
          if(bestDist<=m_int_highThd && (bestLevel!=secondBestLevel || bestDist<=m_flt_bestSecondRatio*secondBestDist)){
            _frame.m_v_sptrMapPoints[bestIdx] = sptrMapPoint;
            matchNum++;
          }
        }
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchByProjectionInLastAndCurrentFrame(Frame &_currentFrame, Frame &_lastFrame, const float _thd){
    int matchNum = 0;
    const cv::Mat rotation_currentC2w         = _currentFrame.m_cvMat_T_c2w.rowRange(0,3).colRange(0,3);
    const cv::Mat translation_currentC2w      = _currentFrame.m_cvMat_T_c2w.rowRange(0,3).col(3);
    const cv::Mat translation_w2currentC      = -rotation_currentC2w.t() * translation_currentC2w;
    const cv::Mat rotation_lastC2w            = _lastFrame.m_cvMat_T_c2w.rowRange(0,3).colRange(0,3);
    const cv::Mat translation_lastC2w         = _lastFrame.m_cvMat_T_c2w.rowRange(0,3).col(3);
    const cv::Mat translation_lastC2currentC  = rotation_lastC2w * translation_w2currentC + translation_lastC2w;
    //rotation histogram for rotation consistency check
    std::vector<std::vector<int>> rotationHist = std::vector<std::vector<int>>(m_int_histLen,std::vector<int>());
    for(std::vector<int> &rotation : rotationHist){
      rotation.reserve(500);
    }
    const float factor      = 1.0/m_int_histLen;
    const bool bIsForward   = translation_lastC2currentC.at<float>(2)>Frame::m_flt_baseLine;
    const bool bIsBackward  = -translation_lastC2currentC.at<float>(2)>Frame::m_flt_baseLine;
    int i_for = 0;
    for(std::shared_ptr<MapPoint> &sptrMapPoint : _lastFrame.m_v_sptrMapPoints){
      if(sptrMapPoint && !_lastFrame.m_v_isOutliers[i_for]){
        //3D projection
        cv::Mat mapPointPosInWorld  = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInCamera = rotation_currentC2w * mapPointPosInWorld + translation_currentC2w;
        const float mapPointXcordInCamera = mapPointPosInCamera.at<float>(0);
        const float mapPointYcordInCamera = mapPointPosInCamera.at<float>(1);
        const float mapPointZcordInCamera = mapPointPosInCamera.at<float>(2);
        const float mapPointProjXInImage  = Frame::m_flt_fx*mapPointXcordInCamera/mapPointZcordInCamera+Frame::m_flt_cx;
        const float mapPointProjYInImage  = Frame::m_flt_fy*mapPointYcordInCamera/mapPointZcordInCamera+Frame::m_flt_cy;
        if(mapPointZcordInCamera>=0.0 && _currentFrame.isInImage(mapPointProjXInImage,mapPointProjYInImage)){
          const float radius = _thd*_currentFrame.m_v_scaleFactors[_lastFrame.m_v_keyPoints[i_for].octave];
          std::vector<int> vIndices;
          if(bIsForward){
            vIndices = _currentFrame.getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius,_lastFrame.m_v_keyPoints[i_for].octave);
          }else if(bIsBackward){
            vIndices = _currentFrame.getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius,0,_lastFrame.m_v_keyPoints[i_for].octave);
          }else{
            vIndices = _currentFrame.getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius,_lastFrame.m_v_keyPoints[i_for].octave-1,_lastFrame.m_v_keyPoints[i_for].octave+1);
          }
          if(!vIndices.empty()){
             int bestDist = 256;
             int bestIdx  = -1;
             for(const int &idx : vIndices){
              if((!_currentFrame.m_v_sptrMapPoints[idx] || \
              _currentFrame.m_v_sptrMapPoints[idx]->getObservationsNum()<=0) && \
              (_currentFrame.m_v_rightXcords[idx]<=0.0 || \
              fabs(mapPointProjXInImage-Frame::m_flt_baseLineTimesFx/mapPointZcordInCamera-_currentFrame.m_v_rightXcords[idx])<=radius)){
                const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_currentFrame.m_cvMat_descriptors.row(idx));
                if(dist<bestDist){
                  bestDist  = dist;
                  bestIdx   = idx;
                }
              }
             }
             if(bestDist<m_int_highThd){
              _currentFrame.m_v_sptrMapPoints[bestIdx] = sptrMapPoint;
              matchNum++;
              if(m_b_isToCheckOrientation){
                float rotation = _lastFrame.m_v_keyPoints[i_for].angle - _currentFrame.m_v_keyPoints[bestIdx].angle;
                if(rotation<0.0){
                  rotation += 360.0;
                }
                int tmpBestIdx = round(rotation*factor);
                if(tmpBestIdx==m_int_histLen){
                  tmpBestIdx=0;
                }
                assert(tmpBestIdx>=0 && tmpBestIdx<m_int_histLen);
                rotationHist[tmpBestIdx].push_back(bestIdx);
              }
             }
          }
        }
      }
      i_for++;
    }
    if(m_b_isToCheckOrientation){
      int idx1 = -1;
      int idx2 = -1;
      int idx3 = -1;
      computeThreeMaxima(rotationHist,m_int_histLen,idx1,idx2,idx3);
      i_for = 0;
      for(const std::vector<int> &rotation : rotationHist){
        if(i_for!=idx1 && i_for!=idx2 && i_for!=idx3){
          for(const int &idx : rotation){
            _currentFrame.m_v_sptrMapPoints[idx]=std::shared_ptr<MapPoint>(nullptr);
            matchNum--;
          }
        }
        i_for++;
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchByProjectionInKeyFrameAndCurrentFrame(Frame &_currentFrame, std::shared_ptr<KeyFrame> _sptrKeyFrame, const std::set<std::shared_ptr<MapPoint>> &_setFoundMapPoints, const float _thd, const int _orbDist){
    int matchNum = 0;
    const cv::Mat rotation_c2w        = _currentFrame.m_cvMat_T_c2w.rowRange(0,3).colRange(0,3);
    const cv::Mat translation_c2w     = _currentFrame.m_cvMat_T_c2w.rowRange(0,3).col(3);
    const cv::Mat cameraOriginInWorld = -rotation_c2w*translation_c2w;
    //rotation histogram for rotation consistency check
    std::vector<std::vector<int>> rotationHist = std::vector<std::vector<int>>(m_int_histLen,std::vector<int>());
    for(std::vector<int> &rotation : rotationHist){
      rotation.reserve(500);
    }
    const float factor = 1.0/m_int_histLen;
    const std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPointsInKeyFrame = _sptrKeyFrame->getMatchedMapPointsVec();
    int i_for = 0;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : vSptrMatchedMapPointsInKeyFrame){
      if(sptrMapPoint && !sptrMapPoint->isBad() && !_setFoundMapPoints.count(sptrMapPoint)){
        //3D projection
        cv::Mat mapPointPosInWorld  = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInCamera = rotation_c2w * mapPointPosInWorld + translation_c2w;
        const float mapPointXcordInCamera = mapPointPosInCamera.at<float>(0);
        const float mapPointYcordInCamera = mapPointPosInCamera.at<float>(1);
        const float mapPointZcordInCamera = mapPointPosInCamera.at<float>(2);
        const float mapPointProjXInImage  = Frame::m_flt_fx*mapPointXcordInCamera/mapPointZcordInCamera+Frame::m_flt_cx;
        const float mapPointProjYInImage  = Frame::m_flt_fy*mapPointYcordInCamera/mapPointZcordInCamera+Frame::m_flt_cy;
        //compute predicted scale level
        const float dist3D = cv::norm(mapPointPosInWorld - cameraOriginInWorld);
        const float maxDistance = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance = sptrMapPoint->getMinDistanceInvariance();
        const int predictedScaleLevel = sptrMapPoint->predictScaleLevel(dist3D,_currentFrame);
        //search in window
        const float radius = _thd*_currentFrame.m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices = _currentFrame.getKeyPointsInArea(mapPointProjXInImage, mapPointProjYInImage, radius, predictedScaleLevel-1, predictedScaleLevel+1);
        if(mapPointZcordInCamera>=0.0 && \
        _currentFrame.isInImage(mapPointProjXInImage,mapPointProjYInImage) && \
        dist3D>=minDistance&&dist3D<=maxDistance && \
        !vIndices.empty()){
          int bestDist = 256;
          int bestIdx  = -1;
          for(const int &idx : vIndices){
            if(!_currentFrame.m_v_sptrMapPoints[idx]){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_currentFrame.m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist  = dist;
                bestIdx   = idx;
              }
            }
          }
          if(bestDist<=_orbDist){
            _currentFrame.m_v_sptrMapPoints[bestIdx] = sptrMapPoint;
            matchNum++;
            if(m_b_isToCheckOrientation){
              float rotation = _sptrKeyFrame->m_v_keyPoints[i_for].angle - _currentFrame.m_v_keyPoints[bestIdx].angle;
              if(rotation<0.0){
                rotation += 360.0;
              }
              int tmpBestIdx = round(rotation*factor);
              if(tmpBestIdx==m_int_histLen){
                tmpBestIdx=0;
              }
              assert(tmpBestIdx>=0 && tmpBestIdx<m_int_histLen);
              rotationHist[tmpBestIdx].push_back(bestIdx);
            }
          }
        }
      }
      i_for++;
    }
    if(m_b_isToCheckOrientation){
      int idx1 = -1;
      int idx2 = -1;
      int idx3 = -1;
      computeThreeMaxima(rotationHist,m_int_histLen,idx1,idx2,idx3);
      i_for = 0;
      for(const std::vector<int> &rotation : rotationHist){
        if(i_for!=idx1 && i_for!=idx2 && i_for!=idx3){
          for(const int &idx : rotation){
            _currentFrame.m_v_sptrMapPoints[idx]=std::shared_ptr<MapPoint>(nullptr);
            matchNum--;
          }
        }
        i_for++;
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchByProjectionInSim(std::shared_ptr<KeyFrame> _sptrKeyFrame, cv::Mat &_sim_c2w, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const int _thd){
    //decompose _sim_c2w
    cv::Mat simScaledRotation_c2w   = _sim_c2w.rowRange(0,3).colRange(0,3);
    const float simScale_c2w        = sqrt(simScaledRotation_c2w.row(0).dot(simScaledRotation_c2w.row(0)));
    cv::Mat simRotation_c2w         = simScaledRotation_c2w/simScale_c2w;
    cv::Mat simTranslation_c2w      = _sim_c2w.rowRange(0,3).col(3);  //this line needs double check that if division by scale is required.
    cv::Mat simCameraOriginInWorld  = -simRotation_c2w.t()*simTranslation_c2w;
    //set of map points that are already found in the key frame
    std::set<std::shared_ptr<MapPoint>> setSptrFoundMapPoints(_vSptrMatchedMapPoints.begin(),_vSptrMatchedMapPoints.end());
    setSptrFoundMapPoints.erase(std::shared_ptr<MapPoint>(nullptr));
    int matchNum = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : _vSptrMapPoints){
      //discard bad map points and those are alreayd found
      if(sptrMapPoint && !sptrMapPoint->isBad() && !setSptrFoundMapPoints.count(sptrMapPoint)){
        //get map point 3D coordinate
        cv::Mat mapPointPosInWorld  = sptrMapPoint->getPosInWorld();
        //transform into camera coordinate
        cv::Mat mapPointPosInCamera = simRotation_c2w * mapPointPosInWorld + simTranslation_c2w;
        // Project into Image
        const float mapPointXcordInCamera = mapPointPosInCamera.at<float>(0);
        const float mapPointYcordInCamera = mapPointPosInCamera.at<float>(1);
        const float mapPointZcordInCamera = mapPointPosInCamera.at<float>(2);
        const float mapPointProjXInImage  = Frame::m_flt_fx*mapPointXcordInCamera/mapPointZcordInCamera+Frame::m_flt_cx;
        const float mapPointProjYInImage  = Frame::m_flt_fy*mapPointYcordInCamera/mapPointZcordInCamera+Frame::m_flt_cy;
        const float maxDistance           = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance           = sptrMapPoint->getMinDistanceInvariance();
        cv::Mat vecCamera2MapPointInWorld = mapPointPosInWorld - simCameraOriginInWorld;
        const float distance              = cv::norm(vecCamera2MapPointInWorld);
        int predictedScaleLevel           = sptrMapPoint->predictScaleLevel(distance,_sptrKeyFrame);
        const float radius                = _thd*_sptrKeyFrame->m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices   = _sptrKeyFrame->getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius);
        //depth should be positive && 
        //inside the scale invariance region of the point && 
        //point should be inside the image &&
        //viewing angle should be less than 60 deg
        //search should exist
        if(mapPointZcordInCamera>=0.0 &&\
        _sptrKeyFrame->isInImage(mapPointProjXInImage,mapPointProjYInImage) && \
        distance>=minDistance && distance<=maxDistance && \
        vecCamera2MapPointInWorld.dot(sptrMapPoint->getNormal())>=0.5*distance && \
        !vIndices.empty()){
          int bestDist = 256;
          int bestIdx  = -1;
          for(const int &idx : vIndices){
            if(!_vSptrMatchedMapPoints[idx] && \
            _sptrKeyFrame->m_v_keyPoints[idx].octave>=predictedScaleLevel-1 && \
            _sptrKeyFrame->m_v_keyPoints[idx].octave<=predictedScaleLevel){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_sptrKeyFrame->m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist = dist;
                bestIdx  = idx;
              }
            }
          }
          if(bestDist<=m_int_lowThd){
            _vSptrMatchedMapPoints[bestIdx]=sptrMapPoint;
            matchNum++;
          }
        }
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchByBowInKeyFrameAndFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame, Frame &_frame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints){
    const std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPointsInKeyFrame = _sptrKeyFrame->getMatchedMapPointsVec();
    _vSptrMatchedMapPoints = std::vector<std::shared_ptr<MapPoint>>(_frame.m_int_keyPointsNum,std::shared_ptr<MapPoint>(nullptr));
    int matchNum = 0;
    std::vector<std::vector<int>> rotationHist = std::vector<std::vector<int>>(m_int_histLen,std::vector<int>());
    for(std::vector<int> &rotation : rotationHist){
      rotation.reserve(500);
    }
    const float factor = 1.0/m_int_histLen;
    //perform the matching over ORB that belongs to the same vocabulary node at a certain level
    DBoW3::FeatureVector::const_iterator keyFrameIter     = _sptrKeyFrame->m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator frameIter        = _frame.m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator keyFrameIterEnd  = _sptrKeyFrame->m_bow_keyPointsVec.end();
    DBoW3::FeatureVector::const_iterator frameIterEnd     = _frame.m_bow_keyPointsVec.end();
    while(keyFrameIter!=keyFrameIterEnd && frameIter!=frameIterEnd){
      if(keyFrameIter->first == frameIter->first){
        for(const int &realKeyFrameIdx : keyFrameIter->second){
          if(vSptrMatchedMapPointsInKeyFrame[realKeyFrameIdx] && !vSptrMatchedMapPointsInKeyFrame[realKeyFrameIdx]->isBad()){
            int bestDist        = 256;
            int secondBestDist  = 256;
            int bestFrameIdx    = -1;
            for(const int &realFrameIdx : frameIter->second){
              if(!_vSptrMatchedMapPoints[realFrameIdx]){
                const int dist = computeDescriptorsDistance(_sptrKeyFrame->m_cvMat_descriptors.row(realKeyFrameIdx),_frame.m_cvMat_descriptors.row(realFrameIdx));
                if(dist<bestDist){
                  secondBestDist  = bestDist;
                  bestDist        = dist;
                  bestFrameIdx    = realFrameIdx;
                }else if(dist<secondBestDist){
                  secondBestDist  = dist;
                }
              }
            }
            if(bestDist<=m_int_lowThd && static_cast<float>(bestDist)<m_flt_bestSecondRatio*static_cast<float>(secondBestDist)){
              _vSptrMatchedMapPoints[bestFrameIdx] = vSptrMatchedMapPointsInKeyFrame[realKeyFrameIdx];
              if(m_b_isToCheckOrientation){
                float rotation = _sptrKeyFrame->m_v_keyPoints[realKeyFrameIdx].angle - _frame.m_v_keyPoints[bestFrameIdx].angle;
                if(rotation<0.0){
                  rotation += 360.0;
                }
                int tmpBestIdx = round(rotation*factor);
                if(tmpBestIdx==m_int_histLen){
                  tmpBestIdx=0;
                }
                assert(tmpBestIdx>=0 && tmpBestIdx<m_int_histLen);
                rotationHist[tmpBestIdx].push_back(bestFrameIdx);
              }
              matchNum++;
            }
          }
        }
        keyFrameIter++;
        frameIter++;
      }else if(keyFrameIter->first < frameIter->first){
        keyFrameIter = _sptrKeyFrame->m_bow_keyPointsVec.lower_bound(frameIter->first);
      }else{
        frameIter = _frame.m_bow_keyPointsVec.lower_bound(keyFrameIter->first);
      }
    }
    if(m_b_isToCheckOrientation){
      int idx1 = -1;
      int idx2 = -1;
      int idx3 = -1;
      computeThreeMaxima(rotationHist,m_int_histLen,idx1,idx2,idx3);
      int i_for = 0;
      for(const std::vector<int> &rotation : rotationHist){
        if(i_for!=idx1 && i_for!=idx2 && i_for!=idx3){
          for(const int &idx : rotation){
            _vSptrMatchedMapPoints[idx]=std::shared_ptr<MapPoint>(nullptr);
            matchNum--;
          }
        }
        i_for++;
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchByBowInTwoKeyFrames(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints){
    const std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPointsInFirstKeyFrame = _sptrFirstKeyFrame->getMatchedMapPointsVec();
    const cv::Mat descriptorsInFirstKeyFrame = _sptrFirstKeyFrame->m_cvMat_descriptors;
    const std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPointsInSecondKeyFrame = _sptrSecondKeyFrame->getMatchedMapPointsVec();
    const cv::Mat descriptorsInSecondKeyFrame = _sptrSecondKeyFrame->m_cvMat_descriptors;
    _vSptrMatchedMapPoints = std::vector<std::shared_ptr<MapPoint>>(vSptrMatchedMapPointsInFirstKeyFrame.size(),std::shared_ptr<MapPoint>(nullptr));
    std::vector<bool> vbIsMatchedInSecondKeyFrame(vSptrMatchedMapPointsInSecondKeyFrame.size(),false);
    int matchNum = 0;
    std::vector<std::vector<int>> rotationHist = std::vector<std::vector<int>>(m_int_histLen,std::vector<int>());
    for(std::vector<int> &rotation : rotationHist){
      rotation.reserve(500);
    }
    const float factor = 1.0/m_int_histLen;
    DBoW3::FeatureVector::const_iterator firstKeyFrameIter         = _sptrFirstKeyFrame->m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator secondKeyFrameIter        = _sptrSecondKeyFrame->m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator firstKeyFrameIterEnd      = _sptrFirstKeyFrame->m_bow_keyPointsVec.end();
    DBoW3::FeatureVector::const_iterator secondKeyFrameIterEnd     = _sptrSecondKeyFrame->m_bow_keyPointsVec.end();
    while(firstKeyFrameIter!=firstKeyFrameIterEnd && secondKeyFrameIter!=secondKeyFrameIterEnd){
      if(firstKeyFrameIter->first == secondKeyFrameIter->first){
        for(const int &firstKeyFrameIdx : firstKeyFrameIter->second){
          if(vSptrMatchedMapPointsInFirstKeyFrame[firstKeyFrameIdx] && !vSptrMatchedMapPointsInFirstKeyFrame[firstKeyFrameIdx]->isBad()){
            int bestDist                = 256;
            int secondKeyFrameBestDist  = 256;
            int secondKeyFrameBestIdx   = -1;
            for(const int &secondKeyFrameIdx : secondKeyFrameIter->second){
              if(!vbIsMatchedInSecondKeyFrame[secondKeyFrameIdx] && \
              vSptrMatchedMapPointsInSecondKeyFrame[secondKeyFrameIdx] && \
              !vSptrMatchedMapPointsInSecondKeyFrame[secondKeyFrameIdx]->isBad()){
                const int dist = computeDescriptorsDistance(descriptorsInFirstKeyFrame.row(firstKeyFrameIdx),descriptorsInSecondKeyFrame.row(secondKeyFrameIdx));
                if(dist<bestDist){
                  secondKeyFrameBestDist  = bestDist;
                  bestDist                = dist;
                  secondKeyFrameBestIdx   = secondKeyFrameIdx;
                }else if(dist<secondKeyFrameBestDist){
                  secondKeyFrameBestDist  = dist;
                }
              }
            }
            if(bestDist<=m_int_lowThd && static_cast<float>(bestDist)<m_flt_bestSecondRatio*static_cast<float>(secondKeyFrameBestDist)){
              _vSptrMatchedMapPoints[firstKeyFrameIdx] = vSptrMatchedMapPointsInSecondKeyFrame[secondKeyFrameBestIdx];
              vbIsMatchedInSecondKeyFrame[secondKeyFrameBestIdx] = true;
              if(m_b_isToCheckOrientation){
                float rotation = _sptrFirstKeyFrame->m_v_keyPoints[firstKeyFrameIdx].angle - _sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameBestIdx].angle;
                if(rotation<0.0){
                  rotation += 360.0;
                }
                int tmpBestIdx = round(rotation*factor);
                if(tmpBestIdx==m_int_histLen){
                  tmpBestIdx=0;
                }
                assert(tmpBestIdx>=0 && tmpBestIdx<m_int_histLen);
                rotationHist[tmpBestIdx].push_back(firstKeyFrameIdx);
              }
              matchNum++;
            }
          }
        }
        firstKeyFrameIter++;
        secondKeyFrameIter++;
      }else if(firstKeyFrameIter->first < secondKeyFrameIter->first){
        firstKeyFrameIter = _sptrFirstKeyFrame->m_bow_keyPointsVec.lower_bound(secondKeyFrameIter->first);
      }else{
        secondKeyFrameIter = _sptrSecondKeyFrame->m_bow_keyPointsVec.lower_bound(firstKeyFrameIter->first);
      }
    }
    if(m_b_isToCheckOrientation){
      int idx1 = -1;
      int idx2 = -1;
      int idx3 = -1;
      computeThreeMaxima(rotationHist,m_int_histLen,idx1,idx2,idx3);
      int i_for = 0;
      for(const std::vector<int> &rotation : rotationHist){
        if(i_for!=idx1 && i_for!=idx2 && i_for!=idx3){
          for(const int &idx : rotation){
            _vSptrMatchedMapPoints[idx]=std::shared_ptr<MapPoint>(nullptr);
            matchNum--;
          }
        }
        i_for++;
      }
    }
    return matchNum;
  }
  int OrbMatcher::searchForTriangulation(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, cv::Mat &_fMatrix_first2second, std::vector<std::pair<int,int>> &_vMatchedPairs, const bool _bIsStereoOnly){
    //compute epipole in second image
    cv::Mat firstCameraOriginInWorld = _sptrFirstKeyFrame->getCameraOriginInWorld();
    cv::Mat secondRotation_c2w       = _sptrSecondKeyFrame->getRotation_c2w();
    cv::Mat secondTranslation_c2w    = _sptrSecondKeyFrame->getTranslation_c2w();
    cv::Mat firstOriginInSecond      = secondRotation_c2w * firstCameraOriginInWorld + secondTranslation_c2w;
    const float xCord = Frame::m_flt_fx * firstOriginInSecond.at<float>(0) / firstOriginInSecond.at<float>(2) + Frame::m_flt_cx;
    const float yCord = Frame::m_flt_fy * firstOriginInSecond.at<float>(1) / firstOriginInSecond.at<float>(2) + Frame::m_flt_cy;
    //find matches between key points that are not tracked
    //speed up matching by vocabulary
    //only compare ORB that share the same node
    int matchNum = 0;
    std::vector<bool> vbIsMatchedInSecondKeyFrame(_sptrSecondKeyFrame->m_int_keyPointsNum,false);
    std::vector<int> vFirstKeyFrameMatchedKeyPointsInSecondKeyFrame(_sptrFirstKeyFrame->m_int_keyPointsNum,-1);
    std::vector<std::vector<int>> rotationHist = std::vector<std::vector<int>>(m_int_histLen,std::vector<int>());
    for(std::vector<int> &rotation : rotationHist){
      rotation.reserve(500);
    }
    const float factor = 1.0/m_int_histLen;
    DBoW3::FeatureVector::const_iterator firstKeyFrameIter         = _sptrFirstKeyFrame->m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator secondKeyFrameIter        = _sptrSecondKeyFrame->m_bow_keyPointsVec.begin();
    DBoW3::FeatureVector::const_iterator firstKeyFrameIterEnd      = _sptrFirstKeyFrame->m_bow_keyPointsVec.end();
    DBoW3::FeatureVector::const_iterator secondKeyFrameIterEnd     = _sptrSecondKeyFrame->m_bow_keyPointsVec.end();
    while(firstKeyFrameIter!=firstKeyFrameIterEnd && secondKeyFrameIter!=secondKeyFrameIterEnd){
      if(firstKeyFrameIter->first == secondKeyFrameIter->first){
        for(const int &firstKeyFrameIdx : firstKeyFrameIter->second){
          const bool bIsFirstStereoGood = _sptrFirstKeyFrame->m_v_rightXcords[firstKeyFrameIdx]>=0;
          //if there is not a map point yet and (not stereo only or first stereo is good)
          if(!_sptrFirstKeyFrame->getMapPoint(firstKeyFrameIdx) && (!_bIsStereoOnly || bIsFirstStereoGood)){
            const cv::KeyPoint &firstKeyPoint = _sptrFirstKeyFrame->m_v_keyPoints[firstKeyFrameIdx];
            int bestDist              = m_int_lowThd;
            int secondKeyFrameBestIdx = -1;
            for(const int &secondKeyFrameIdx : secondKeyFrameIter->second){
              const bool bIsSecondStereoGood = _sptrSecondKeyFrame->m_v_rightXcords[secondKeyFrameIdx]>=0;
              if(!vbIsMatchedInSecondKeyFrame[secondKeyFrameIdx] && \
              !_sptrSecondKeyFrame->getMapPoint(secondKeyFrameIdx) && \
              (!_bIsStereoOnly || bIsSecondStereoGood)){
                const int dist = computeDescriptorsDistance(_sptrFirstKeyFrame->m_cvMat_descriptors.row(firstKeyFrameIdx),_sptrSecondKeyFrame->m_cvMat_descriptors.row(secondKeyFrameIdx));
                const cv::KeyPoint &secondKeyPoint = _sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx];
                if(dist<=m_int_lowThd && dist<=bestDist && (bIsFirstStereoGood || bIsSecondStereoGood || \
                pow((float)xCord-(float)(_sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].pt.x),2.0) + \
                pow((float)yCord-(float)(_sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].pt.y),2.0) >= \
                100*_sptrSecondKeyFrame->m_v_scaleFactors[secondKeyPoint.octave])){
                  if(isEpipolarLineDistCorrect(firstKeyPoint,secondKeyPoint,_fMatrix_first2second,_sptrSecondKeyFrame)){
                    secondKeyFrameBestIdx = secondKeyFrameIdx;
                    bestDist = dist;
                  }
                }
              }
            }
            if(secondKeyFrameBestIdx>=0){
              const cv::KeyPoint &secondKeyPoint = _sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameBestIdx];
              vbIsMatchedInSecondKeyFrame[secondKeyFrameBestIdx] = true;
              vFirstKeyFrameMatchedKeyPointsInSecondKeyFrame[firstKeyFrameIdx] = secondKeyFrameBestIdx;
              matchNum++;
              if(m_b_isToCheckOrientation){
                float rotation = firstKeyPoint.angle-secondKeyPoint.angle;
                if(rotation<0.0)
                    rotation+=360.0f;
                int tmpBestIdx = round(rotation*factor);
                if(tmpBestIdx==m_int_histLen)
                    tmpBestIdx=0;
                assert(tmpBestIdx>=0 && tmpBestIdx<m_int_histLen);
                rotationHist[tmpBestIdx].push_back(firstKeyFrameIdx);
              }
            }
          }
        }
        firstKeyFrameIter++;
        secondKeyFrameIter++;
      }else if(firstKeyFrameIter->first < secondKeyFrameIter->first){
        firstKeyFrameIter = _sptrFirstKeyFrame->m_bow_keyPointsVec.lower_bound(secondKeyFrameIter->first);
      }else{
        secondKeyFrameIter = _sptrSecondKeyFrame->m_bow_keyPointsVec.lower_bound(firstKeyFrameIter->first);
      }
    }
    if(m_b_isToCheckOrientation){
      int idx1 = -1;
      int idx2 = -1;
      int idx3 = -1;
      computeThreeMaxima(rotationHist,m_int_histLen,idx1,idx2,idx3);
      int i_for = 0;
      for(const std::vector<int> &rotation : rotationHist){
        if(i_for!=idx1 && i_for!=idx2 && i_for!=idx3){
          for(const int &idx : rotation){
            vFirstKeyFrameMatchedKeyPointsInSecondKeyFrame[idx] = -1;
            matchNum--;
          }
        }
        i_for++;
      }
    }
    _vMatchedPairs.clear();
    _vMatchedPairs.reserve(matchNum);
    int i_for = 0;
    for(const int &idx : vFirstKeyFrameMatchedKeyPointsInSecondKeyFrame){
      if(idx>=0){
        _vMatchedPairs.push_back(std::make_pair(i_for,idx));
      }
      i_for++;
    }
    return matchNum;
  }
  int OrbMatcher::searchBySim3(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const float &_scale_first2second, const cv::Mat &_rotation_first2second, const cv::Mat &_translation_first2second, const float _thd){
    cv::Mat rotation_firstC2w     = _sptrFirstKeyFrame->getRotation_c2w();
    cv::Mat translation_firstC2w  = _sptrFirstKeyFrame->getTranslation_c2w();
    cv::Mat rotation_secondC2w    = _sptrSecondKeyFrame->getRotation_c2w();
    cv::Mat translation_secondC2w = _sptrSecondKeyFrame->getTranslation_c2w();
    //sim transformation between two frames
    cv::Mat simRotation_firstC2secondC = _scale_first2second * _rotation_first2second;
    cv::Mat simRotation_secondC2firstC = _rotation_first2second.t()/_scale_first2second;
    cv::Mat translation_secondC2firstC = -simRotation_secondC2firstC * _translation_first2second;
    const std::vector<std::shared_ptr<MapPoint>> vSptrFirstKeyFrameMatchedMapPoints  = _sptrFirstKeyFrame->getMatchedMapPointsVec();
    const std::vector<std::shared_ptr<MapPoint>> vSptrSecondKeyFrameMatchedMapPoints = _sptrSecondKeyFrame->getMatchedMapPointsVec();
    std::vector<bool> vIsMatchedInFirstKeyFrame(vSptrFirstKeyFrameMatchedMapPoints.size(),false);
    std::vector<bool> vIsMatchedInSecondKeyFrame(vSptrSecondKeyFrameMatchedMapPoints.size(),false);
    std::vector<int> vFirstKeyFrameMatches(vSptrFirstKeyFrameMatchedMapPoints.size(),-1);
    std::vector<int> vSecondKeyFrameMatches(vSptrSecondKeyFrameMatchedMapPoints.size(),-1);
    int i_for = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : _vSptrMatchedMapPoints){
      if(sptrMapPoint){
        vIsMatchedInFirstKeyFrame[i_for]=true;
        const int idx = sptrMapPoint->getIdxInKeyFrame(_sptrSecondKeyFrame);
        if(idx>=0 && idx<vSptrSecondKeyFrameMatchedMapPoints.size())
          vIsMatchedInSecondKeyFrame[idx]=true;
      }
      i_for++;
    }
    //transform from first key frame to second key frame and search
    i_for = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : vSptrFirstKeyFrameMatchedMapPoints){
      if(sptrMapPoint && !vIsMatchedInFirstKeyFrame[i_for] && !sptrMapPoint->isBad()){
        cv::Mat mapPointPosInWorld        = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInSecondCamera = rotation_secondC2w * mapPointPosInWorld + translation_secondC2w;
        const float mapPointXcordInSecondCamera = mapPointPosInSecondCamera.at<float>(0);
        const float mapPointYcordInSecondCamera = mapPointPosInSecondCamera.at<float>(1);
        const float mapPointZcordInSecondCamera = mapPointPosInSecondCamera.at<float>(2);
        const float mapPointProjXInSecondImage  = Frame::m_flt_fx*mapPointXcordInSecondCamera/mapPointZcordInSecondCamera+Frame::m_flt_cx;
        const float mapPointProjYInSecondImage  = Frame::m_flt_fy*mapPointYcordInSecondCamera/mapPointZcordInSecondCamera+Frame::m_flt_cy;
        const float dist3D = cv::norm(mapPointPosInSecondCamera);
        const float maxDistance = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance = sptrMapPoint->getMinDistanceInvariance();
        const int predictedScaleLevel = sptrMapPoint->predictScaleLevel(dist3D,_sptrSecondKeyFrame);
        const float radius = _thd*_sptrSecondKeyFrame->m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices = _sptrSecondKeyFrame->getKeyPointsInArea(mapPointProjXInSecondImage,mapPointProjYInSecondImage,radius);
        if(mapPointZcordInSecondCamera>=0.0 && \
        _sptrSecondKeyFrame->isInImage(mapPointProjXInSecondImage,mapPointProjYInSecondImage) && \
        dist3D>=minDistance && dist3D<=maxDistance && \
        !vIndices.empty()){
          //match to the most similar key point in the radius
          int bestDist = 256;
          int bestIdx = -1;
          for(const int &idx : vIndices){
            if(_sptrSecondKeyFrame->m_v_keyPoints[idx].octave>=predictedScaleLevel-1 && _sptrSecondKeyFrame->m_v_keyPoints[idx].octave<=predictedScaleLevel){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_sptrSecondKeyFrame->m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
              }
            }
          }
          if(bestDist<=m_int_highThd){
            vFirstKeyFrameMatches[i_for]=bestIdx;
          }
        }
      }
      i_for++;
    }
    //transform from second key frame to first key frame and search
    i_for = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : vSptrSecondKeyFrameMatchedMapPoints){
      if(sptrMapPoint && !vIsMatchedInSecondKeyFrame[i_for] && !sptrMapPoint->isBad()){
        cv::Mat mapPointPosInWorld        = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInFirstCamera  = rotation_firstC2w * mapPointPosInWorld + translation_firstC2w;
        const float mapPointXcordInFirstCamera = mapPointPosInFirstCamera.at<float>(0);
        const float mapPointYcordInFirstCamera = mapPointPosInFirstCamera.at<float>(1);
        const float mapPointZcordInFirstCamera = mapPointPosInFirstCamera.at<float>(2);
        const float mapPointProjXInFirstImage  = Frame::m_flt_fx*mapPointXcordInFirstCamera/mapPointZcordInFirstCamera+Frame::m_flt_cx;
        const float mapPointProjYInFirstImage  = Frame::m_flt_fy*mapPointYcordInFirstCamera/mapPointZcordInFirstCamera+Frame::m_flt_cy;
        const float dist3D = cv::norm(mapPointPosInFirstCamera);
        const float maxDistance = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance = sptrMapPoint->getMinDistanceInvariance();
        const int predictedScaleLevel = sptrMapPoint->predictScaleLevel(dist3D,_sptrFirstKeyFrame);
        const float radius = _thd*_sptrFirstKeyFrame->m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices = _sptrFirstKeyFrame->getKeyPointsInArea(mapPointProjXInFirstImage,mapPointProjYInFirstImage,radius);
        if(mapPointZcordInFirstCamera>=0.0 && _sptrFirstKeyFrame->isInImage(mapPointProjXInFirstImage,mapPointProjYInFirstImage) && dist3D>=minDistance && dist3D<=maxDistance && !vIndices.empty()){
          //match to the most similar key point in the radius
          int bestDist = 256;
          int bestIdx = -1;
          for(const int &idx : vIndices){
            if(_sptrFirstKeyFrame->m_v_keyPoints[idx].octave>=predictedScaleLevel-1 && _sptrFirstKeyFrame->m_v_keyPoints[idx].octave<=predictedScaleLevel){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_sptrFirstKeyFrame->m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
              }
            }
          }
          if(bestDist<=m_int_highThd){
            vSecondKeyFrameMatches[i_for]=bestIdx;
          }
        }
      }
      i_for++;
    }
    //check agreement
    int foundNum = 0;
    for(int i=0;i<vSptrFirstKeyFrameMatchedMapPoints.size();i++){
      int secondKeyFrameIdx = vFirstKeyFrameMatches[i];
      if(secondKeyFrameIdx>=0){
        int firstKeyFrameIdx = vSecondKeyFrameMatches[secondKeyFrameIdx];
        if(firstKeyFrameIdx==i){
          _vSptrMatchedMapPoints[i] = vSptrSecondKeyFrameMatchedMapPoints[secondKeyFrameIdx];
          foundNum++;
        }
      }
    }
    return foundNum;
  }
  int OrbMatcher::FuseByProjection(std::shared_ptr<KeyFrame> _sptrKeyFrame, const std::vector<std::shared_ptr<MapPoint>> _vSptrMapPoints, const float _thd){
    cv::Mat rotation_c2w        = _sptrKeyFrame->getRotation_c2w();
    cv::Mat translation_c2w     = _sptrKeyFrame->getTranslation_c2w();
    cv::Mat cameraOriginInWorld = _sptrKeyFrame->getCameraOriginInWorld();
    int fuseNum = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : _vSptrMapPoints){
      if(sptrMapPoint && !sptrMapPoint->isBad() && !sptrMapPoint->isInKeyFrame(_sptrKeyFrame)){
        cv::Mat mapPointPosInWorld        = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInCamera       = rotation_c2w * mapPointPosInWorld + translation_c2w;
        const float mapPointXcordInCamera = mapPointPosInCamera.at<float>(0);
        const float mapPointYcordInCamera = mapPointPosInCamera.at<float>(1);
        const float mapPointZcordInCamera = mapPointPosInCamera.at<float>(2);
        const float mapPointProjXInImage  = Frame::m_flt_fx*mapPointXcordInCamera/mapPointZcordInCamera+Frame::m_flt_cx;
        const float mapPointProjYInImage  = Frame::m_flt_fy*mapPointYcordInCamera/mapPointZcordInCamera+Frame::m_flt_cy;
        const float mapPointProjXInRightImage  = mapPointProjXInImage-Frame::m_flt_baseLineTimesFx/mapPointZcordInCamera;
        const float maxDistance = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance = sptrMapPoint->getMinDistanceInvariance();
        cv::Mat vecCamera2MapPointInWorld = mapPointPosInWorld - cameraOriginInWorld;
        const float dist3D = cv::norm(vecCamera2MapPointInWorld);
        const int predictedScaleLevel = sptrMapPoint->predictScaleLevel(dist3D,_sptrKeyFrame);
        const float radius = _thd*_sptrKeyFrame->m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices = _sptrKeyFrame->getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius);
        if(mapPointZcordInCamera>=0.0 && \
        _sptrKeyFrame->isInImage(mapPointProjXInImage,mapPointProjYInImage) && \
        dist3D>=minDistance && dist3D<=maxDistance && \
        vecCamera2MapPointInWorld.dot(sptrMapPoint->getNormal())>=0.5*dist3D && \
        !vIndices.empty()){
          int bestDist = 256;
          int bestIdx = -1;
          for(const int &idx : vIndices){
            const int &scaleLevel = _sptrKeyFrame->m_v_keyPoints[idx].octave;
            const float monoSquaredErr   = pow(_sptrKeyFrame->m_v_keyPoints[idx].pt.x - mapPointProjXInImage,2.0) + \
                                           pow(_sptrKeyFrame->m_v_keyPoints[idx].pt.y - mapPointProjYInImage,2.0);
            const float stereoSquaredErr = monoSquaredErr + pow(_sptrKeyFrame->m_v_rightXcords[idx] - mapPointProjXInRightImage,2.0);
            if(scaleLevel>=predictedScaleLevel-1 && \
            scaleLevel<=predictedScaleLevel && \
            ((_sptrKeyFrame->m_v_rightXcords[idx]>=0 && stereoSquaredErr*_sptrKeyFrame->m_v_invScaleFactorSquares[scaleLevel]<=7.81) || \
            (_sptrKeyFrame->m_v_rightXcords[idx]<0 && monoSquaredErr*_sptrKeyFrame->m_v_invScaleFactorSquares[scaleLevel]<=5.99))){
              const int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_sptrKeyFrame->m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
              }
            }
          }
          if(bestDist<=m_int_lowThd){
            std::shared_ptr<MapPoint> sptrMapPointInKeyFrame = _sptrKeyFrame->getMapPoint(bestIdx);
            if(sptrMapPointInKeyFrame){
              if(!sptrMapPointInKeyFrame->isBad()&&sptrMapPointInKeyFrame->getObservationsNum()>sptrMapPoint->getObservationsNum()){
                sptrMapPoint->beReplacedBy(sptrMapPointInKeyFrame);
              }else if(!sptrMapPointInKeyFrame->isBad()&&sptrMapPointInKeyFrame->getObservationsNum()<=sptrMapPoint->getObservationsNum()){
                sptrMapPointInKeyFrame->beReplacedBy(sptrMapPoint);
              }
            }else {
              sptrMapPoint->addObservation(_sptrKeyFrame,bestIdx);
              _sptrKeyFrame->addMapPoint(sptrMapPoint,bestIdx);
            }
            fuseNum++;  //if map point is bad why still fuseNum++?
          }
        }
      }
    }
    return fuseNum;
  }
  int OrbMatcher::FuseBySim3(std::shared_ptr<KeyFrame> _sptrKeyFrame, cv::Mat &_sim_c2w, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, float _thd, std::vector<std::shared_ptr<MapPoint>> &_vSptrReplacement){
    //decompose _sim_c2w
    cv::Mat simScaledRotation_c2w   = _sim_c2w.rowRange(0,3).colRange(0,3);
    const float simScale_c2w        = sqrt(simScaledRotation_c2w.row(0).dot(simScaledRotation_c2w.row(0)));
    cv::Mat simRotation_c2w         = simScaledRotation_c2w/simScale_c2w;
    cv::Mat simTranslation_c2w      = _sim_c2w.rowRange(0,3).col(3);
    cv::Mat simCameraOriginInWorld  = -simRotation_c2w.t()*simTranslation_c2w;
    int fuseNum = 0;
    //set of map points already found in key frame
    const std::set<std::shared_ptr<MapPoint>> setSptrFoundMapPoint = _sptrKeyFrame->getMatchedMapPointsSet();
    int i_for = 0;
    for(std::shared_ptr<MapPoint> sptrMapPoint : _vSptrMapPoints){
      if(sptrMapPoint && !sptrMapPoint->isBad() && !setSptrFoundMapPoint.count(sptrMapPoint)){
        cv::Mat mapPointPosInWorld        = sptrMapPoint->getPosInWorld();
        cv::Mat mapPointPosInCamera       = simRotation_c2w * mapPointPosInWorld + simTranslation_c2w;
        const float mapPointXcordInCamera = mapPointPosInCamera.at<float>(0);
        const float mapPointYcordInCamera = mapPointPosInCamera.at<float>(1);
        const float mapPointZcordInCamera = mapPointPosInCamera.at<float>(2);
        const float mapPointProjXInImage  = Frame::m_flt_fx*mapPointXcordInCamera/mapPointZcordInCamera+Frame::m_flt_cx;
        const float mapPointProjYInImage  = Frame::m_flt_fy*mapPointYcordInCamera/mapPointZcordInCamera+Frame::m_flt_cy;
        const float maxDistance = sptrMapPoint->getMaxDistanceInvariance();
        const float minDistance = sptrMapPoint->getMinDistanceInvariance();
        cv::Mat vecCamera2MapPointInWorld = mapPointPosInWorld - simCameraOriginInWorld;
        const float dist3D = cv::norm(vecCamera2MapPointInWorld);
        const int predictedScaleLevel = sptrMapPoint->predictScaleLevel(dist3D,_sptrKeyFrame);
        const float radius = _thd*_sptrKeyFrame->m_v_scaleFactors[predictedScaleLevel];
        const std::vector<int> vIndices = _sptrKeyFrame->getKeyPointsInArea(mapPointProjXInImage,mapPointProjYInImage,radius);
        if(mapPointZcordInCamera>=0.0 && \
        _sptrKeyFrame->isInImage(mapPointProjXInImage,mapPointProjYInImage) && \
        dist3D>=minDistance && dist3D<=maxDistance && \
        vecCamera2MapPointInWorld.dot(sptrMapPoint->getNormal())>=0.5*dist3D && \
        !vIndices.empty()){
          int bestDist = 256;
          int bestIdx = -1;
          for(const int &idx : vIndices){
            const int &scaleLevel = _sptrKeyFrame->m_v_keyPoints[idx].octave;
            if(scaleLevel>=predictedScaleLevel-1 && scaleLevel<=predictedScaleLevel){
              int dist = computeDescriptorsDistance(sptrMapPoint->getDescriptor(),_sptrKeyFrame->m_cvMat_descriptors.row(idx));
              if(dist<bestDist){
                bestDist = dist;
                bestIdx = idx;
              }
            }
          }
          if(bestDist<=m_int_lowThd){
            std::shared_ptr<MapPoint> sptrMapPointInKeyFrame = _sptrKeyFrame->getMapPoint(bestIdx);
            if(sptrMapPointInKeyFrame){
              if(!sptrMapPointInKeyFrame->isBad()&&sptrMapPointInKeyFrame->getObservationsNum()>sptrMapPoint->getObservationsNum()){
                sptrMapPoint->beReplacedBy(sptrMapPointInKeyFrame);
              }else if(!sptrMapPointInKeyFrame->isBad()&&sptrMapPointInKeyFrame->getObservationsNum()<=sptrMapPoint->getObservationsNum()){
                _vSptrReplacement[i_for]=sptrMapPointInKeyFrame;
              }
            }else {
              sptrMapPoint->addObservation(_sptrKeyFrame,bestIdx);
              _sptrKeyFrame->addMapPoint(sptrMapPoint,bestIdx);
            }
            fuseNum++;
          }
        }
      }
      i_for++;
    }
    return fuseNum;
  }
  bool OrbMatcher::isEpipolarLineDistCorrect(const cv::KeyPoint &_firstKeyPoint, const cv::KeyPoint &_secondKeyPoint, const cv::Mat &_fMatrix_first2second, const std::shared_ptr<KeyFrame> _sptrKeyFrame){
    //epipolar line in the second image, l = _fMatrix_first2second * x1 = [a, b, c]'
    const float a = _fMatrix_first2second.at<float>(0,0)*_firstKeyPoint.pt.x+_fMatrix_first2second.at<float>(1,0)*_firstKeyPoint.pt.y+_fMatrix_first2second.at<float>(2,0);
    const float b = _fMatrix_first2second.at<float>(0,1)*_firstKeyPoint.pt.x+_fMatrix_first2second.at<float>(1,1)*_firstKeyPoint.pt.y+_fMatrix_first2second.at<float>(2,1);
    const float c = _fMatrix_first2second.at<float>(0,2)*_firstKeyPoint.pt.x+_fMatrix_first2second.at<float>(1,2)*_firstKeyPoint.pt.y+_fMatrix_first2second.at<float>(2,2);
    if((a*a+b*b)>0){
      return ((((a * _secondKeyPoint.pt.x + b * _secondKeyPoint.pt.y + c) * (a * _secondKeyPoint.pt.x + b * _secondKeyPoint.pt.y + c)) / ((a*a+b*b) * (a*a+b*b))) < \
              (3.841 * _sptrKeyFrame->m_v_scaleFactorSquares[_secondKeyPoint.octave]));
    }else {
      return false;
    }
  }
  float OrbMatcher::getRadiusByViewCos(const float &_viewCos){
    if(_viewCos>0.998){
      return 2.5;
    }else {
      return 4.0;
    }
  }
  void OrbMatcher::computeThreeMaxima(std::vector<std::vector<int>> &_hist, const int &_len, int &_idx1, int &_idx2, int &_idx3){
    int max1=0, max2=0, max3=0;
    for(int i=0;i<_len;i++){
      const int histogram_len = _hist[i].size();
      if(histogram_len>max1){
        max3=max2;
        max2=max1;
        max1=histogram_len;
        _idx3=_idx2;
        _idx2=_idx1;
        _idx1=i;
      }else if(histogram_len>max2){
        max3=max2;
        max2=histogram_len;
        _idx3=_idx2;
        _idx2=i;
      }else if(histogram_len>max3){
        max3=histogram_len;
        _idx3=i;
      }
    }
    if(max2<max1/10){
      _idx3=-1;
      _idx2=-1;
    }else if(max3<max1/10){
      _idx3=-1;
    }
  }
}//namespace YDORBSLAM
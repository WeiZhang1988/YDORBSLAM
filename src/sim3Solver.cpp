#include "sim3Solver.hpp"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "keyFrame.hpp"
#include "orbMatcher.hpp"

namespace YDORBSLAM{
  Sim3Solver::Sim3Solver(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const bool _bIsScaleFixed):
  m_sptr_firstKeyFrame(_sptrFirstKeyFrame),m_sptr_secondKeyFrame(_sptrSecondKeyFrame),m_v_matchedMapPoints(_vSptrMatchedMapPoints),m_b_isScaleFixed(_bIsScaleFixed){
    std::vector<std::shared_ptr<MapPoint>> vFirstKeyFrameMapPoints = m_sptr_firstKeyFrame->getMatchedMapPointsVec();
    m_int_matchedMapPointsNum = m_v_matchedMapPoints.size();
    m_v_firstMapPoints.reserve(m_int_matchedMapPointsNum);
    m_v_secondMapPoints.reserve(m_int_matchedMapPointsNum);
    m_v_firstIndices.reserve(m_int_matchedMapPointsNum);
    m_v_firstMapPointsPosInCamera.reserve(m_int_matchedMapPointsNum);
    m_v_secondMapPointsPosInCamera.reserve(m_int_matchedMapPointsNum);
    m_v_allIndices.reserve(m_int_matchedMapPointsNum);
    cv::Mat firstRotation_c2w = m_sptr_firstKeyFrame->getRotation_c2w();
    cv::Mat firstTranslation_c2w = m_sptr_firstKeyFrame->getTranslation_c2w();
    cv::Mat secondRotation_c2w = m_sptr_secondKeyFrame->getRotation_c2w();
    cv::Mat secondTranslation_c2w = m_sptr_secondKeyFrame->getTranslation_c2w();
    int idx = 0;
    for(int i1=0;i1<m_int_matchedMapPointsNum;i1++){
      if(m_v_matchedMapPoints[i1]){
        std::shared_ptr<MapPoint> sptrFirstMapPoint = vFirstKeyFrameMapPoints[i1];
        std::shared_ptr<MapPoint> sptrSecondMapPoint = m_v_matchedMapPoints[i1];
        if(sptrFirstMapPoint && !sptrFirstMapPoint->isBad() && sptrSecondMapPoint && !sptrSecondMapPoint->isBad()){
          int firstKeyFrameIdx = sptrFirstMapPoint->getIdxInKeyFrame(m_sptr_firstKeyFrame);
          int secondKeyFrameIdx = sptrSecondMapPoint->getIdxInKeyFrame(m_sptr_secondKeyFrame);
          if(firstKeyFrameIdx>=0 && secondKeyFrameIdx>=0){
            m_v_firstMaxError.push_back(9.210*m_sptr_firstKeyFrame->m_v_scaleFactorSquares[m_sptr_firstKeyFrame->m_v_keyPoints[firstKeyFrameIdx].octave]);
            m_v_secondMaxError.push_back(9.210*m_sptr_secondKeyFrame->m_v_scaleFactorSquares[m_sptr_secondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].octave]);
            m_v_firstMapPoints.push_back(sptrFirstMapPoint);
            m_v_secondMapPoints.push_back(sptrSecondMapPoint);
            m_v_firstIndices.push_back(i1);
            cv::Mat firstMapPointPosInWorld = sptrFirstMapPoint->getPosInWorld();
            m_v_firstMapPointsPosInCamera.push_back(firstRotation_c2w*firstMapPointPosInWorld+firstTranslation_c2w);
            cv::Mat secondMapPointPosInWorld = sptrSecondMapPoint->getPosInWorld();
            m_v_secondMapPointsPosInCamera.push_back(secondRotation_c2w*secondMapPointPosInWorld+secondTranslation_c2w);
            m_v_allIndices.push_back(idx);
            idx++;
          }
        }
      }
    }
    m_cvMat_firstintParMat = m_sptr_firstKeyFrame->m_cvMat_intParMat;
    m_cvMat_secondintParMat = m_sptr_secondKeyFrame->m_cvMat_intParMat;
    transferCamera2Image(m_v_firstMapPointsPosInCamera,m_v_P1im1,m_cvMat_firstintParMat);
    transferCamera2Image(m_v_secondMapPointsPosInCamera,m_v_P2im2,m_cvMat_secondintParMat);
    setRansacParameters();
  }
  void Sim3Solver::setRansacParameters(float _probability, int _minInliersNum, int _maxIterNum){
    m_flt_ransacProb = _probability;
    m_int_minRansacInliersNum = _minInliersNum;
    m_int_maxRansacIterNum = _maxIterNum;
    m_int_correspondencesNum = m_v_firstMapPoints.size(); //number of correspondeces
    m_v_isInliers.resize(m_int_correspondencesNum);
    //adjust parameters according to number of correspondeces
    float epsilon = (float)m_int_minRansacInliersNum/(float)m_int_correspondencesNum;
    //set ransac iterations according to probability, epsilon and max iterations number
    int iterNum;
    if(m_int_minRansacInliersNum==m_int_correspondencesNum){
      iterNum = 1;
    }else{
      iterNum=std::ceil(std::log(1.0f-m_flt_ransacProb)/std::log(1.0f-std::pow(epsilon,3.0f)));
    }
    m_int_maxRansacIterNum = std::max(1,std::min(iterNum,m_int_maxRansacIterNum));
    m_int_iterNum = 0;
  }
  cv::Mat Sim3Solver::iterate(int _iterNum, bool &_bIsNoMore, std::vector<bool> &_vIsInliers, int &_inliersNum){
    _bIsNoMore = false;
    _vIsInliers = std::vector<bool>(m_int_matchedMapPointsNum,false);
    _inliersNum = 0;
    if(m_int_correspondencesNum<m_int_minRansacInliersNum){
      _bIsNoMore = true;
      return cv::Mat();
    }
    std::vector<int> vAvailableIndices;
    cv::Mat firstMapPoint3DInCamera(3, 3, CV_32F);
    cv::Mat secondMapPoint3DInCamera(3, 3, CV_32F);
    int currentIterNum = 0;
    while(m_int_iterNum<m_int_maxRansacIterNum && currentIterNum<_iterNum){
      currentIterNum++;
      m_int_iterNum++;
      vAvailableIndices = m_v_allIndices;
      //get min set of points
      for(int i=0;i<3;i++){
        int randIdx = std::rand()%vAvailableIndices.size();
        int idx = vAvailableIndices[randIdx];
        m_v_firstMapPointsPosInCamera[idx].copyTo(firstMapPoint3DInCamera.col(i));
        m_v_secondMapPointsPosInCamera[idx].copyTo(secondMapPoint3DInCamera.col(i));
        vAvailableIndices[randIdx] = vAvailableIndices.back();
        vAvailableIndices.pop_back();
      }
      computeSim3(firstMapPoint3DInCamera,secondMapPoint3DInCamera);
      checkInliers();
      if(m_int_inliersNum>=m_int_bestInliersNum){
        m_v_isBestInliers = m_v_isInliers;
        m_int_bestInliersNum = m_int_inliersNum;
        m_cvMat_bestTransform_first2second = m_cvMat_currentTransform_first2second.clone();
        m_cvMat_bestRotation_first2second = m_cvMat_currentRotation_first2second.clone();
        m_cvMat_bestTranslation_first2second = m_cvMat_currentTranslation_first2second.clone();
        m_flt_bestScale = m_flt_currentScale_first2second;
        if(m_int_inliersNum>m_int_minRansacInliersNum){
          _inliersNum = m_int_inliersNum;
          for(int i=0;i<m_int_correspondencesNum;i++){
            if(m_v_isInliers[i]){
              _vIsInliers[m_v_firstIndices[i]] = true;
            }
          }
          return m_cvMat_bestTransform_first2second;
        }
      }
    }
    if(m_int_iterNum>=m_int_maxRansacIterNum){
      _bIsNoMore = true;
    }
    return cv::Mat();
  }
}//namespace YDORBSLAM
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
    m_cvMat_firstIntParMat = m_sptr_firstKeyFrame->m_cvMat_intParMat;
    m_cvMat_secondIntParMat = m_sptr_secondKeyFrame->m_cvMat_intParMat;
    transferCamera2Image(m_v_firstMapPointsPosInCamera,m_v_firstPointInFirstImage,m_cvMat_firstIntParMat);
    transferCamera2Image(m_v_secondMapPointsPosInCamera,m_v_secondPointInSecondImage,m_cvMat_secondIntParMat);
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
  cv::Mat Sim3Solver::find(std::vector<bool> &_vIsInliers, int &_inliersNum){
    bool bIsNoMore;
    return iterate(m_int_maxRansacIterNum,bIsNoMore,_vIsInliers,_inliersNum);
  }
  void Sim3Solver::computeCentroid(cv::Mat &_mapPoint3DInCamera, cv::Mat &_mapPoint3DRel2Centroid, cv::Mat &_centroid){
    cv::reduce(_mapPoint3DInCamera,_centroid,1,CV_REDUCE_SUM);
    _centroid = _centroid / _mapPoint3DInCamera.cols;
    for(int i=0;i<_mapPoint3DInCamera.cols;i++){
      _mapPoint3DRel2Centroid.col(i) = _mapPoint3DInCamera.col(i) - _centroid;
    }
  }
  void Sim3Solver::computeSim3(cv::Mat &_firstMapPoint3DInCamera, cv::Mat &_secondMapPoint3DInCamera){
    //custom implementation of:
    //Horn 1987, closed-form solution of absolute orientataion using unit quaternions
    //step 1: centroid and relative coordinates
    cv::Mat firstMapPoint3DRel2Centroid(_firstMapPoint3DInCamera.size(),_firstMapPoint3DInCamera.type());//relative coordinates to centroid (set 1)
    cv::Mat secondMapPoint3DRel2Centroid(_secondMapPoint3DInCamera.size(),_secondMapPoint3DInCamera.type());//relative coordinates to centroid (set 2)
    cv::Mat firstCentroid(3,1,firstMapPoint3DRel2Centroid.type());//centroid of _firstMapPoint3DInCamera
    cv::Mat secondCentroid(3,1,secondMapPoint3DRel2Centroid.type());//centroid of _secondMapPoint3DInCamera
    computeCentroid(_firstMapPoint3DInCamera,firstMapPoint3DRel2Centroid,firstCentroid);
    computeCentroid(_secondMapPoint3DInCamera,secondMapPoint3DRel2Centroid,secondCentroid);
    //step 2: compute M matrix
    cv::Mat M = secondMapPoint3DRel2Centroid*firstMapPoint3DRel2Centroid.t();
    //step 3: compute N matrix
    float N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;
    cv::Mat N(4,4,_firstMapPoint3DInCamera.type());
    N11 =  M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 =  M.at<float>(1,2)-M.at<float>(2,1);
    N13 =  M.at<float>(2,0)-M.at<float>(0,2);
    N14 =  M.at<float>(0,1)-M.at<float>(1,0);
    N22 =  M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 =  M.at<float>(0,1)+M.at<float>(1,0);
    N24 =  M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 =  M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);
    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);
    //step 4: eigenvector of the highest eigenvalue
    cv::Mat eval, evec;
    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation
    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)
    //rotation angle. sin is the norm of the imaginary part, cos is the real part
    float ang=atan2(norm(vec),evec.at<float>(0,0));
    vec = 2*ang*vec/norm(vec); //angle-axis representation. quaternion angle is the half
    m_cvMat_currentRotation_first2second.create(3,3,_firstMapPoint3DInCamera.type());
    cv::Rodrigues(vec,m_cvMat_currentRotation_first2second); //computes the rotation matrix from angle-axis
    //step 5: rotate set 2
    cv::Mat P3 = m_cvMat_currentRotation_first2second*secondMapPoint3DRel2Centroid;
    //step 6: scale
    if(m_b_isScaleFixed){
      m_flt_currentScale_first2second = 1.0f;
    }else{
      float nom = firstMapPoint3DRel2Centroid.dot(P3);
      cv::Mat aux_P3(P3.size(),P3.type());
      aux_P3 = P3;
      cv::pow(P3,2,aux_P3);
      float den = 0;
      for(int i=0;i<aux_P3.rows;i++){
        for(int j=0;j<aux_P3.cols;j++){
          den+=aux_P3.at<float>(i,j);
        }
      }
      m_flt_currentScale_first2second = nom/den;
    }
    //step 7: translation
    m_cvMat_currentTranslation_first2second.create(1 ,3, _firstMapPoint3DInCamera.type());
    m_cvMat_currentTranslation_first2second = firstCentroid - m_flt_currentScale_first2second * m_cvMat_currentRotation_first2second * secondCentroid;
    //step 8: transformation
    //step 8.1 T12
    m_cvMat_currentTransform_first2second = cv::Mat::eye(4, 4, _firstMapPoint3DInCamera.type());
    cv::Mat sR = m_flt_currentScale_first2second * m_cvMat_currentRotation_first2second;
    sR.copyTo(m_cvMat_currentTransform_first2second.rowRange(0,3).colRange(0,3));
    m_cvMat_currentTranslation_first2second.copyTo(m_cvMat_currentTransform_first2second.rowRange(0,3).col(3));
    //step 8.2 T21
    m_cvMat_currentTransform_second2first = cv::Mat::eye(4, 4, _firstMapPoint3DInCamera.type());
    cv::Mat sRinv = (1.0f/m_flt_currentScale_first2second) * m_cvMat_currentRotation_first2second.t();
    sRinv.copyTo(m_cvMat_currentTransform_second2first.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv * m_cvMat_currentTranslation_first2second;
    tinv.copyTo(m_cvMat_currentTransform_second2first.rowRange(0,3).col(3));
  }
  void Sim3Solver::checkInliers(){
    std::vector<cv::Mat> vFirstPointsPosInSecondImage, vSecondPointsPosInFirstImage;
    project(m_v_secondMapPointsPosInCamera,vSecondPointsPosInFirstImage,m_cvMat_currentTransform_first2second,m_cvMat_firstIntParMat);
    project(m_v_firstMapPointsPosInCamera,vFirstPointsPosInSecondImage,m_cvMat_currentTransform_second2first,m_cvMat_secondIntParMat);
    m_int_inliersNum = 0;
    for(int i=0;i<m_v_firstPointInFirstImage.size();i++){
      cv::Mat dist1 = m_v_firstPointInFirstImage[i] - vSecondPointsPosInFirstImage[i];
      cv::Mat dist2 = vFirstPointsPosInSecondImage[i] - m_v_secondPointInSecondImage[i];
      const float err1 = dist1.dot(dist1);
      const float err2 = dist2.dot(dist2);
      if(err1<m_v_firstMaxError[i] && err2<m_v_secondMaxError[i]){
        m_v_isInliers[i] = true;
        m_int_inliersNum++;
      }else{
        m_v_isInliers[i] = false;
      }
    }
  }
  cv::Mat Sim3Solver::getEstimatedRotation(){
    return m_cvMat_bestRotation_first2second.clone();
  }
  cv::Mat Sim3Solver::getEstimatedTranslation(){
    return m_cvMat_bestTranslation_first2second.clone();
  }
  float Sim3Solver::getEstimatedScale(){
    return m_flt_bestScale;
  }
  void Sim3Solver::project(const std::vector<cv::Mat> &_vMapPointsPosInCamera, std::vector<cv::Mat> &_vPointsPosInImage, cv::Mat _transformation_target2origin, cv::Mat _interParam){
    cv::Mat rotation_target2origin = _transformation_target2origin.rowRange(0,3).colRange(0,3);
    cv::Mat translation_target2origin = _transformation_target2origin.rowRange(0,3).col(3);
    _vPointsPosInImage.clear();
    _vPointsPosInImage.reserve(_vMapPointsPosInCamera.size());
    for(int i=0;i<_vMapPointsPosInCamera.size();i++){
      cv::Mat P3Dc = rotation_target2origin*_vMapPointsPosInCamera[i]+translation_target2origin;
      _vPointsPosInImage.push_back((cv::Mat_<float>(2,1) << _interParam.at<float>(0,0)*P3Dc.at<float>(0)/(P3Dc.at<float>(2))+_interParam.at<float>(0,2), \
                                                            _interParam.at<float>(1,1)*P3Dc.at<float>(1)/(P3Dc.at<float>(2))+_interParam.at<float>(1,2)));
    }
  }
  void Sim3Solver::transferCamera2Image(const std::vector<cv::Mat> &_vMapPointsPosInCamera, std::vector<cv::Mat> &_vPointsPosInImage, cv::Mat _interParam){
    _vPointsPosInImage.clear();
    _vPointsPosInImage.reserve(_vMapPointsPosInCamera.size());
    for(int i=0;i<_vMapPointsPosInCamera.size();i++){
      _vPointsPosInImage.push_back((cv::Mat_<float>(2,1) << _interParam.at<float>(0,0)*_vMapPointsPosInCamera[i].at<float>(0)/(_vMapPointsPosInCamera[i].at<float>(2))+_interParam.at<float>(0,2), \
                                                            _interParam.at<float>(1,1)*_vMapPointsPosInCamera[i].at<float>(1)/(_vMapPointsPosInCamera[i].at<float>(2))+_interParam.at<float>(1,2)));
    }
  }
}//namespace YDORBSLAM
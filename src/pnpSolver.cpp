#include "pnpSolver.hpp"
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <cmath>

namespace YDORBSLAM{
  PnPsolver::PnPsolver(const Frame &_frame, const std::vector<std::shared_ptr<MapPoint>> &_v_matchedMapPoints){
    m_v_matchedMapPoints = _v_matchedMapPoints;
    m_v_point2D.reserve(_frame.m_v_sptrMapPoints.size());
    m_v_scaleFactorSquares.reserve(_frame.m_v_sptrMapPoints.size());
    m_v_point3D.reserve(_frame.m_v_sptrMapPoints.size());
    m_v_keyPointsIndices.reserve(_frame.m_v_sptrMapPoints.size());
    m_v_allIndices.reserve(_frame.m_v_sptrMapPoints.size());
    int idx=0;
    for(int i=0; i<m_v_matchedMapPoints.size();i++){
      if(m_v_matchedMapPoints[i] && !m_v_matchedMapPoints[i]->isBad()){
        m_v_point2D.push_back(_frame.m_v_keyPoints[i].pt);
        m_v_scaleFactorSquares.push_back(_frame.m_v_scaleFactorSquares[_frame.m_v_keyPoints[i].octave]);
        cv::Mat pos = m_v_matchedMapPoints[i]->getPosInWorld();
        m_v_point3D.push_back(cv::Point3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2)));
        m_v_keyPointsIndices.push_back(i);
        m_v_allIndices.push_back(idx);
        idx++;
      }
    }
    //set camera calibration parameters
    m_dbl_fu = _frame.m_flt_fx;
    m_dbl_fv = _frame.m_flt_fy;
    m_dbl_uc = _frame.m_flt_cx;
    m_dbl_vc = _frame.m_flt_cy;
    setRansacParameters();
  }
  void PnPsolver::setRansacParameters(const double &_dbl_probability = 0.99, const int &_int_minInliersNum = 8, const int &_int_maxIterNum = 300, const int &_int_minSet = 4, const float &_flt_epsilon = 0.4, const float &_flt_thd = 5.991){
    m_dbl_ransacProb = _dbl_probability;
    m_int_minRansacInliersNum = _int_minInliersNum;
    m_int_maxRansacIterNum = _int_maxIterNum;
    m_int_minRansacSet = _int_minSet;
    m_flt_ransacEpsilon = _flt_epsilon;
    m_int_correspondencesNum = m_v_point2D.size();
    m_v_isInliers.resize(m_int_correspondencesNum);
    //adjust parameters according to number of correspondences
    int minInlierNum = m_int_correspondencesNum * m_flt_ransacEpsilon;
    minInlierNum = minInlierNum < m_int_minRansacInliersNum?m_int_minRansacInliersNum:minInlierNum;
    minInlierNum = minInlierNum < m_int_minRansacSet?m_int_minRansacSet:minInlierNum;
    m_int_minRansacInliersNum = minInlierNum;
    m_flt_ransacEpsilon = m_flt_ransacEpsilon<(float)m_int_minRansacInliersNum/m_int_correspondencesNum?(float)m_int_minRansacInliersNum/m_int_correspondencesNum:m_flt_ransacEpsilon;
    //set ransac iterations according to probability, epsilon, and max iterations
    int iterNum;
    iterNum = m_int_minRansacInliersNum==m_int_correspondencesNum?1:ceil(log(1-m_dbl_ransacProb)/log(1-pow(m_flt_ransacEpsilon,3)));
    m_int_maxRansacIterNum = std::max(1,std::min(iterNum,m_int_maxRansacIterNum));
    m_v_maxErrors.reserve(m_v_scaleFactorSquares.size());
    for(const float &scaleFactorSquare:m_v_scaleFactorSquares){
      m_v_maxErrors.push_back(scaleFactorSquare*_flt_thd);
    }
  }
  cv::Mat PnPsolver::find(std::vector<bool> &_v_isInliers, int &_int_inliersNum){
    bool bFlag;
    return iterate(m_int_maxRansacIterNum, bFlag, _v_isInliers, _int_inliersNum);
  }
  cv::Mat PnPsolver::iterate(const int &_int_iterNum, bool &_b_isNoMore, std::vector<bool> &_v_isInliers, int &_int_inliersNum){
    _b_isNoMore = false;
    _v_isInliers.clear();
    _int_inliersNum = 0;
    set_maximum_number_of_correspondences(m_int_minRansacSet);
    if(m_int_correspondencesNum<m_int_minRansacInliersNum){
      _b_isNoMore = true;
      return cv::Mat();
    }
    std::vector<int> vAvailableIndices;
    int currentIterNum = 0;
    while(m_int_currentRansacIterNum<m_int_maxRansacIterNum || currentIterNum<_int_iterNum){
      currentIterNum++;
      m_int_currentRansacIterNum++;
      reset_correspondences();
      vAvailableIndices = m_v_allIndices;
      //get min set of points
      for(int i=0;i<m_int_minRansacSet;i++){
        int randInt = rand() % vAvailableIndices.size();
        int idx = vAvailableIndices[randInt];
        add_correspondence(m_v_point3D[idx].x,m_v_point3D[idx].y,m_v_point3D[idx].z,m_v_point2D[idx].x,m_v_point2D[idx].y);
        vAvailableIndices[randInt] = vAvailableIndices.back();
        vAvailableIndices.pop_back();
      }
      //compute camera pose
      compute_pose(m_v_currentRotation,m_v_currentTranslation);
      //check inliners
      checkInliers();
      //stop here
    }
  }
}//namespace YDORBSLAM
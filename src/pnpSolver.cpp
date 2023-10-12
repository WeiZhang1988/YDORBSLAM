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
    m_flt_fu = _frame.m_flt_fx;
    m_flt_fv = _frame.m_flt_fy;
    m_flt_uc = _frame.m_flt_cx;
    m_flt_vc = _frame.m_flt_cy;
    setRansacParameters();
  }
  void PnPsolver::setRansacParameters(const float &_flt_probability = 0.99, const int &_int_minInliersNum = 8, const int &_int_maxIterNum = 300, const int &_int_minSet = 4, const float &_flt_epsilon = 0.4, const float &_flt_thd = 5.991){
    m_flt_ransacProb = _flt_probability;
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
    iterNum = m_int_minRansacInliersNum==m_int_correspondencesNum?1:ceil(log(1-m_flt_ransacProb)/log(1-pow(m_flt_ransacEpsilon,3)));
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
      compute_pose(m_cvMat_currentRotation,m_cvMat_currentTranslation);
      //check inliners
      checkInliers();
      if(m_int_inliersNum>=m_int_minRansacInliersNum){
        //if it is the best solution so far, save it
        if(m_int_inliersNum>m_int_bestInliersNum){
          m_v_isBestInliers = m_v_isInliers;
          m_int_bestInliersNum = m_int_inliersNum;
          m_bestT_c2w = cv::Mat::eye(4,4,CV_32F);
          m_cvMat_currentRotation.copyTo(m_bestT_c2w.rowRange(0,3).colRange(0,3));
          m_cvMat_currentTranslation.copyTo(m_bestT_c2w.rowRange(0,3).col(3));
        }
        if(refine()){
          _int_inliersNum = m_int_refinedInliersNum;
          _v_isInliers = std::vector<bool>(m_v_matchedMapPoints.size(),false);
          for(int i=0;i<m_int_correspondencesNum;i++){
            if(m_v_isRefinedInliers[i]){
              _v_isInliers[m_v_keyPointsIndices[i]] = true;
            }
          }
          return m_cvMat_refinedT_c2w.clone();
        }
      }
    }
    if(m_int_currentRansacIterNum>=m_int_maxRansacIterNum){
      _b_isNoMore = true;
      if(m_int_bestInliersNum>=m_int_minRansacInliersNum){
        _v_isInliers = std::vector<bool>(m_v_matchedMapPoints.size(),false);
        for(int i=0;i<m_int_correspondencesNum;i++){
          if(m_v_isBestInliers[i]){
            _v_isInliers[m_v_keyPointsIndices[i]] = true;
          }
        }
        return m_cvMat_bestT_c2w.clone();
      }
    }
    return cv::Mat();
  }
  void PnPsolver::checkInliers(){
    m_int_inliersNum = 0;
    for(int i=0;i<m_int_correspondencesNum;i++){
      cv::Point3f point3D = m_v_point3D[i];
      cv::Point2f point2D = m_v_point2D[i];
      float xc = m_cvMat_currentRotation.at<float>(0,0)*point3D.x + m_cvMat_currentRotation.at<float>(0,1)*point3D.y + m_cvMat_currentRotation.at<float>(0,2)*point3D.z + m_cvMat_currentTranslation.at<float>(0);
      float yc = m_cvMat_currentRotation.at<float>(1,0)*point3D.x + m_cvMat_currentRotation.at<float>(1,1)*point3D.y + m_cvMat_currentRotation.at<float>(1,2)*point3D.z + m_cvMat_currentTranslation.at<float>(1);
      float ue = m_flt_uc + m_flt_fu * xc / (m_cvMat_currentRotation.at<float>(2,0)*point3D.x + m_cvMat_currentRotation.at<float>(2,1)*point3D.y + m_cvMat_currentRotation.at<float>(2,2)*point3D.z + m_cvMat_currentTranslation.at<float>(2));
      float ve = m_flt_vc + m_flt_fv * yc / (m_cvMat_currentRotation.at<float>(2,0)*point3D.x + m_cvMat_currentRotation.at<float>(2,1)*point3D.y + m_cvMat_currentRotation.at<float>(2,2)*point3D.z + m_cvMat_currentTranslation.at<float>(2));
      float distX = point2D.x - ue;
      float distY = point2D.y - ve;
      float errorSquare = distX * distX + distY * distY;
      if(errorSquare<m_v_maxErrors[i]){
        m_v_isInliers[i] = true;
        m_int_inliersNum++;
      }else{
        m_v_isInliers[i] = false;
      }
    }
  }
  bool PnPsolver::refine(){
    std::vector<int> vIndices;
    vIndices.reserve(m_v_isBestInliers.size());
    for(int i=0;i<m_v_isBestInliers.size();i++){
      if(m_v_isBestInliers[i]){
        vIndices.push_back(i);
      }
    }
    set_maximum_number_of_correspondences(vIndices.size());
    reset_correspondences();
    for(const int &idx : vIndices){
      add_correspondence(m_v_point3D[idx].x,m_v_point3D[idx].y,m_v_point3D[idx].z,m_v_point2D[idx].x,m_v_point2D[idx].y); 
    }
    compute_pose(m_cvMat_currentRotation,m_cvMat_currentTranslation);
    checkInliers();
    m_int_refinedInliersNum = m_int_inliersNum;
    m_v_isRefinedInliers = m_v_isInliers;
    if(m_int_inliersNum>m_int_minRansacInliersNum){
      cv::Mat R_c2w, T_c2w;
      m_cvMat_currentRotation.convertTo(R_c2w,CV_32F);
      m_cvMat_currentTranslation.convertTo(T_c2w,CV_32F);
      m_cvMat_refinedT_c2w = cv::Mat::eye(4,4,CV_32F);
      R_c2w.copyTo(m_cvMat_refinedT_c2w.rowRange(0,3).colRange(0,3));
      T_c2w.copyTo(m_cvMat_refinedT_c2w.rowRange(0,3).col(3));
      return true;
    }else{
      return false;
    }
  }
  void PnPsolver::set_maximum_number_of_correspondences(const int _int_num){
    if(m_int_maxCorrespondencesNum < _int_num){
      m_int_maxCorrespondencesNum = _int_num;
      m_v_pws     = std::vector<float>(3*m_int_maxCorrespondencesNum);
      m_v_us      = std::vector<float>(2*m_int_maxCorrespondencesNum);
      m_v_alphas  = std::vector<float>(4*m_int_maxCorrespondencesNum);
      m_v_pcs     = std::vector<float>(3*m_int_maxCorrespondencesNum);
    }
  }
  void PnPsolver::reset_correspondences(void){
    m_int_correspondencesNum = 0;
  }
  void PnPsolver::add_correspondence(const float &_flt_3DX, const float &_flt_3DY, const float &_flt_3DZ, const float &_flt_2DX, const float &_flt_2DY){
    m_v_pws[3*m_int_correspondencesNum + 0] = _flt_3DX;
    m_v_pws[3*m_int_correspondencesNum + 1] = _flt_3DY;
    m_v_pws[3*m_int_correspondencesNum + 2] = _flt_3DZ;
    m_v_us[2*m_int_correspondencesNum + 0] = _flt_2DX;
    m_v_us[2*m_int_correspondencesNum + 1] = _flt_2DY;
    m_int_correspondencesNum++;
  }
  void PnPsolver::compute_pose(cv::Mat &_rotation, cv::Mat &_translation){
    choose_control_points();
    compute_barycentric_coordinates();
    cv::Mat M = cv::Mat::zeros(2*m_int_correspondencesNum,12,CV_32F);
    for(int i=0;i<m_int_correspondencesNum;i++){
      fill_M(M,2*i,std::vector<float>(m_v_alphas.begin() + 4*i, m_v_alphas.begin() + 4*(i+1)),m_v_us[2*i],m_v_us[2*i+1]);
    }
    float mtm[12 * 12] = {}, d[12] = {}, ut[12 * 12] = {}, vt[12 * 12] = {};
    cv::Mat MtM = cv::Mat(12,12,CV_32F,mtm);
    cv::Mat D   = cv::Mat(12, 1,CV_32F,d);
    cv::Mat Ut  = cv::Mat(12,12,CV_32F,ut);
    cv::Mat Vt  = cv::Mat(12,12,CV_32F,vt);
    cv::mulTransposed(M,MtM,true);
    cv::SVD::compute(MtM, D, Ut.t(), Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    M.release();
    float l_6x10[6 * 10] = {}, rho[6] = {};
    cv::Mat L_6x10 = cv::Mat(6,10,CV_32F,l_6x10);
    cv::Mat Rho    = cv::Mat(6, 1,CV_32F,rho);
    compute_L_6x10(u,l_6x10);
    compute_rho(rho);
    float Betas[4][4] = {}, rep_errors[4] = {};
    float Rs[4][3][3] = {}, ts[4][3] = {};
    find_betas_approx_1(L_6x10, Rho, Betas[1]);
    gauss_newton(L_6x10, Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);
    find_betas_approx_2(L_6x10, Rho, Betas[2]);
    gauss_newton(L_6x10, Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);
    find_betas_approx_3(L_6x10, Rho, Betas[3]);
    gauss_newton(L_6x10, Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;
    cv::Mat(3,1,CV_32F,ts[N]).copyTo(_translation);
    cv::Mat(3,3,CV_32F,Rs[N]).copyTo(_rotation);
  }
  void PnPsolver::fill_M(cv::Mat &_M, const int &_row, const std::vector<float> &_alphas, const float &_u, const float &_v){
    float *M1 = M.ptr<float>(0) + _row * 12;
    float *M2 = M1 + 12;
    for(int i = 0; i < 4; i++) {
      M1[3 * i    ] = _alphas[i] * m_flt_fu;
      M1[3 * i + 1] = 0.0;
      M1[3 * i + 2] = _alphas[i] * (m_flt_uc - _u);
      M2[3 * i    ] = 0.0;
      M2[3 * i + 1] = _alphas[i] * m_flt_fv;
      M2[3 * i + 2] = _alphas[i] * (m_flt_vc - _v);
    }
  }
  void PnPsolver::compute_L_6x10(const float *_u, float *_l_6x10){
    const float * v[4];
    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 *  9;
    v[3] = ut + 12 *  8;
    float dv[4][6][3] = {};
    for(int i = 0; i < 4; i++) {
      int a = 0, b = 1;
      for(int j = 0; j < 6; j++) {
        dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
        dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
        dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

        b++;
        if (b > 3) {
          a++;
          b = a + 1;
        }
      }
    }
    for(int i = 0; i < 6; i++) {
      float * row = l_6x10 + 10 * i;
      row[0] =        dot(dv[0][i], dv[0][i]);
      row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
      row[2] =        dot(dv[1][i], dv[1][i]);
      row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
      row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
      row[5] =        dot(dv[2][i], dv[2][i]);
      row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
      row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
      row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
      row[9] =        dot(dv[3][i], dv[3][i]);
    }
  }
  void PnPsolver::compute_rho(float *_rho){
    rho[0] = dist2(m_vv_cws[0], m_vv_cws[1]);
    rho[1] = dist2(m_vv_cws[0], m_vv_cws[2]);
    rho[2] = dist2(m_vv_cws[0], m_vv_cws[3]);
    rho[3] = dist2(m_vv_cws[1], m_vv_cws[2]);
    rho[4] = dist2(m_vv_cws[1], m_vv_cws[3]);
    rho[5] = dist2(m_vv_cws[2], m_vv_cws[3]);
  }
  float PnPsolver::dot(const float *_v1, const float *_v2){
    return _v1[0] * _v2[0] + _v1[1] * _v2[1] + _v1[2] * _v2[2];
  }
  float PnPsolver::dist2(const float *_p1, const float *_p2){
    return
    (_p1[0] - _p2[0]) * (_p1[0] - _p2[0]) +
    (_p1[1] - _p2[1]) * (_p1[1] - _p2[1]) +
    (_p1[2] - _p2[2]) * (_p1[2] - _p2[2]);
  }
  float PnPsolver::compute_R_and_t(const float *_ut, const float *_betas, float _rotation[3][3], float _translation[3]){
    compute_css(_betas,_ut);
    compute_pcs();
    solve_for_sign();
    estimate_R_and_t(_R, _t);
    return reprojection_error(_R, _t);
  }
  void PnPsolver::compute_ccs(const float *_betas, const float *_ut){
    for(int i = 0; i < 4; i++)
      m_vv_ccs[i][0] = m_vv_ccs[i][1] = m_vv_ccs[i][2] = 0.0f;
    for(int i = 0; i < 4; i++) {
      const float *v = _ut + 12 * (11 - i);
      for(int j = 0; j < 4; j++){
        for(int k = 0; k < 3; k++){
  	      m_vv_ccs[j][k] += betas[i] * v[3 * j + k];
        }
      }
    }
  }
  void PnPsolver::compute_pcs(void){
    for(int i = 0; i < m_int_correspondencesNum; i++) {
      float * a = &m_v_alphas[0] + 4 * i;
      float * pc = &m_v_pcs[0] + 3 * i;

      for(int j = 0; j < 3; j++)
        pc[j] = a[0] * m_vv_ccs[0][j] + a[1] * m_vv_ccs[1][j] + a[2] * m_vv_ccs[2][j] + a[3] * m_vv_ccs[3][j];
    }
  }
  void PnPsolver::solve_for_sign(void){
    if (m_v_pcs[2] < 0.0) {
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 3; j++){
          m_vv_ccs[i][j] = -m_vv_ccs[i][j];
        }
      }

      for(int i = 0; i < m_int_correspondencesNum; i++) {
        m_v_pcs[3 * i    ] = -m_v_pcs[3 * i];
        m_v_pcs[3 * i + 1] = -m_v_pcs[3 * i + 1];
        m_v_pcs[3 * i + 2] = -m_v_pcs[3 * i + 2];
      }
    }
  }
  void PnPsolver::estimate_R_and_t(float _rotation[3][3], float _translation[3]){
    float pc0[3] = {}, pw0[3] = {};
    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;
    for(int i = 0; i < number_of_correspondences; i++) {
      const float * pc = &m_v_pcs[3 * i];
      const float * pw = &m_v_pws[3 * i];
      for(int j = 0; j < 3; j++) {
        pc0[j] += pc[j];
        pw0[j] += pw[j];
      }
    }
    for(int j = 0; j < 3; j++) {
      pc0[j] /= m_int_correspondencesNum;
      pw0[j] /= m_int_correspondencesNum;
    }
    float abt[3 * 3] = {}, abt_d[3] = {}, abt_u[3 * 3] = {}, abt_v[3 * 3] = {};
    cv::Mat ABt   = cv::Mat(3,3,CV_32F,abt);
    cv::Mat ABt_D = cv::Mat(3,1,CV_32F,abt_d);
    cv::Mat ABt_U = cv::Mat(3,3,CV_32F,abt_u);
    cv::Mat ABt_V = cv::Mat(3,3,CV_32F,abt_v);
    ABt.setTo(cv::Scalar::all(0.0f));
    for(int i = 0; i < m_int_correspondencesNum; i++) {
      float * pc = &m_v_pcs[3 * i];
      float * pw = &m_v_pws[3 * i];

      for(int j = 0; j < 3; j++) {
        abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
        abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
        abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
      }
    }
    cv::SVD::compute(ABt, ABt_D, ABt_U, ABt_V.t(), cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        _rotation[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);
      }
    }
    const float det =
    _rotation[0][0] * _rotation[1][1] * _rotation[2][2] + _rotation[0][1] * _rotation[1][2] * _rotation[2][0] + _rotation[0][2] * _rotation[1][0] * _rotation[2][1] -
    _rotation[0][2] * _rotation[1][1] * _rotation[2][0] - _rotation[0][1] * _rotation[1][0] * _rotation[2][2] - _rotation[0][0] * _rotation[1][2] * _rotation[2][1];
    if(det<0){
      _rotation[2][0] = -_rotation[2][0];
      _rotation[2][1] = -_rotation[2][1];
      _rotation[2][2] = -_rotation[2][2];
    }
    _translation[0] = pc0[0] - dot(_rotation[0], pw0);
    _translation[1] = pc0[1] - dot(_rotation[1], pw0);
    _translation[2] = pc0[2] - dot(_rotation[2], pw0);
  }
  float PnPsolver::reprojection_error(const float _rotation[3][3], const float &_translation[3]){
    float squaredSum = 0.0;
    for(int i=0;i<m_int_correspondencesNum;i++){
      float * pw = &m_v_pws[3 * i];
      float Xc = dot(_rotation[0], pw) + _translation[0];
      float Yc = dot(_rotation[1], pw) + _translation[1];
      float ue = uc + fu * Xc  / (dot(_rotation[2], pw) + _translation[2]);
      float ve = vc + fv * Yc  / (dot(_rotation[2], pw) + _translation[2]);
      float u = us[2 * i], v = us[2 * i + 1];
      squaredSum += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
    }
    return squaredSum / m_int_correspondencesNum;
  }
  void PnPsolver::choose_control_points(void){
    // Take C0 as the reference points centroid:
    m_vv_cws[0][0] = m_vv_cws[0][1] = m_vv_cws[0][2] = 0;
    for(int i=0;i<m_int_correspondencesNum;i++){
      for(int j=0;j<3;j++){
        m_vv_cws[0][j] += m_v_pws[3 * i + j];
      }
    }
    for(int j=0;j<3;j++){
      m_vv_cws[0][j] /= m_int_correspondencesNum;
    }
    // Take C1, C2, and C3 from PCA on the reference points:
    cv::Mat PW0 = cv::Mat(m_int_correspondencesNum,3,CV_32F);
    float pw0tpw0[3 * 3] = {}, dc[3] = {}, uct[3 * 3] = {};
    cv::Mat PW0tPW0 = cv::Mat(3, 3, CV_32F, pw0tpw0);
    cv::Mat DC      = cv::Mat(3, 1, CV_32F, dc);
    cv::Mat UCt     = cv::Mat(3, 3, CV_32F, uct);
    cv::Mat VCt     = cv::Mat(3, 3, CV_32F);
    for(int i=0;i<m_int_correspondencesNum;i++){
      for(int j=0;j<3;j++){
        PW0.at<float>(3*i + j) = m_v_pws[3*i + j] - m_vv_cws[0][j];
      }
    }
    cv::mulTransposed(PW0,PW0tPW0,true);
    cv::SVD::compute(PW0tPW0, DC, UCt.t(), VCt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    PW0.release();
    for(int i=1;i<4;i++){
      float k = sqrt(dc[i-1]/m_int_correspondencesNum);
      for(int j=0;j<3;j++){
        m_vv_cws[i][j] = m_vv_cws[0][j] + k*uct[3*(i-1)+j];
      }
    }
  }
  void PnPsolver::compute_barycentric_coordinates(void){
    float cc[3 * 3] = {}, cc_inv[3 * 3] = {};
    cv::Mat CC     = cv::Mat(3, 3, CV_32F, cc);
    cv::Mat CC_inv = cv::Mat(3, 3, CV_32F, cc_inv);

    for(int i=0;i<3;i++){
      for(int j=1;j<4;j++){
        cc[3 * i + j - 1] = m_vv_cws[j][i] - m_vv_cws[0][i];
      }
    }
    cv::invert(CC,CC_inv,CV_SVD);
    float *ci = cc_inv;
    for(int i=0;i<m_int_correspondencesNum;i++) {
      float *pi = &m_v_pws[0] + 3 * i;
      float *a  = &m_v_alphas[0] + 4 * i;

      for(int j=0;j<3;j++)
      {
        a[1 + j] =
          ci[3 * j    ] * (pi[0] - cws[0][0]) +
          ci[3 * j + 1] * (pi[1] - cws[0][1]) +
          ci[3 * j + 2] * (pi[2] - cws[0][2]);
      }
      a[0] = 1.0f - a[1] - a[2] - a[3];
    }
  }
  void PnPsolver::find_betas_approx_1(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas){
    float l_6x4[6 * 4] = {}, b4[4] = {};
    cv::Mat L_6x4 = cv::Mat(6, 4, CV_32F, l_6x4);
    cv::Mat B4    = cv::Mat(4, 1, CV_32F, b4);
    for(int i=0;i<6;i++){
      L_6x4.at<float>(i,0) = _L_6x10.at<float>(i,0);
      L_6x4.at<float>(i,1) = _L_6x10.at<float>(i,1);
      L_6x4.at<float>(i,2) = _L_6x10.at<float>(i,3);
      L_6x4.at<float>(i,3) = _L_6x10.at<float>(i,6);
    }
    cv::solve(L_6x4, _Rho, B4, CV_SVD);
    if(b4[0]<0){
      _betas[0] = sqrt(-b4[0]);
      _betas[1] = -b4[1] / _betas[0];
      _betas[2] = -b4[2] / _betas[0];
      _betas[3] = -b4[3] / _betas[0];
    }else{
      _betas[0] = sqrt(b4[0]);
      _betas[1] = b4[1] / _betas[0];
      _betas[2] = b4[2] / _betas[0];
      _betas[3] = b4[3] / _betas[0];
    }
  }
  void PnPsolver::find_betas_approx_2(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas){
    float l_6x3[6 * 3] = {}, b3[3] = {};
    cv::Mat L_6x3 = cv::Mat(6, 3, CV_32F, l_6x3);
    cv::Mat B3    = cv::Mat(3, 1, CV_32F, b3);
    for(int i=0;i<6;i++){
      L_6x3.at<float>(i,0) = _L_6x10.at<float>(i,0);
      L_6x3.at<float>(i,1) = _L_6x10.at<float>(i,1);
      L_6x3.at<float>(i,2) = _L_6x10.at<float>(i,2);
    }
    cv::solve(L_6x3, _Rho, B3, CV_SVD);
    if(b3[0]<0){
      _betas[0] = sqrt(-b3[0]);
      _betas[1] = (b3[2]<0) ? sqrt(-b3[2]) : 0.0;
    }else{
      _betas[0] = sqrt(b4[0]);
      _betas[1] = (b3[2]>0) ? sqrt(b3[2]) : 0.0;
    }
    if (b3[1]<0) _betas[0] = -_betas[0];
    _betas[2] = 0.0;
    _betas[3] = 0.0;
  }
  void PnPsolver::find_betas_approx_3(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas){
    float l_6x5[6 * 5] = {}, b5[5] = {};
    cv::Mat L_6x5 = cv::Mat(6, 5, CV_32F, l_6x5);
    cv::Mat B5    = cv::Mat(5, 1, CV_32F, b5);
    for(int i=0;i<6;i++){
      L_6x5.at<float>(i,0) = _L_6x10.at<float>(i,0);
      L_6x5.at<float>(i,1) = _L_6x10.at<float>(i,1);
      L_6x5.at<float>(i,2) = _L_6x10.at<float>(i,2);
      L_6x5.at<float>(i,3) = _L_6x10.at<float>(i,3);
      L_6x5.at<float>(i,4) = _L_6x10.at<float>(i,4);
    }
    cv::solve(L_6x5, _Rho, B5, CV_SVD);
    if(b5[0]<0){
      _betas[0] = sqrt(-b5[0]);
      _betas[1] = (b5[2]<0) ? sqrt(-b5[2]) : 0.0;
    }else{
      _betas[0] = sqrt(b5[0]);
      _betas[1] = (b5[2]>0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1]<0) _betas[0] = -_betas[0];
    _betas[2] = b5[3] / _betas[0];
    _betas[3] = 0.0;
  }
  void PnPsolver::qr_solve(cv::Mat &_A, cv::Mat &_b, cv::Mat &_X){
    static int max_nr = 0;
    static float *A1, *A2;
    const int nr = _A->rows;
    const int nc = _A->cols;
    if(nc<=0 || nr<=0){
      return;
    }
    if(max_nr!=0 && max_nr<nr){
      delete [] A1;
      delete [] A2;
    }
    if(max_nr<nr){
      max_nr = nr;
      A1 = new float[nr];
      A2 = new float[nr];
    }
    float *pA = (float*)_A.data, *ppAkk = pA;
    for(int k=0;k<nc;k++){
      float *ppAik1 = ppAkk, eta = fabs(*ppAik1);
      for(int i=k+1;i<nr;i++){
        float elt = fabs(*ppAik1);
        if(eta<elt){
          eta = elt;
        }
        ppAik1 += nc;
      }
      if(eta==0){
        A1[k] = A2[k] = 0.0;
        //cerr << "God damnit, A is singular, this shouldn't happen." << endl;
        return;
      }else{
        float *ppAik2 = ppAkk, squaredSum = 0.0; inv_eta = 1.0 / eta;
        for(int i=k;i<nr;i++){
          *ppAik2 *= inv_eta;
          squaredSum += *ppAik2 * *ppAik2;
          ppAik2 += nc;
        }
        float sigma = sqrt(squaredSum);
        if(*ppAkk<0){
          sigma = -sigma;
        }
        *ppAkk += sigma;
        A1[k] = sigma * *ppAkk;
        A2[k] = -eta * sigma;
        for(int j=k+1;j<nc;j++){
          float *ppAik = ppAkk, sum = 0;
          for(int i=k;i<nr;i++){
            sum += *ppAik * ppAik[j-k];
            ppAik += nc;
          }
          float tau = sum / A1[k];
          ppAik = ppAkk;
          for(int i=k;i<nr;i++){
            ppAik[j-k] -= tau * *ppAik;
            ppAik += nc;
          }
        }
      }
      ppAkk += nc + 1;
    }
    // b <- Qt b
    float *ppAjj = pA, *pb = (float*)_b.data;
    for(int j=0;j<nc;j++){
      float *ppAij = ppAjj, tau = 0.0;
      for(int i=j;i<nr;i++){
        tau += *ppAij * pb[i];
        ppAij += nc;
      }
      tau /= A1[j];
      ppAij = ppAjj;
      for(int i=j;i<nr;i++){
        pb[i] -= tau * *ppAij;
        ppAij += nc;
      }
      ppAjj += nc + 1;
    }
    // X = R-1 b
    float *pX = (float)_X.data;
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for(int i=nc-2;i>=0;i--){
      float * ppAij = pA + i * nc + (i + 1), sum = 0;
      for(int j=i+1;j<nc;j++){
        sum += *ppAij * pX[j];
        ppAij++;
      }
      pX[i] = (pb[i] - sum) / A2[i];
    }
  }
  void PnPsolver::gauss_newton(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float _current_betas[4]){
    const int iterations_number = 5;
    double a[6*4] = {}, b[6] = {}, x[4] = {};
    cv::Mat A = cv::Mat(6, 4, CV_32F, a);
    cv::Mat B = cv::Mat(6, 1, CV_32F, b);
    cv::Mat X = cv::Mat(4, 1, CV_32F, x);
    for(int k=0;k<iterations_number;k++){
      compute_A_and_b_gauss_newton((float*)_L_6x10.data,(float*)_Rho.data,_current_betas,A,B);
      qr_solve(A,B,X);
      for(int i=0;i<4;i++){
        _current_betas[i]+=x[i];
      }
    }
  }
  void PnPsolver::compute_A_and_b_gauss_newton(const float *_l_6x10, const float *_rho, float _cb[4], cv::Mat &_A, cv::Mat &_b){
    for(int i=0;i<6;i++){
      const float *rowL = _l_6x10 + i * 10;
      float *rowA = (float*)_A.data + i * 4;
      rowA[0] = 2 * rowL[0] * _cb[0] +     rowL[1] * _cb[1] +     rowL[3] * _cb[2] +     rowL[6] * _cb[3];
      rowA[1] =     rowL[1] * _cb[0] + 2 * rowL[2] * _cb[1] +     rowL[4] * _cb[2] +     rowL[7] * _cb[3];
      rowA[2] =     rowL[3] * _cb[0] +     rowL[4] * _cb[1] + 2 * rowL[5] * _cb[2] +     rowL[8] * _cb[3];
      rowA[3] =     rowL[6] * _cb[0] +     rowL[7] * _cb[1] +     rowL[8] * _cb[2] + 2 * rowL[9] * _cb[3];
      _b.at<float>(i,0) = _rho[i] - (
      rowL[0] * _cb[0] * _cb[0] +
      rowL[1] * _cb[0] * _cb[1] +
      rowL[2] * _cb[1] * _cb[1] +
      rowL[3] * _cb[0] * _cb[2] +
      rowL[4] * _cb[1] * _cb[2] +
      rowL[5] * _cb[2] * _cb[2] +
      rowL[6] * _cb[0] * _cb[3] +
      rowL[7] * _cb[1] * _cb[3] +
      rowL[8] * _cb[2] * _cb[3] +
      rowL[9] * _cb[3] * _cb[3]
      );
    }
  }
}//namespace YDORBSLAM
#include "frame.hpp"
#include "keyFrame.hpp"
#include "converter.hpp"
#include "orbMatcher.hpp"
#include <thread>

namespace YDORBSLAM
{
  long int Frame::m_int_reservedID = 0;
  bool Frame::m_b_isComputeInit = true;
  cv::Mat Frame::m_cvMat_intParMat;
  float Frame::m_flt_cx, Frame::m_flt_cy, Frame::m_flt_fx, Frame::m_flt_fy, Frame::m_flt_invFx, Frame::m_flt_invFy;
  cv::Mat Frame::m_cvMat_imageDistCoef;
  float Frame::m_flt_baseLineTimesFx;
  float Frame::m_flt_baseLine;
  float Frame::m_flt_gridCellHeightInv, Frame::m_flt_gridCellWidthInv;
  float Frame::m_flt_minX, Frame::m_flt_minY, Frame::m_flt_maxX, Frame::m_flt_maxY;
  std::mutex Frame::m_mutex_ID;
  //copy from object reference
  Frame::Frame(const Frame &_frame){
    m_sptr_vocab              = _frame.m_sptr_vocab;
    m_sptr_leftOrbExtractor   = _frame.m_sptr_leftOrbExtractor;
    m_sptr_rightOrbExtractor  = _frame.m_sptr_rightOrbExtractor;
    m_d_timeStamp             = _frame.m_d_timeStamp;
    m_flt_baseLineTimesFx     = _frame.m_flt_baseLineTimesFx;
    m_flt_baseLine            = _frame.m_flt_baseLine;
    m_flt_depthThd            = _frame.m_flt_depthThd;
    m_int_keyPointsNum        = _frame.m_int_keyPointsNum;
    m_v_keyPoints             = _frame.m_v_keyPoints;
    m_v_rightKeyPoints        = _frame.m_v_rightKeyPoints;
    m_v_rightXcords           = _frame.m_v_rightXcords;
    m_v_depth                 = _frame.m_v_depth;
    m_bow_wordVec             = _frame.m_bow_wordVec;
    m_bow_keyPointsVec        = _frame.m_bow_keyPointsVec;
    m_cvMat_descriptors       = _frame.m_cvMat_descriptors.clone();
    m_cvMat_rightDescriptors  = _frame.m_cvMat_rightDescriptors.clone();
    m_v_sptrMapPoints         = _frame.m_v_sptrMapPoints;
    m_v_isOutliers            = _frame.m_v_isOutliers;
    m_vvv_grid                = _frame.m_vvv_grid;
    m_sptr_refKeyFrame        = _frame.m_sptr_refKeyFrame;
    m_int_scaleLevelsNum      = _frame.m_int_scaleLevelsNum;
    m_flt_scaleFactor         = _frame.m_flt_scaleFactor;
    m_flt_logScaleFactor      = _frame.m_flt_logScaleFactor;
    m_v_scaleFactors          = _frame.m_v_scaleFactors;
    m_v_invScaleFactors       = _frame.m_v_invScaleFactors;
    m_v_scaleFactorSquares    = _frame.m_v_scaleFactorSquares;
    m_v_invScaleFactorSquares = _frame.m_v_invScaleFactorSquares;
    {
      std::unique_lock<std::mutex> lock(m_mutex_ID);
      m_int_ID                = _frame.m_int_ID;
    }
    m_cvMat_R_c2w             = _frame.m_cvMat_R_c2w.clone();
    m_cvMat_R_w2c             = _frame.m_cvMat_R_w2c.clone();
    m_cvMat_t_c2w             = _frame.m_cvMat_t_c2w.clone();
    m_cvMat_T_c2w             = _frame.m_cvMat_T_c2w.clone();
    m_cvMat_origin            = _frame.m_cvMat_origin.clone();
  }
  Frame::Frame(const cv::Mat &_leftImage, const cv::Mat &_rightImage, const double &_timeStamp, const cv::Mat &_camIntParMat, const cv::Mat &_imageDistCoef, const float &_baseLineTimesFx, const float &_depthThd, std::shared_ptr<OrbExtractor> _sptrLeftExtractor, std::shared_ptr<OrbExtractor> _sptrRightExtractor, std::shared_ptr<DBoW3::Vocabulary> _sptrVocab):\
  m_sptr_vocab(_sptrVocab),m_sptr_leftOrbExtractor(_sptrLeftExtractor),m_sptr_rightOrbExtractor(_sptrRightExtractor),\
  m_d_timeStamp(_timeStamp),m_flt_depthThd(_depthThd){
    {
      std::unique_lock<std::mutex> lock(m_mutex_ID);
      m_int_ID = m_int_reservedID++;
    }
    m_cvMat_intParMat = _camIntParMat;
    m_cvMat_imageDistCoef = _imageDistCoef;
    m_flt_baseLineTimesFx = _baseLineTimesFx;
    m_int_scaleLevelsNum = m_sptr_leftOrbExtractor->getLevelsNum();
    m_flt_scaleFactor = m_sptr_leftOrbExtractor->getScaleFactor();
    m_flt_logScaleFactor = log(m_flt_scaleFactor);
    m_v_scaleFactors = m_sptr_leftOrbExtractor->getScaleFactors();
    m_v_invScaleFactors = m_sptr_leftOrbExtractor->getInvScaleFactors();
    m_v_scaleFactorSquares = m_sptr_leftOrbExtractor->getScaleFactorSquares();
    m_v_invScaleFactorSquares = m_sptr_leftOrbExtractor->getInvScaleFactorSquares();
    std::thread thread_extractLeftOrb(&Frame::extractOrb,this,_leftImage,false);
    std::thread thread_extractRightOrb(&Frame::extractOrb,this,_rightImage,true);
    thread_extractLeftOrb.join();
    thread_extractRightOrb.join();
    m_int_keyPointsNum = m_v_keyPoints.size();
    if(m_v_keyPoints.empty()){
      return;
    }
    undistortKeyPoints();
    computeStereoMatches();
    m_v_sptrMapPoints = std::vector<std::shared_ptr<MapPoint>>(m_int_keyPointsNum,static_cast<std::shared_ptr<MapPoint>>(nullptr));
    m_v_isOutliers = std::vector<bool>(m_int_keyPointsNum,false);
    if(m_b_isComputeInit){
      computeImageBounds(_leftImage);
      m_flt_gridCellWidthInv=static_cast<float>(m_int_gridColsNum)/(m_flt_maxX - m_flt_minX);
      m_flt_gridCellHeightInv=static_cast<float>(m_int_gridRowsNum)/(m_flt_maxY - m_flt_minY);
      m_flt_fx = m_cvMat_intParMat.at<float>(0,0);
      m_flt_fy = m_cvMat_intParMat.at<float>(1,1);
      m_flt_cx = m_cvMat_intParMat.at<float>(0,2);
      m_flt_cy = m_cvMat_intParMat.at<float>(1,2);
      m_flt_invFx = 1.0/m_flt_fx;
      m_flt_invFy = 1.0/m_flt_fy;
      m_b_isComputeInit = false;
    }
    m_flt_baseLine = m_flt_baseLineTimesFx/m_flt_fx;
    assignKeyPointsToGrid();
  }
  Frame::Frame(const cv::Mat &_grayImage, const cv::Mat &_depthImage, const double &_timeStamp, const cv::Mat &_camIntParMat, const cv::Mat &_imageDistCoef, const float &_baseLineTimesFx, const float &_depthThd, std::shared_ptr<OrbExtractor> _sptrExtractor, std::shared_ptr<DBoW3::Vocabulary> _sptrVocab):\
  m_sptr_vocab(_sptrVocab),m_sptr_leftOrbExtractor(_sptrExtractor),m_sptr_rightOrbExtractor(static_cast<std::shared_ptr<OrbExtractor>>(nullptr)),\
  m_d_timeStamp(_timeStamp),m_flt_depthThd(_depthThd){
    {
      std::unique_lock<std::mutex> lock(m_mutex_ID);
      m_int_ID = m_int_reservedID++;
    }
    m_cvMat_intParMat = _camIntParMat;
    m_cvMat_imageDistCoef = _imageDistCoef;
    m_flt_baseLineTimesFx = _baseLineTimesFx;
    m_int_scaleLevelsNum = m_sptr_leftOrbExtractor->getLevelsNum();
    m_flt_scaleFactor = m_sptr_leftOrbExtractor->getScaleFactor();
    m_flt_logScaleFactor = log(m_flt_scaleFactor);
    m_v_scaleFactors = m_sptr_leftOrbExtractor->getScaleFactors();
    m_v_invScaleFactors = m_sptr_leftOrbExtractor->getInvScaleFactors();
    m_v_scaleFactorSquares = m_sptr_leftOrbExtractor->getScaleFactorSquares();
    m_v_invScaleFactorSquares = m_sptr_leftOrbExtractor->getInvScaleFactorSquares();
    extractOrb(_grayImage);
    m_int_keyPointsNum = m_v_keyPoints.size();
    if(m_v_keyPoints.empty()){
      return;
    }
    undistortKeyPoints();
    computeStereoFromRGBD(_depthImage);
    m_v_sptrMapPoints = std::vector<std::shared_ptr<MapPoint>>(m_int_keyPointsNum,static_cast<std::shared_ptr<MapPoint>>(nullptr));
    m_v_isOutliers = std::vector<bool>(m_int_keyPointsNum,false);
    if(m_b_isComputeInit){
      computeImageBounds(_grayImage);
      m_flt_gridCellWidthInv=static_cast<float>(m_int_gridColsNum)/(m_flt_maxX - m_flt_minX);
      m_flt_gridCellHeightInv=static_cast<float>(m_int_gridRowsNum)/(m_flt_maxY - m_flt_minY);
      m_flt_fx = m_cvMat_intParMat.at<float>(0,0);
      m_flt_fy = m_cvMat_intParMat.at<float>(1,1);
      m_flt_cx = m_cvMat_intParMat.at<float>(0,2);
      m_flt_cy = m_cvMat_intParMat.at<float>(1,2);
      m_flt_invFx = 1.0/m_flt_fx;
      m_flt_invFy = 1.0/m_flt_fy;
      m_b_isComputeInit = false;
    }
    m_flt_baseLine = m_flt_baseLineTimesFx/m_flt_fx;
    assignKeyPointsToGrid();
  }
  void Frame::extractOrb(const cv::Mat &_image, const bool &_isRight){
    if(_isRight){
      m_sptr_rightOrbExtractor->extractAndCompute(_image,m_v_rightKeyPoints,m_cvMat_rightDescriptors);
    }else{
      m_sptr_leftOrbExtractor->extractAndCompute(_image,m_v_keyPoints,m_cvMat_descriptors);
    }
  }
  void Frame::undistortKeyPoints(){
    if(m_cvMat_imageDistCoef.at<float>(0)!=0.0){
      //fill matrix with points
      cv::Mat mat(m_int_keyPointsNum,2,CV_32F);
      for(int i=0;i<m_int_keyPointsNum;i++){
        mat.at<float>(i,0)=m_v_keyPoints[i].pt.x;
        mat.at<float>(i,1)=m_v_keyPoints[i].pt.y;
      }
      //undistort points
      mat = mat.reshape(2);
      cv::undistortPoints(mat,mat,m_cvMat_intParMat,m_cvMat_imageDistCoef,cv::Mat(),m_cvMat_intParMat);
      mat = mat.reshape(1);
      //fill undistorted key point vector
      for(int i=0;i<m_int_keyPointsNum;i++){
        m_v_keyPoints[i].pt.x = mat.at<float>(i,0);
        m_v_keyPoints[i].pt.y = mat.at<float>(i,1);
      }
    }
  }
  void Frame::computeStereoFromRGBD(const cv::Mat &_depthImage){
    m_v_rightXcords = std::vector<float>(m_int_keyPointsNum,-1);
    m_v_depth = std::vector<float>(m_int_keyPointsNum,-1);
    for(int i=0;i<m_int_keyPointsNum;i++){
      const float depth = _depthImage.at<float>(m_v_keyPoints[i].pt.y,m_v_keyPoints[i].pt.x);
      if(depth>0){
        m_v_depth[i] = depth;
        m_v_rightXcords[i] = m_v_keyPoints[i].pt.x - m_flt_baseLineTimesFx / depth;
      }
    }
  }
  void Frame::computeImageBounds(const cv::Mat &_imageLeft){
    if(m_cvMat_imageDistCoef.at<float>(0)==0.0){
      m_flt_minX = 0.0;
      m_flt_maxX = _imageLeft.cols;
      m_flt_minY = 0.0;
      m_flt_maxY = _imageLeft.rows;
    }else{
      cv::Mat mat(4,2,CV_32F);
      mat.at<float>(0,0) = 0.0;
      mat.at<float>(0,1) = 0.0;
      mat.at<float>(1,0) = _imageLeft.cols;
      mat.at<float>(1,1) = 0.0;
      mat.at<float>(2,0) = 0.0;
      mat.at<float>(2,1) = _imageLeft.rows;
      mat.at<float>(3,0) = _imageLeft.cols;
      mat.at<float>(3,1) = _imageLeft.rows;
      //undistort corners
      mat = mat.reshape(2);
      cv::undistortPoints(mat,mat,m_cvMat_intParMat,m_cvMat_imageDistCoef,cv::Mat(),m_cvMat_intParMat);
      mat = mat.reshape(1);
      m_flt_minX = std::min(mat.at<float>(0,0),mat.at<float>(2,0));
      m_flt_maxX = std::max(mat.at<float>(1,0),mat.at<float>(3,0));
      m_flt_minY = std::min(mat.at<float>(0,1),mat.at<float>(1,1));
      m_flt_maxY = std::max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
  }
  void Frame::assignKeyPointsToGrid(){
    int reservedCapacity = (int)((float)m_int_keyPointsNum/(float)m_int_gridColsNum/(float)m_int_gridRowsNum);
    for(auto &vv : m_vvv_grid){
      for(auto &v : vv){
        v.reserve(reservedCapacity);
      }
    }
    int i_for = 0;
    for(const cv::KeyPoint &keyPoint : m_v_keyPoints){
      int gridLocX, gridLocY;
      if(computeLocationInGrid(keyPoint,gridLocX,gridLocY)){
        m_vvv_grid[gridLocX][gridLocY].push_back(i_for);
      }
      i_for++;
    }
  }
  void Frame::computeBoW(){
    if(m_bow_wordVec.empty()){
      std::vector<cv::Mat> vCurrentDesc = Converter::descriptors_cvMat_cvMatVector(m_cvMat_descriptors);
      //key points vector associate key points with nodes in the 4th level (from leaves up)
      //assume the vocabulary tree has 6 levels, change the 4 otherwise
      m_sptr_vocab->transform(vCurrentDesc,m_bow_wordVec,m_bow_keyPointsVec,4);
    }
  }
  void Frame::setCameraPoseByTransrom_c2w(cv::Mat _T_c2w){
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    m_cvMat_T_c2w = _T_c2w.clone();
    updatePoseMatrices();
  }
  void Frame::updatePoseMatrices()
  { 
    m_cvMat_R_c2w   = m_cvMat_T_c2w.rowRange(0,3).colRange(0,3);
    m_cvMat_R_w2c   = m_cvMat_R_c2w.t();
    m_cvMat_t_c2w   = m_cvMat_T_c2w.rowRange(0,3).col(3);
    m_cvMat_origin  = -m_cvMat_R_w2c*m_cvMat_t_c2w;

    m_cvMat_T_w2c = cv::Mat::eye(4,4,m_cvMat_T_c2w.type());
    m_cvMat_R_w2c.copyTo(m_cvMat_T_w2c.rowRange(0,3).colRange(0,3));
    m_cvMat_origin.copyTo(m_cvMat_T_w2c.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << m_flt_baseLine/2.0, 0 , 0, 1.0);
    m_cvMat_center = m_cvMat_T_w2c * center;
  }
  bool Frame::isInImage(const float &_posX, const float &_posY) const
  {
    return (_posX>=m_flt_minX && _posX<m_flt_maxX && _posY>=m_flt_minY && _posY<m_flt_maxY);
  }
  bool Frame::isInCameraFrustum(std::shared_ptr<MapPoint> _sptrMapPoint, const float &_viewCosLimit){
    _sptrMapPoint->m_b_isTrackInView = false;
    //3D in world coordinate
    cv::Mat mapPointPosInWorld = _sptrMapPoint->getPosInWorld();
    //3D in camera coordinate
    cv::Mat mapPointPosInCamera = m_cvMat_R_c2w * mapPointPosInWorld + m_cvMat_t_c2w;
    //project into image and check if it is inside or outside
    const float xCord = m_flt_fx * mapPointPosInCamera.at<float>(0) / mapPointPosInCamera.at<float>(2) + m_flt_cx;
    const float yCord = m_flt_fy * mapPointPosInCamera.at<float>(1) / mapPointPosInCamera.at<float>(2) + m_flt_cy;
    //check if distance is in the scale invariance region of the map point
    const float maxDistance = _sptrMapPoint->getMaxDistanceInvariance();
    const float minDistance = _sptrMapPoint->getMinDistanceInvariance();
    const cv::Mat vectorCameraPos = mapPointPosInWorld - m_cvMat_origin;
    const float distance = cv::norm(vectorCameraPos);
    //check angle of view
    cv::Mat normalVector = _sptrMapPoint->getNormal();
    const float viewCos = vectorCameraPos.dot(normalVector)/distance;
    if(mapPointPosInCamera.at<float>(2)>=0.0 && isInImage(xCord,yCord) && distance>=minDistance && distance<=maxDistance && viewCos>=_viewCosLimit){
      //predict scale in the image
      const int predictedScaleLevel = _sptrMapPoint->predictScaleLevel(distance,*this);
      //used by tracking
      _sptrMapPoint->m_b_isTrackInView      = true;
      _sptrMapPoint->m_flt_trackProjX       = xCord;
      _sptrMapPoint->m_flt_trackProjRightX  = xCord - m_flt_baseLineTimesFx / mapPointPosInCamera.at<float>(2);
      _sptrMapPoint->m_flt_trackProjY       = yCord;
      _sptrMapPoint->m_int_trackScaleLevel  = predictedScaleLevel;  //optimize calculation time of m_int_trackScaleLevel
      _sptrMapPoint->m_flt_trackViewCos     = viewCos;
      return true;
    }else {
      return false;
    }
  }
  bool Frame::computeLocationInGrid(const cv::KeyPoint &_keyPoint, int &_locX, int &_locY){
    _locX = round((_keyPoint.pt.x - m_flt_minX) * m_flt_gridCellWidthInv);
    _locY = round((_keyPoint.pt.y - m_flt_minX) * m_flt_gridCellHeightInv);
    //as key points' coordinates are undistorted, they could go out of the image
    if(_locX<0 || _locX>=m_int_gridColsNum || _locY<0 || _locY>=m_int_gridRowsNum){
      return false;
    }else{
      return true;
    }
  }
  std::vector<int> Frame::getKeyPointsInArea(const float &_posX, const float &_posY, const float &_radius, const int &_minScaleLevel, const int &_maxScaleLevel) const{
    std::vector<int> vIndices;
    vIndices.reserve(m_int_keyPointsNum);
    const int minCellX = std::max(0,(int)floor((_posX-m_flt_minX-_radius)*m_flt_gridCellWidthInv));
    const int maxCellX = std::min((int)m_int_gridColsNum-1,(int)ceil((_posX-m_flt_minX+_radius)*m_flt_gridCellWidthInv));
    const int minCellY = std::max(0,(int)floor((_posY-m_flt_minY-_radius)*m_flt_gridCellHeightInv));
    const int maxCellY = std::min((int)m_int_gridRowsNum-1,(int)ceil((_posY-m_flt_minY+_radius)*m_flt_gridCellHeightInv));
    if(minCellX<m_int_gridColsNum && maxCellX>=0 && minCellY<m_int_gridRowsNum && maxCellY>=0){
      for(int ix = minCellX; ix<=maxCellX; ix++){
        for(int iy = minCellY; iy<=maxCellY; iy++){
          for(const int &idx : m_vvv_grid[ix][iy]){
            if(_minScaleLevel>0 || _maxScaleLevel>=0){  // check level
              if(m_v_keyPoints[idx].octave<_minScaleLevel || (_maxScaleLevel>=0 && m_v_keyPoints[idx].octave<_maxScaleLevel)){
                continue;
              }
            }
            if(fabs(m_v_keyPoints[idx].pt.x-_posX)>_radius && fabs(m_v_keyPoints[idx].pt.y-_posY)<_radius){
              vIndices.push_back(idx);
            }
          }
        }
      }
    }
    return vIndices;
  }
  void Frame::computeStereoMatches(){
    m_v_rightXcords = std::vector<float>(m_int_keyPointsNum, -1.0);
    m_v_depth       = std::vector<float>(m_int_keyPointsNum, -1.0);
    const int orbDistThd = (OrbMatcher::m_int_highThd + OrbMatcher::m_int_lowThd)/2;
    const int rowsNum = m_sptr_leftOrbExtractor->m_v_imagePyramid[0].rows;
    //assign key points to row table
    std::vector<std::vector<int>> vRowIndices(rowsNum,std::vector<int>());
    for(std::vector<int> &row : vRowIndices){
      row.reserve(200);
    }
    int i_for = 0;
    for(const cv::KeyPoint &keyPoint : m_v_rightKeyPoints){
      //original code may cause index out of range, so max min are added to ensure iy in range
      for(int iy=std::max(ceil(keyPoint.pt.y-2.0f*m_v_scaleFactors[keyPoint.octave]),0.0f);iy<=std::min(floor(keyPoint.pt.y+2.0f*m_v_scaleFactors[keyPoint.octave]),(float)rowsNum-1.0f);iy++){
        vRowIndices[iy].push_back(i_for);
      }
      i_for++;
    }
    //set limits for search
    const float minD = 0.0;
    const float maxD = m_flt_baseLineTimesFx / m_flt_baseLine;
    //for each left key point, search a match in the right
    std::vector<std::pair<int, int>> vDistIndices;
    vDistIndices.reserve(m_int_keyPointsNum);
    int leftIdx=0;
    for(const cv::KeyPoint &keyPoint : m_v_keyPoints){
      if(!vRowIndices[keyPoint.pt.y].empty() && keyPoint.pt.x>=minD){
        int bestDist = 256;
        int bestRightIdx = 0;
        //compare descriptor to right key points
        for(const int &rightIdx : vRowIndices[keyPoint.pt.y]){
          if(m_v_rightKeyPoints[rightIdx].octave>=keyPoint.octave-1 && m_v_rightKeyPoints[rightIdx].octave<=keyPoint.octave+1 && \
          m_v_rightKeyPoints[rightIdx].pt.x>=(keyPoint.pt.x-maxD) && m_v_rightKeyPoints[rightIdx].pt.x<=(keyPoint.pt.x-minD)){
            const int dist = OrbMatcher::computeDescriptorsDistance(m_cvMat_descriptors.row(leftIdx),m_cvMat_rightDescriptors.row(rightIdx));
            if(dist<bestDist){
              bestDist = dist;
              bestRightIdx = rightIdx;
            }
          }
        }
        //subpixel match by correlation
        if(bestDist<orbDistThd){
          //coordinates in image pyramid at key point scale
          const float leftScaledX  = round(keyPoint.pt.x*m_v_invScaleFactors[keyPoint.octave]);
          const float leftScaledY  = round(keyPoint.pt.y*m_v_invScaleFactors[keyPoint.octave]);
          const float rightScaledX = round(m_v_rightKeyPoints[bestRightIdx].pt.x*m_v_invScaleFactors[keyPoint.octave]);
          //sliding window search
          const int sadWindowSize = 5;  //sad means sum of absolute differences, an algorithm to match images
          cv::Mat leftImage = m_sptr_leftOrbExtractor->m_v_imagePyramid[keyPoint.octave].rowRange(leftScaledY-sadWindowSize,leftScaledY+sadWindowSize+1).colRange(leftScaledX-sadWindowSize,leftScaledX+sadWindowSize+1);
          leftImage.convertTo(leftImage,CV_32F);
          leftImage -= leftImage.at<float>(sadWindowSize,sadWindowSize)*cv::Mat::ones(leftImage.rows,leftImage.cols,CV_32F);
          int bestDist = 256;
          int bestRightCol = 0;
          const int slidingSize = 5;
          std::vector<float> vDists;
          vDists.reserve(2*slidingSize+1);
          if((rightScaledX+slidingSize-sadWindowSize)<0 || (rightScaledX+slidingSize+sadWindowSize+1)>=m_sptr_rightOrbExtractor->m_v_imagePyramid[keyPoint.octave].cols){
            continue;
          }
          for(int i=-slidingSize; i<=slidingSize; i++){
            cv::Mat rightImage = m_sptr_rightOrbExtractor->m_v_imagePyramid[keyPoint.octave].rowRange(leftScaledY-sadWindowSize,leftScaledY+sadWindowSize+1).colRange(rightScaledX+i-sadWindowSize,rightScaledX+i+sadWindowSize+1);
            rightImage.convertTo(rightImage,CV_32F);
            rightImage -= rightImage.at<float>(sadWindowSize,sadWindowSize)*cv::Mat::ones(rightImage.rows,rightImage.cols,CV_32F);
            const float dist = cv::norm(leftImage,rightImage,cv::NORM_L1);
            if(dist<bestDist){
              bestDist = dist;
              bestRightCol = i;
            }
            vDists.push_back(dist);
          }
          if(bestRightCol==-slidingSize || bestRightCol==slidingSize){
            continue;
          }
          //subpixel match (Parabola fitting)
          const float rightDelta = \
          (vDists[slidingSize+bestRightCol-1] - vDists[slidingSize+bestRightCol+1])/ \
          (2.0*(vDists[slidingSize+bestRightCol-1]+vDists[slidingSize+bestRightCol+1]-2.0*vDists[slidingSize+bestRightCol]));
          if(rightDelta<-1 || rightDelta>1){
            continue;
          }
          //rescale coordinate
          float bestRightX = m_v_scaleFactors[keyPoint.octave]*((float)rightScaledX+(float)rightDelta);
          float disparity = keyPoint.pt.x - bestRightX;
          if(disparity>=minD && disparity<maxD){
            if(disparity<=0){
              disparity=0.01;
              bestRightX = keyPoint.pt.x - 0.01;
            }
            m_v_depth[leftIdx] = m_flt_baseLineTimesFx / disparity;
            m_v_rightXcords[leftIdx] = bestRightX;
            vDistIndices.push_back(std::pair<int,int>(bestDist,leftIdx));
          }
        }
        leftIdx++;
      }
    }
    sort(vDistIndices.begin(),vDistIndices.end());
    for(const std::pair<int,int> &distIdx : vDistIndices){
      if(distIdx.first>=1.5*1.4*vDistIndices[vDistIndices.size()/2].first){
        m_v_rightXcords[distIdx.second]=-1;
        m_v_depth[distIdx.second]=-1;
      }else{
        break;
      }
    }
  }
  cv::Mat Frame::inverseProject(const int &_idx){
    if(m_v_depth[_idx]>0){
      cv::Mat point3DInCamera = (cv::Mat_<float>(3,1)<<\
      (m_v_keyPoints[_idx].pt.x-m_flt_cx)*m_v_depth[_idx]*m_flt_invFx,\
      (m_v_keyPoints[_idx].pt.y-m_flt_cy)*m_v_depth[_idx]*m_flt_invFy,\
      m_v_depth[_idx]);
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_R_w2c*point3DInCamera+m_cvMat_origin;
    }else{
      return cv::Mat();
    }
  }
}//YDORBSLAM
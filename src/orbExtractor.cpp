#include "orbExtractor.hpp"

namespace YDORBSLAM{
  void QuadTreeNode::divideNode(QuadTreeNode &_node1, QuadTreeNode &_node2, \
  QuadTreeNode &_node3, QuadTreeNode &_node4){
    //coordinate of center point in the ceil
    const int center_x = ceil(static_cast<float>(m_cvP_tr.x+m_cvP_tl.x)/2.0);
    const int center_y = ceil(static_cast<float>(m_cvP_bl.y+m_cvP_tl.y)/2.0);
    const cv::Point2i center = cv::Point2i(center_x,center_y);

    //boundaries of child ceils
    _node1.m_cvP_tl = m_cvP_tl;
    _node1.m_cvP_tr = cv::Point2i(center_x,   m_cvP_tl.y);
    _node1.m_cvP_bl = cv::Point2i(m_cvP_tl.x, center_y);
    _node1.m_cvP_br = center;
    _node1.m_v_keyPoints.reserve(m_v_keyPoints.size());
    
    _node2.m_cvP_tl = _node1.m_cvP_tr;
    _node2.m_cvP_tr = m_cvP_tr;
    _node2.m_cvP_bl = center;
    _node2.m_cvP_br = cv::Point2i(m_cvP_tr.x, center_y);
    _node2.m_v_keyPoints.reserve(m_v_keyPoints.size());

    _node3.m_cvP_tl = _node1.m_cvP_bl;
    _node3.m_cvP_tr = center;
    _node3.m_cvP_bl = m_cvP_bl;
    _node3.m_cvP_br = cv::Point2i(center_x,   m_cvP_tl.y);
    _node3.m_v_keyPoints.reserve(m_v_keyPoints.size());
 
    _node4.m_cvP_tl = center;
    _node4.m_cvP_tr = _node2.m_cvP_br;
    _node4.m_cvP_bl = _node3.m_cvP_br;
    _node4.m_cvP_br = m_cvP_br;
    _node4.m_v_keyPoints.reserve(m_v_keyPoints.size());

    //associate features (key points) to children
    for(const cv::KeyPoint &kp : m_v_keyPoints){
      if(kp.pt.x<center_x && kp.pt.y<center_y){
        _node1.m_v_keyPoints.push_back(kp);
      }else if(kp.pt.x<center_x && kp.pt.y>=center_y){
        _node3.m_v_keyPoints.push_back(kp);
      }else if(kp.pt.x>=center_x && kp.pt.y<center_y){
        _node2.m_v_keyPoints.push_back(kp);
      }else{
        _node4.m_v_keyPoints.push_back(kp);
      }
    }

    //check if the nodes are indivisable
    _node1.m_b_indivisible = (bool)(_node1.m_v_keyPoints.size()==1);
    _node2.m_b_indivisible = (bool)(_node2.m_v_keyPoints.size()==1);
    _node3.m_b_indivisible = (bool)(_node3.m_v_keyPoints.size()==1);
    _node4.m_b_indivisible = (bool)(_node4.m_v_keyPoints.size()==1);
  }

  const std::vector<cv::Point> OrbExtractor::m_v_pattern = {\
    cv::Point(8,-3),cv::Point(9,5),\
    cv::Point(4,2),cv::Point(7,-12),\
    cv::Point(-11,9),cv::Point(-8,2),\
    cv::Point(7,-12),cv::Point(12,-13),\
    cv::Point(2,-13),cv::Point(2,12),\
    cv::Point(1,-7),cv::Point(1,6),\
    cv::Point(-2,-10),cv::Point(-2,-4),\
    cv::Point(-13,-13),cv::Point(-11,-8),\
    cv::Point(-13,-3),cv::Point(-12,-9),\
    cv::Point(10,4),cv::Point(11,9),\
    cv::Point(-13,-8),cv::Point(-8,-9),\
    cv::Point(-11,7),cv::Point(-9,12),\
    cv::Point(7,7),cv::Point(12,6),\
    cv::Point(-4,-5),cv::Point(-3,0),\
    cv::Point(-13,2),cv::Point(-12,-3),\
    cv::Point(-9,0),cv::Point(-7,5),\
    cv::Point(12,-6),cv::Point(12,-1),\
    cv::Point(-3,6),cv::Point(-2,12),\
    cv::Point(-6,-13),cv::Point(-4,-8),\
    cv::Point(11,-13),cv::Point(12,-8),\
    cv::Point(4,7),cv::Point(5,1),\
    cv::Point(5,-3),cv::Point(10,-3),\
    cv::Point(3,-7),cv::Point(6,12),\
    cv::Point(-8,-7),cv::Point(-6,-2),\
    cv::Point(-2,11),cv::Point(-1,-10),\
    cv::Point(-13,12),cv::Point(-8,10),\
    cv::Point(-7,3),cv::Point(-5,-3),\
    cv::Point(-4,2),cv::Point(-3,7),\
    cv::Point(-10,-12),cv::Point(-6,11),\
    cv::Point(5,-12),cv::Point(6,-7),\
    cv::Point(5,-6),cv::Point(7,-1),\
    cv::Point(1,0),cv::Point(4,-5),\
    cv::Point(9,11),cv::Point(11,-13),\
    cv::Point(4,7),cv::Point(4,12),\
    cv::Point(2,-1),cv::Point(4,4),\
    cv::Point(-4,-12),cv::Point(-2,7),\
    cv::Point(-8,-5),cv::Point(-7,-10),\
    cv::Point(4,11),cv::Point(9,12)  ,\
    cv::Point(0,-8),cv::Point(1,-13),\
    cv::Point(-13,-2),cv::Point(-8,2),\
    cv::Point(-3,-2),cv::Point(-2,3),\
    cv::Point(-6,9),cv::Point(-4,-9),\
    cv::Point(8,12),cv::Point(10,7),\
    cv::Point(0,9),cv::Point(1,3),\
    cv::Point(7,-5),cv::Point(11,-10),\
    cv::Point(-13,-6),cv::Point(-11,0),\
    cv::Point(10,7),cv::Point(12,1),\
    cv::Point(-6,-3),cv::Point(-6,12),\
    cv::Point(10,-9),cv::Point(12,-4),\
    cv::Point(-13,8),cv::Point(-8,-12),\
    cv::Point(-13,0),cv::Point(-8,-4),\
    cv::Point(3,3),cv::Point(7,8),\
    cv::Point(5,7),cv::Point(10,-7),\
    cv::Point(-1,7),cv::Point(1,-12),\
    cv::Point(3,-10),cv::Point(5,6),\
    cv::Point(2,-4),cv::Point(3,-10),\
    cv::Point(-13,0),cv::Point(-13,5),\
    cv::Point(-13,-7),cv::Point(-12,12),\
    cv::Point(-13,3),cv::Point(-11,8),\
    cv::Point(-7,12),cv::Point(-4,7),\
    cv::Point(6,-10),cv::Point(12,8),\
    cv::Point(-9,-1),cv::Point(-7,-6),\
    cv::Point(-2,-5),cv::Point(0,12),\
    cv::Point(-12,5),cv::Point(-7,5),\
    cv::Point(3,-10),cv::Point(8,-13),\
    cv::Point(-7,-7),cv::Point(-4,5),\
    cv::Point(-3,-2),cv::Point(-1,-7),\
    cv::Point(2,9),cv::Point(5,-11),\
    cv::Point(-11,-13),cv::Point(-5,-13),\
    cv::Point(-1,6),cv::Point(0,-1),\
    cv::Point(5,-3),cv::Point(5,2 ),\
    cv::Point(-4,-13),cv::Point(-4,12),\
    cv::Point(-9,-6),cv::Point(-9,6),\
    cv::Point(-12,-10),cv::Point(-8,-4),\
    cv::Point(10,2),cv::Point(12,-3),\
    cv::Point(7,12),cv::Point(12,12),\
    cv::Point(-7,-13),cv::Point(-6,5),\
    cv::Point(-4,9),cv::Point(-3,4),\
    cv::Point(7,-1),cv::Point(12,2),\
    cv::Point(-7,6),cv::Point(-5,1),\
    cv::Point(-13,11),cv::Point(-12,5),\
    cv::Point(-3,7),cv::Point(-2,-6),\
    cv::Point(7,-8),cv::Point(12,-7),\
    cv::Point(-13,-7),cv::Point(-11,-12),\
    cv::Point(1,-3),cv::Point(12,12),\
    cv::Point(2,-6),cv::Point(3,0),\
    cv::Point(-4,3),cv::Point(-2,-13),\
    cv::Point(-1,-13),cv::Point(1,9),\
    cv::Point(7,1),cv::Point(8,-6),\
    cv::Point(1,-1),cv::Point(3,12),\
    cv::Point(9,1),cv::Point(12,6),\
    cv::Point(-1,-9),cv::Point(-1,3),\
    cv::Point(-13,-13),cv::Point(-10,5),\
    cv::Point(7,7),cv::Point(10,12),\
    cv::Point(12,-5),cv::Point(12,9),\
    cv::Point(6,3),cv::Point(7,11),\
    cv::Point(5,-13),cv::Point(6,10),\
    cv::Point(2,-12),cv::Point(2,3),\
    cv::Point(3,8),cv::Point(4,-6),\
    cv::Point(2,6),cv::Point(12,-13),\
    cv::Point(9,-12),cv::Point(10,3),\
    cv::Point(-8,4),cv::Point(-7,9),\
    cv::Point(-11,12),cv::Point(-4,-6),\
    cv::Point(1,12),cv::Point(2,-8),\
    cv::Point(6,-9),cv::Point(7,-4),\
    cv::Point(2,3),cv::Point(3,-2),\
    cv::Point(6,3),cv::Point(11,0),\
    cv::Point(3,-3),cv::Point(8,-8),\
    cv::Point(7,8),cv::Point(9,3),\
    cv::Point(-11,-5),cv::Point(-6,-4),\
    cv::Point(-10,11),cv::Point(-5,10),\
    cv::Point(-5,-8),cv::Point(-3,12),\
    cv::Point(-10,5),cv::Point(-9,0),\
    cv::Point(8,-1),cv::Point(12,-6),\
    cv::Point(4,-6),cv::Point(6,-11),\
    cv::Point(-10,12),cv::Point(-8,7),\
    cv::Point(4,-2),cv::Point(6,7),\
    cv::Point(-2,0),cv::Point(-2,12),\
    cv::Point(-5,-8),cv::Point(-5,2),\
    cv::Point(7,-6),cv::Point(10,12),\
    cv::Point(-9,-13),cv::Point(-8,-8),\
    cv::Point(-5,-13),cv::Point(-5,-2),\
    cv::Point(8,-8),cv::Point(9,-13),\
    cv::Point(-9,-11),cv::Point(-9,0),\
    cv::Point(1,-8),cv::Point(1,-2),\
    cv::Point(7,-4),cv::Point(9,1),\
    cv::Point(-2,1),cv::Point(-1,-4),\
    cv::Point(11,-6),cv::Point(12,-11),\
    cv::Point(-12,-9),cv::Point(-6,4),\
    cv::Point(3,7),cv::Point(7,12),\
    cv::Point(5,5),cv::Point(10,8),\
    cv::Point(0,-4),cv::Point(2,8),\
    cv::Point(-9,12),cv::Point(-5,-13),\
    cv::Point(0,7),cv::Point(2,12),\
    cv::Point(-1,2),cv::Point(1,7),\
    cv::Point(5,11),cv::Point(7,-9),\
    cv::Point(3,5),cv::Point(6,-8),\
    cv::Point(-13,-4),cv::Point(-8,9),\
    cv::Point(-5,9),cv::Point(-3,-3),\
    cv::Point(-4,-7),cv::Point(-3,-12),\
    cv::Point(6,5),cv::Point(8,0),\
    cv::Point(-7,6),cv::Point(-6,12),\
    cv::Point(-13,6),cv::Point(-5,-2),\
    cv::Point(1,-10),cv::Point(3,10),\
    cv::Point(4,1),cv::Point(8,-4),\
    cv::Point(-2,-2),cv::Point(2,-13),\
    cv::Point(2,-12),cv::Point(12,12),\
    cv::Point(-2,-13),cv::Point(0,-6),\
    cv::Point(4,1),cv::Point(9,3),\
    cv::Point(-6,-10),cv::Point(-3,-5),\
    cv::Point(-3,-13),cv::Point(-1,1),\
    cv::Point(7,5),cv::Point(12,-11),\
    cv::Point(4,-2),cv::Point(5,-7),\
    cv::Point(-13,9),cv::Point(-9,-5),\
    cv::Point(7,1),cv::Point(8,6),\
    cv::Point(7,-8),cv::Point(7,6),\
    cv::Point(-7,-4),cv::Point(-7,1),\
    cv::Point(-8,11),cv::Point(-7,-8),\
    cv::Point(-13,6),cv::Point(-12,-8),\
    cv::Point(2,4),cv::Point(3,9),\
    cv::Point(10,-5),cv::Point(12,3),\
    cv::Point(-6,-5),cv::Point(-6,7),\
    cv::Point(8,-3) ,cv::Point(9,-8),\
    cv::Point(2,-12),cv::Point(2,8),\
    cv::Point(-11,-2),cv::Point(-10,3),\
    cv::Point(-12,-13),cv::Point(-7,-9),\
    cv::Point(-11,0),cv::Point(-10,-5),\
    cv::Point(5,-3),cv::Point(11,8 ),\
    cv::Point(-2,-13),cv::Point(-1,12),\
    cv::Point(-1,-8),cv::Point(0,9),\
    cv::Point(-13,-11),cv::Point(-12,-5),\
    cv::Point(-10,-2),cv::Point(-10,11),\
    cv::Point(-3,9),cv::Point(-2,-13),\
    cv::Point(2,-3),cv::Point(3,2),\
    cv::Point(-9,-13),cv::Point(-4,0),\
    cv::Point(-4,6),cv::Point(-3,-10),\
    cv::Point(-4,12),cv::Point(-2,-7),\
    cv::Point(-6,-11),cv::Point(-4,9),\
    cv::Point(6,-3),cv::Point(6,11),\
    cv::Point(-13,11),cv::Point(-5,5),\
    cv::Point(11,11),cv::Point(12,6),\
    cv::Point(7,-5),cv::Point(12,-2),\
    cv::Point(-1,12),cv::Point(0,7),\
    cv::Point(-4,-8),cv::Point(-3,-2),\
    cv::Point(-7,1),cv::Point(-6,7),\
    cv::Point(-13,-12),cv::Point(-8,-13),\
    cv::Point(-7,-2),cv::Point(-6,-8),\
    cv::Point(-8,5),cv::Point(-6,-9),\
    cv::Point(-5,-1),cv::Point(-4,5),\
    cv::Point(-13,7),cv::Point(-8,10),\
    cv::Point(1,5),cv::Point(5,-13),\
    cv::Point(1,0),cv::Point(10,-13),\
    cv::Point(9,12),cv::Point(10,-1),\
    cv::Point(5,-8),cv::Point(10,-9),\
    cv::Point(-1,11),cv::Point(1,-13),\
    cv::Point(-9,-3),cv::Point(-6,2),\
    cv::Point(-1,-10),cv::Point(1,12),\
    cv::Point(-13,1),cv::Point(-8,-10),\
    cv::Point(8,-11),cv::Point(10,-6),\
    cv::Point(2,-13),cv::Point(3,-6),\
    cv::Point(7,-13),cv::Point(12,-9),\
    cv::Point(-10,-10),cv::Point(-5,-7),\
    cv::Point(-10,-8),cv::Point(-8,-13),\
    cv::Point(4,-6),cv::Point(8,5 ),\
    cv::Point(3,12),cv::Point(8,-13),\
    cv::Point(-4,2),cv::Point(-3,-3),\
    cv::Point(5,-13),cv::Point(10,-12),\
    cv::Point(4,-13),cv::Point(5,-1),\
    cv::Point(-9,9),cv::Point(-4,3),\
    cv::Point(0,3),cv::Point(3,-9),\
    cv::Point(-12,1),cv::Point(-6,1),\
    cv::Point(3,2),cv::Point(4,-8),\
    cv::Point(-10,-10),cv::Point(-10,9),\
    cv::Point(8,-13),cv::Point(12,12),\
    cv::Point(-8,-12),cv::Point(-6,-5),\
    cv::Point(2,2),cv::Point(3,7),\
    cv::Point(10,6),cv::Point(11,-8),\
    cv::Point(6,8),cv::Point(8,-12),\
    cv::Point(-7,10),cv::Point(-6,5),\
    cv::Point(-3,-9),cv::Point(-3,9),\
    cv::Point(-1,-13),cv::Point(-1,5),\
    cv::Point(-3,-7),cv::Point(-3,4),\
    cv::Point(-8,-2),cv::Point(-8,3),\
    cv::Point(4,2),cv::Point(12,12),\
    cv::Point(2,-5),cv::Point(3,11),\
    cv::Point(6,-9),cv::Point(11,-13),\
    cv::Point(3,-1),cv::Point(7,12),\
    cv::Point(11,-1),cv::Point(12,4),\
    cv::Point(-3,0),cv::Point(-3,6),\
    cv::Point(4,-11),cv::Point(4,12),\
    cv::Point(2,-4),cv::Point(2,1),\
    cv::Point(-10,-6),cv::Point(-8,1),\
    cv::Point(-13,7),cv::Point(-11,1),\
    cv::Point(-13,12),cv::Point(-11,-13),\
    cv::Point(6,0),cv::Point(11,-13),\
    cv::Point(0,-1),cv::Point(1,4),\
    cv::Point(-13,3),cv::Point(-9,-2),\
    cv::Point(-9,8),cv::Point(-6,-3),\
    cv::Point(-13,-6),cv::Point(-8,-2),\
    cv::Point(5,-9),cv::Point(8,10),\
    cv::Point(2,7),cv::Point(3,-9),\
    cv::Point(-1,-6),cv::Point(-1,-1),\
    cv::Point(9,5),cv::Point(11,-2),\
    cv::Point(11,-3),cv::Point(12,-8),\
    cv::Point(3,0),cv::Point(3,5),\
    cv::Point(-1,4),cv::Point(0,10),\
    cv::Point(3,-6),cv::Point(4,5),\
    cv::Point(-13,0),cv::Point(-10,5),\
    cv::Point(5,8),cv::Point(12,11),\
    cv::Point(8,9),cv::Point(9,-6),\
    cv::Point(7,-4),cv::Point(8,-12),\
    cv::Point(-10,4),cv::Point(-10,9),\
    cv::Point(7,3),cv::Point(12,4),\
    cv::Point(9,-7),cv::Point(10,-2),\
    cv::Point(7,0),cv::Point(12,-2),\
    cv::Point(-1,-6),cv::Point(0,-11)      
  };

  OrbExtractor::OrbExtractor(int _keyPointsNum, float _scaleFactor, int _levelsNum,\
  int _initFastThd, int _minFastThd):m_int_keyPointsNum(_keyPointsNum),\
  m_flt_scaleFactor(_scaleFactor),m_int_levelsNum(_levelsNum),\
  m_int_initFastThd(_initFastThd),m_int_minFastThd(_initFastThd){
    m_v_scaleFactors.reserve(m_int_levelsNum);
    m_v_scaleFactorSquares.reserve(m_int_levelsNum);
    m_v_invScaleFactors.reserve(m_int_levelsNum);
    m_v_invScaleFactorSquares.reserve(m_int_levelsNum);
    m_v_imagePyramid.reserve(m_int_levelsNum);
    m_v_keyPointsNumsPerLevel.reserve(m_int_levelsNum);
    int keyPointsNumPerLevel = round(\
    m_int_keyPointsNum*(1 - 1.0/m_flt_scaleFactor)/(1.0 - pow(1.0/m_flt_scaleFactor, m_int_levelsNum)));
    int keyPointsNumSum = 0;
    for(int i=0;i<m_int_levelsNum;i++){
      m_v_scaleFactors.push_back(pow(m_flt_scaleFactor,i));
      m_v_scaleFactorSquares.push_back(pow(m_flt_scaleFactor,2*i));
      m_v_invScaleFactors.push_back(pow(m_flt_scaleFactor,-i));
      m_v_invScaleFactorSquares.push_back(pow(m_flt_scaleFactor,-2*i));
      if(i<m_int_levelsNum-1){
        m_v_keyPointsNumsPerLevel.push_back(keyPointsNumPerLevel);
        keyPointsNumSum += keyPointsNumPerLevel;
        keyPointsNumPerLevel =round((float)keyPointsNumPerLevel/m_flt_scaleFactor);
      }else{
        m_v_keyPointsNumsPerLevel.push_back(std::max(m_int_keyPointsNum - keyPointsNumSum, 0));
      }   
    }
    m_v_maxXcords.resize(m_int_halfPatchSize + 1);
    int maxYcord = floor(m_int_halfPatchSize * sqrt(2.0)/2.0 + 1.0);
    int minYcord = ceil(m_int_halfPatchSize * sqrt(2.0)/2.0);
    for(int v=0;v<=maxYcord;v++){
      m_v_maxXcords.push_back(round(sqrt(pow(m_int_halfPatchSize,2) + pow(v,2))));
    }
    for(int v=m_int_halfPatchSize, i=0; v>=minYcord;v--){
      while(m_v_maxXcords[i]==m_v_maxXcords[i+1]){
        i++;
      }
      m_v_maxXcords[v]=i;
      i++;
    }
  }
  void OrbExtractor::extractAndCompute(const cv::InputArray &_image, \
  std::vector<cv::KeyPoint> &_keyPoints, cv::OutputArray &_descriptors){
    if(_image.empty()){
      return;
    }
    cv::Mat image = _image.getMat();
    assert(image.type()==CV_8UC1);
    computePyramid(image);
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
    computeKeyPointsPyramid(allKeyPoints);
    cv::Mat descriptors;
    int keyPointsNum = 0;
    for(int level=0;level<m_int_levelsNum;level++){
      keyPointsNum+=(int)allKeyPoints[level].size();
    }
    if(keyPointsNum==0){
      _descriptors.release();
    }else{
      _descriptors.create(keyPointsNum, 32, CV_8U);
      descriptors = _descriptors.getMat();
    }
    _keyPoints.clear();
    _keyPoints.reserve(keyPointsNum);
    int offset=0;
    for(int level=0;level<m_int_levelsNum;level++){
      int levelKeyPointsNum = (int)allKeyPoints[level].size();
      if(levelKeyPointsNum==0){
        continue;
      }
      //prepocess the resized image
      cv::Mat workingImage = m_v_imagePyramid[level].clone();
      cv::GaussianBlur(workingImage,workingImage,cv::Size(7,7),2,2,cv::BORDER_REFLECT_101);
      cv::Mat desc = descriptors.rowRange(offset,offset+levelKeyPointsNum);
      computeDescriptors(workingImage,allKeyPoints[level],m_v_pattern,desc);
      offset += levelKeyPointsNum;
      //scale key point coordinates
      if(level!=0){
        float scale=m_v_scaleFactors[level];
        for(cv::KeyPoint &keyPoint : allKeyPoints[level]){
          keyPoint.pt *= scale;
        }
      }
      _keyPoints.insert(_keyPoints.end(),allKeyPoints[level].begin(),allKeyPoints[level].end());
    }
  }
  void OrbExtractor::computeOrientation(const cv::Mat &_image, const std::vector<int> &_maxXcords, std::vector<cv::KeyPoint> &_keyPoints){
    for(cv::KeyPoint &keyPoint : _keyPoints){
      int m01 = 0, m10 = 0;
      //treat the center line differently, v=0
      for(int u=-m_int_halfPatchSize;u<=m_int_halfPatchSize;u++){
        m10 += u * _image.at<unsigned char>(cvRound(keyPoint.pt.y),cvRound(keyPoint.pt.x)+u);
      }
      for(int v=1;v<=m_int_halfPatchSize;v++){
        //proceed over the two lines
        int vSum = 0;
        int d = m_v_maxXcords[v];
        for (int u=-d;u<=d;u++){
          int posVal = _image.at<unsigned char>(cvRound(keyPoint.pt.y)+v,cvRound(keyPoint.pt.x)+u);
          int negVal = _image.at<unsigned char>(cvRound(keyPoint.pt.y)-v,cvRound(keyPoint.pt.x)+u);
          vSum += (posVal - negVal);
          m10 += u * (posVal + negVal);
        }
        m01 += v * vSum;
      }
      keyPoint.angle = cv::fastAtan2((float)m01,(float)m10);
    }
  }
  void OrbExtractor::computeDescriptors(const cv::Mat &_image, const std::vector<cv::KeyPoint> &_keyPoints, const std::vector<cv::Point> &_pattern, cv::Mat &_descriptors){
    _descriptors = cv::Mat::zeros((int)_keyPoints.size(),32,CV_8UC1);
    for(int i=0;i<_keyPoints.size();i++){
      float angle = (float)_keyPoints[i].angle * m_flt_deg2radFactor;
      float cosA = (float)cos(angle), sinB = (float)sin(angle);
      #define GET_VALUE(idx) \
        _image.at<unsigned char>(cvRound(_keyPoints[i].pt.y)+cvRound(_pattern[idx].x*sinB+_pattern[idx].y*cosA), \
                             cvRound(_keyPoints[i].pt.x)+cvRound(_pattern[idx].x*cosA-_pattern[idx].y*sinB))
      for(int j=0;j<32;j++){
        unsigned char t0,t1,val;
        int offset;
        offset = j*16;
        t0 = GET_VALUE(0+offset); t1 = GET_VALUE(1+offset);
        val = t0 < t1;
        t0 = GET_VALUE(2+offset); t1 = GET_VALUE(3+offset);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4+offset); t1 = GET_VALUE(5+offset);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6+offset); t1 = GET_VALUE(7+offset);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8+offset); t1 = GET_VALUE(9+offset);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10+offset); t1 = GET_VALUE(11+offset);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12+offset); t1 = GET_VALUE(13+offset);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14+offset); t1 = GET_VALUE(15+offset);
        val |= (t0 < t1) << 7;
        _descriptors.at<unsigned char>(i,j)=val;
      }
      #undef GET_VALUE
    }
  }
  void OrbExtractor::distributeQuadTree(const std::vector<cv::KeyPoint> &_keyPointsIn, const int &_minX, const int &_maxX, const int &_minY, const int &_maxY, const int &_desiredKeyPointsNum, std::vector<cv::KeyPoint> &_keyPointsOut){
    std::list<QuadTreeNode> nodesList;
    QuadTreeNode node0;
    node0.m_cvP_tl = cv::Point2i(0,0);
    node0.m_cvP_tr = cv::Point2i(_maxX -_minX,0);
    node0.m_cvP_bl = cv::Point2i(node0.m_cvP_tl.x,_maxY-_minY);
    node0.m_cvP_br = cv::Point2i(node0.m_cvP_tr.x,_maxY-_minY);
    for(const cv::KeyPoint &keyPoint : _keyPointsIn){
      node0.m_v_keyPoints.push_back(keyPoint);
    }
    nodesList.push_back(node0);
    for(std::list<QuadTreeNode>::iterator lit=nodesList.begin();lit!=nodesList.end();){
      if(lit->m_v_keyPoints.size()==1){
        lit->m_b_indivisible = true;
        lit++;
      }else if(lit->m_v_keyPoints.empty()){
        lit = nodesList.erase(lit);
      }else{
        lit++;
      }
    }
    int sizeInLastIter=0;
    std::vector<std::pair<int,std::reference_wrapper<QuadTreeNode>>> vSizeAndRefToNode;
    vSizeAndRefToNode.reserve(nodesList.size()*4);
    while((int)nodesList.size()>sizeInLastIter&&((int)nodesList.size()<_desiredKeyPointsNum)){
      int expandableNodesNum = 0;
      sizeInLastIter = nodesList.size();
      vSizeAndRefToNode.clear();
      if(((int)nodesList.size()+expandableNodesNum*3)<=_desiredKeyPointsNum){
        for(std::list<QuadTreeNode>::iterator lit=nodesList.begin();lit!=nodesList.end();){
          if(lit->m_b_indivisible){
            lit++;
            continue;
          }else{
            std::vector<QuadTreeNode> nodesVector = {QuadTreeNode(),QuadTreeNode(),QuadTreeNode(),QuadTreeNode()};
            lit->divideNode(nodesVector[0],nodesVector[1],nodesVector[2],nodesVector[3]);
            for(QuadTreeNode &node : nodesVector){
              if(node.m_v_keyPoints.size()>0){
                if(node.m_v_keyPoints.size()==1){
                  node.m_b_indivisible = true;
                }
                nodesList.push_front(node);
                if(node.m_v_keyPoints.size()>1){
                  vSizeAndRefToNode.push_back(std::make_pair(node.m_v_keyPoints.size(),std::ref(nodesList.front())));
                  expandableNodesNum++;
                  nodesList.front().m_lit_begin = nodesList.begin();
                }
              }
            }
            lit=nodesList.erase(lit);
          }
        }
      }else{
        std::vector<std::pair<int,std::reference_wrapper<QuadTreeNode>>> vPreSizeAndRefToNode = vSizeAndRefToNode;
        vSizeAndRefToNode.clear();
        std::sort(vPreSizeAndRefToNode.begin(),vPreSizeAndRefToNode.end(),\
        [](std::pair<int,std::reference_wrapper<QuadTreeNode>> &a, \
        std::pair<int,std::reference_wrapper<QuadTreeNode>> &b) \
        {return (a.first > b.first);});
        for(int j=vPreSizeAndRefToNode.size()-1;j>=0;j--){
          std::vector<QuadTreeNode> nodesVector = {QuadTreeNode(),QuadTreeNode(),QuadTreeNode(),QuadTreeNode()};
          vPreSizeAndRefToNode[j].second.get().divideNode(nodesVector[0],nodesVector[1],nodesVector[2],nodesVector[3]);
          for(QuadTreeNode &node : nodesVector){
            if(node.m_v_keyPoints.size()>0){
              if(node.m_v_keyPoints.size()==1){
                node.m_b_indivisible = true;
              }
              nodesList.push_front(node);
              if(node.m_v_keyPoints.size()>1){
                vSizeAndRefToNode.push_back(std::make_pair(node.m_v_keyPoints.size(),std::ref(nodesList.front())));
              }
            }
          }
          nodesList.erase(vPreSizeAndRefToNode[j].second.get().m_lit_begin);
        }
      }
    }
    //retain the best point in each node
    _keyPointsOut.clear();
    _keyPointsOut.reserve(_desiredKeyPointsNum*2);
    for(const QuadTreeNode &node : nodesList){
      std::vector<cv::KeyPoint> keyPointsVector = node.m_v_keyPoints;
      std::sort(keyPointsVector.begin(),keyPointsVector.end(),\
      [](cv::KeyPoint &a, cv::KeyPoint &b) {return (a.response > b.response);});
      _keyPointsOut.push_back(keyPointsVector.front());
    }
    if(_keyPointsOut.size()>_desiredKeyPointsNum){
      _keyPointsOut.resize(_desiredKeyPointsNum);
    }
  }
  void OrbExtractor::computeKeyPointsPyramid(std::vector<std::vector<cv::KeyPoint>> &_allKeyPoints){
    _allKeyPoints.reserve(m_int_levelsNum);
    _allKeyPoints.resize(m_int_levelsNum);
    const float widthHeight = 30.0;
    for(int level=0;level<m_int_levelsNum;level++){
      const int minBorderX = m_int_maxPadSize - 3;
      const int minBorderY = minBorderX;
      const int maxBorderX = m_v_imagePyramid[level].cols - (m_int_maxPadSize - 3);
      const int maxBorderY = m_v_imagePyramid[level].rows - (m_int_maxPadSize - 3);
      std::vector<cv::KeyPoint> keyPointsToDistr;
      keyPointsToDistr.reserve(m_int_keyPointsNum*10);
      const float width = (maxBorderX - minBorderX);
      const float height = (maxBorderY - minBorderY);
      const int colsNum = width/widthHeight;
      const int rowsNum = height/widthHeight;
      const int cellWidth = ceil(width/colsNum);
      const int cellHeight = ceil(height/rowsNum);
      for(int i=0;i<rowsNum;i++){
        const float initY = minBorderY + i*cellHeight;
        float maxY = initY + cellHeight + 6;
        if(initY>=maxBorderY-3){
          continue;
        }
        if(maxY>maxBorderY){
          maxY=maxBorderY;
        }
        for(int j=0;j<colsNum;j++){
          const float initX = minBorderX + j*cellWidth;
          float maxX = initX + cellWidth + 6;
          if(initX>=maxBorderX - 6){
            continue;
          }
          if(maxX>maxBorderX){
            maxX=maxBorderX;
          }
          std::vector<cv::KeyPoint> keyPointsInCell;
          cv::FAST(m_v_imagePyramid[level].rowRange(initY,maxY).colRange(initX,maxX),keyPointsInCell,m_int_initFastThd,true);
          if(keyPointsInCell.empty()){
            cv::FAST(m_v_imagePyramid[level].rowRange(initY,maxY).colRange(initX,maxX),keyPointsInCell,m_int_minFastThd,true);
          }
          if(!keyPointsInCell.empty()){
            for(cv::KeyPoint &keyPoint : keyPointsInCell){
                keyPoint.pt.x+=j*cellWidth;
                keyPoint.pt.y+=i*cellHeight;
                keyPointsToDistr.push_back(keyPoint);
              }
          }
        }
      }
      distributeQuadTree(keyPointsToDistr, minBorderX, maxBorderX, minBorderY, maxBorderY, m_v_keyPointsNumsPerLevel[level], _allKeyPoints[level]);
      const int scaledPatchSize = m_int_patchSize * m_v_scaleFactors[level];
      for(cv::KeyPoint &keyPoint : _allKeyPoints[level]){
        keyPoint.pt.x += minBorderX;
        keyPoint.pt.y += minBorderY;
        keyPoint.octave = level;
        keyPoint.size = scaledPatchSize;
      }
      computeOrientation(m_v_imagePyramid[level], m_v_maxXcords, _allKeyPoints[level]);
    }
  }
  void OrbExtractor::computePyramid(cv::Mat &_image){
    int n=0;
    for(int level=0;level<m_int_levelsNum;level++){
      float scale = m_v_invScaleFactors[level];
      cv::Size sz(cvRound((float)_image.cols*scale),cvRound((float)_image.rows*scale));
      cv::Size fullSz(sz.width+m_int_maxPadSize*2,sz.height+m_int_maxPadSize*2);
      cv::Mat tmp(fullSz,_image.type());
      m_v_imagePyramid.push_back(tmp(cv::Rect(m_int_maxPadSize,m_int_maxPadSize,sz.width,sz.height)));
      if(level!=0){
        cv::resize(m_v_imagePyramid[level-1], m_v_imagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(m_v_imagePyramid[level], tmp, m_int_maxPadSize, m_int_maxPadSize, m_int_maxPadSize, m_int_maxPadSize, \
                          cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);            
      }else{
        cv::copyMakeBorder(_image, tmp, m_int_maxPadSize, m_int_maxPadSize, m_int_maxPadSize, m_int_maxPadSize, cv::BORDER_REFLECT_101);            
      }
    }
  }
}//namespace YDORBSLAM

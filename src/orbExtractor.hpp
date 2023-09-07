/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_ORBEXTRACTOR_HPP
#define YDORBSLAM_ORBEXTRACTOR_HPP

#include <memory>
#include <math.h>
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace YDORBSLAM{
  //a quadtree is used for evenly sampling features(key points)
  struct QuadTreeNode{
    QuadTreeNode() : m_b_indivisible(false){}
    void divideNode(QuadTreeNode &_node1, QuadTreeNode &_node2, \
    QuadTreeNode &_node3, QuadTreeNode &_node4);
    std::vector<cv::KeyPoint> m_v_keyPoints;
    cv::Point2i m_cvP_tl, m_cvP_tr, m_cvP_bl, m_cvP_br;
    std::list<QuadTreeNode>::iterator m_lit_begin;
    bool m_b_indivisible;
  };

  class OrbExtractor{
    public:
    OrbExtractor(int _keyPointsNum, float _scaleFactor, int _levelsNum, \
    int _initFastThd, int _minFastThd);
    ~OrbExtractor(){}
    //extract ORB features(key points) and compuite descriptors on an image.
    //ORB are dispersed on the image using an octree (actually quadtree).
    //note: cv objects are shallow copied with operator = by default
    void extractAndCompute(const cv::InputArray &_image, \
    std::vector<cv::KeyPoint> &_keyPoints, cv::OutputArray &_descriptors);
    //gets
    int inline getKeyPointsNum(){return m_int_levelsNum;}
    float inline getScaleFactor(){return m_flt_scaleFactor;}
    int inline getLevelsNum(){return m_int_levelsNum;}
    std::vector<float> inline getScaleFactors(){return m_v_scaleFactors;}
    std::vector<float> inline getInvScaleFactors(){return m_v_invScaleFactors;}
    std::vector<float> inline getScaleFactorSquares(){return m_v_scaleFactorSquares;}
    std::vector<float> inline getInvScaleFactorSquares(){return m_v_invScaleFactorSquares;}
    std::vector<cv::Mat> inline getImagePyramid(){return m_v_imagePyramid;}
    std::vector<cv::Mat> m_v_imagePyramid;
    protected:
    void computeOrientation(const cv::Mat &_image, const std::vector<int> &_maxXcords, std::vector<cv::KeyPoint> &_keyPoints);
    void computeDescriptors(const cv::Mat &_image, const std::vector<cv::KeyPoint> &_keyPoints, const std::vector<cv::Point> &_pattern, cv::Mat &_descriptors);
    void distributeQuadTree(const std::vector<cv::KeyPoint> &_keyPointsIn, const int &_minX, const int &_maxX, const int &_minY, const int &_maxY, const int &_desiredKeyPointsNum, std::vector<cv::KeyPoint> &_keyPointsOut);
    void computeKeyPointsPyramid(std::vector<std::vector<cv::KeyPoint>> &_allKeyPoints);
    void computePyramid(cv::Mat &_image);
    const float m_flt_deg2radFactor = M_PI / 180.0;
    int m_int_keyPointsNum;
    float m_flt_scaleFactor;
    int m_int_levelsNum;
    int m_int_initFastThd;
    int m_int_minFastThd;
    std::vector<int> m_v_keyPointsNumsPerLevel;
    //maximum x coordinate of the circle to compute angle
    std::vector<int> m_v_maxXcords;
    std::vector<float> m_v_scaleFactors;
    std::vector<float> m_v_invScaleFactors;
    std::vector<float> m_v_scaleFactorSquares;
    std::vector<float> m_v_invScaleFactorSquares;
    const int m_int_patchSize = 31;
    const int m_int_halfPatchSize = 15;
    const int m_int_maxPadSize = 19;
    static const std::vector<cv::Point> m_v_pattern;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_ORBEXTRACTOR_HPP
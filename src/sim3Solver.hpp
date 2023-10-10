/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_SIM3_HPP
#define YDORBSLAM_SIM3_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <vector>
#include <memory>

#include "keyFrame.hpp"
#include "mapPoint.hpp"

namespace YDORBSLAM{
  class Sim3Solver{
    public:
    Sim3Solver(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const bool _bIsScaleFixed = true);
    void setRansacParameters(float _probability = 0.99, int _minInliersNum = 6, int _maxIterNum = 300);
    cv::Mat find(std::vector<bool> &_vIsInliers, int &_inliersNum);
    cv::Mat iterate(int _iterNum, bool &_bIsNoMore, std::vector<bool> &_vIsInliers, int &_inliersNum);
    cv::Mat getEstimatedRotation();
    cv::Mat getEstimatedTranslation();
    float getEstimatedScale();
    protected:
    void computeCentroid(cv::Mat &_mapPoint3DInCamera, cv::Mat &_mapPoint3DRel2Centroid, cv::Mat &_centroid);
    void computeSim3(cv::Mat &_firstMapPoint3DInCamera, cv::Mat &_secondMapPoint3DInCamera);
    void checkInliers();
    void project(const std::vector<cv::Mat> &_vMapPointsPosInCamera, std::vector<cv::Mat> &_vPointsPosInImage, cv::Mat _transformation_target2origin, cv::Mat _interParam);
    void transferCamera2Image(const std::vector<cv::Mat> &_P3Dc, std::vector<cv::Mat> &_vP2D, cv::Mat _interParam);
    std::shared_ptr<KeyFrame> m_sptr_firstKeyFrame, m_sptr_secondKeyFrame;
    std::vector<cv::Mat> m_v_firstMapPointsPosInCamera, m_v_secondMapPointsPosInCamera;
    std::vector<std::shared_ptr<MapPoint>> m_v_firstMapPoints, m_v_secondMapPoints, m_v_matchedMapPoints;
    std::vector<int> m_v_firstIndices, m_v_firstSquaredSigmas, m_v_secondSquaredSigmas, m_v_firstMaxError, m_v_secondMaxError;
    int m_int_correspondencesNum, m_int_matchedMapPointsNum;
    //current estimation
    cv::Mat m_cvMat_currentRotation_first2second, m_cvMat_currentTranslation_first2second;
    float m_flt_currentScale_first2second;
    cv::Mat m_cvMat_currentTransform_first2second, m_cvMat_currentTransform_second2first;
    std::vector<bool> m_v_isInliers;
    int m_int_inliersNum;
    //current ransac state
    int m_int_iterNum = 0;
    std::vector<bool> m_v_isBestInliers;
    int m_int_bestInliersNum = 0;
    cv::Mat m_cvMat_bestTransform_first2second, m_cvMat_bestRotation_first2second, m_cvMat_bestTranslation_first2second;
    float m_flt_bestScale;
    //scale is fixed to one in stereo/RGBD case
    bool m_b_isScaleFixed;
    //indices for random selection
    std::vector<int> m_v_allIndices;
    //projections
    std::vector<cv::Mat> m_v_firstPointInFirstImage, m_v_secondPointInSecondImage;
    //ransac probability
    float m_flt_ransacProb;
    //min ransac inliers number and max ransac iterations number
    int m_int_minRansacInliersNum, m_int_maxRansacIterNum;
    //threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*m_flt_squaredSigma
    float m_flt_thd, m_flt_squaredSigma;
    //calibration
    cv::Mat m_cvMat_firstIntParMat, m_cvMat_secondIntParMat;
  };
}//namespace YDORBSLAM

#endif //YDORBSLAM_PNP_HPP
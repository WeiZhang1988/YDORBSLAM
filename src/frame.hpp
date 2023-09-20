/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_FRAME_HPP
#define YDORBSLAM_FRAME_HPP

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"
#include "orbExtractor.hpp"

namespace YDORBSLAM{
  class MapPoint;
  class KeyFrame;
  class Frame {
    public:
    Frame() {
      m_int_ID = m_int_reservedID++;
    };
    //copy constructor, copy from object reference
    Frame(const Frame &_frame);
    //constructor for stereo cameras
    Frame(const cv::Mat &_leftImage, const cv::Mat &_rightImage, const double &_timeStamp, const cv::Mat &_camIntParMat, const cv::Mat &_imageDistCoef, const cv::Mat &_rightImageDistCoef, const float &_baseLineTimesFx, const float &_depthThd, std::shared_ptr<OrbExtractor> _sptrLeftExtractor, std::shared_ptr<OrbExtractor> _sptrRightExtractor, std::shared_ptr<DBoW3::Vocabulary> _sptrVocab);
    //constructor for RGB-D cameras
    Frame(const cv::Mat &_grayImage, const cv::Mat &_depthImage, const double &_timeStamp, const cv::Mat &_camIntParMat, const cv::Mat &_imageDistCoef, const float &_baseLineTimesFx, const float &_depthThd, std::shared_ptr<OrbExtractor> _sptrExtractor, std::shared_ptr<DBoW3::Vocabulary> _sptrVocab);
    //extract ORB on the image. channel flag = false means left, true means right
    void extractOrb(const cv::Mat &_image, const bool &_isRight=false);
    //pose functions#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    //set camera pose
    void setCameraPoseByTransform_c2w(cv::Mat _T_c2w);
    //get camera pose
    inline cv::Mat getCameraPoseByTransform_c2w(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_T_c2w.clone();
    }
    //get inverse of camera pose
    inline cv::Mat getInverseCameraPoseByTransform_w2c(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_T_w2c.clone();
    }
    //get camera origin coordinate
    inline cv::Mat getCameraOriginInWorld(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_origin.clone();
    }
    //get stereo center coordinate
    inline cv::Mat getStereoCenterInWorld(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_center.clone();
    }
    //get rotation from camera to world
    inline cv::Mat getRotation_c2w(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_R_c2w.clone();
    }
    //get rotation from world to camera
    inline cv::Mat getRotation_w2c(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_R_w2c.clone();
    }
    //get translation from camera to world
    inline cv::Mat getTranslation_c2w(){
      std::unique_lock<std::mutex> lock(m_mutex_pose);
      return m_cvMat_t_c2w.clone();
    }
    //compute rotation, translation and camera origin matrices from the camera pose
    //doubt the computation of mOw in original ORB-SLAM is wrong
    void updatePoseMatrices();
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####pose functions
    //compute bag of word representation
    void computeBoW();
    //key points functions#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    //get all key points in an area
    std::vector<int> getKeyPointsInArea(const float &_posX, const float &_posY, const float &_radius, const int &_minScaleLevel=-1, const int &_maxScaleLevel=-1) const;
    //project a keypoint back to 3D world if stereo/depth is available
    cv::Mat inverseProject(const int &_idx);
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####key points functions
    //check whether a pixel coordinate is in the image
    bool isInImage(const float &_posX, const float &_posY) const;
    //check whether a map point is in the frustum of the camera
    //and fill variables of the MapPoint to be used by the tracking
    bool isInCameraFrustum(std::shared_ptr<MapPoint> _sptrMapPoint, const float &_viewCosLimit);
    //compute which grid cell a keypoint is in. return false if outside the grid.
    bool computeLocationInGrid(const cv::KeyPoint &_keyPoint, int &_locX, int &_locY);
    //search a match for each keypoint in the left image to a keypoint in the right image.
    //if match succeed, compute depth and get the x coordinate of the matchted point in the right image associated in the left keypoint is stored.
    void computeStereoMatches();
    //associate a right coordinate to a keypoint if depth is valid in the depthmap
    void computeStereoFromRGBD(const cv::Mat &_depthImage);
    //vocabulary for relocalization
    std::shared_ptr<DBoW3::Vocabulary> m_sptr_vocab;
    //orb extractor
    std::shared_ptr<OrbExtractor> m_sptr_leftOrbExtractor,m_sptr_rightOrbExtractor;
    //frame time stamp
    double m_d_timeStamp;
    //calibration information
    static cv::Mat m_cvMat_intParMat;
    static cv::Mat m_cvMat_imageDistCoef;
    static cv::Mat m_cvMat_rightImageDistCoef;
    static float m_flt_fx, m_flt_fy, m_flt_cx, m_flt_cy, m_flt_invFx, m_flt_invFy;
    //stereo baseline multiplied by fx
    static float m_flt_baseLineTimesFx;
    //stereo baseline in meters
    static float m_flt_baseLine;
    //threshold to decide if a point is close or far
    //close points are inserted from 1 view
    //far points are inserted from 2 views in monocular case
    float m_flt_depthThd;
    //features(key points) number
    int m_int_keyPointsNum;
    //vector of features (key points)
    //for stereo, m_v_undistortKeyPoints is redundant as images must be rectified
    //for rgbd, rgb image can be distorted
    //but actually undistort key points are modified directly in m_v_keyPoints, so no m_v_undistortKeyPoints is required
    std::vector<cv::KeyPoint> m_v_keyPoints, m_v_rightKeyPoints;
    //corresponding stereo coordinate and depth for each key point
    //monocular key points have a negative value
    std::vector<float> m_v_rightXcords, m_v_depth;
    //bag of words vector
    DBoW3::BowVector m_bow_wordVec;
    DBoW3::FeatureVector m_bow_keyPointsVec;
    //Orb descriptor, each row is associated to a key point
    cv::Mat m_cvMat_descriptors, m_cvMat_rightDescriptors;
    //flag vector to associate outlier associations.
    std::vector<bool> m_v_isOutliers;
    //key points are associated to cells in a grid to reduce matching complecity when projecting map points
    static float m_flt_gridCellWidthInv, m_flt_gridCellHeightInv;
    static const int m_int_gridRowsNum = 48;
    static const int m_int_gridColsNum = 64;
    std::vector<std::vector<std::vector<int>>> m_vvv_grid = std::vector<std::vector<std::vector<int>>>(m_int_gridColsNum,std::vector<std::vector<int>>(m_int_gridRowsNum,std::vector<int>()));
    //reference key frame
    std::shared_ptr<KeyFrame> m_sptr_refKeyFrame;
    //scale pyramid info
    int m_int_scaleLevelsNum;
    float m_flt_scaleFactor;
    float m_flt_logScaleFactor;
    std::vector<float> m_v_scaleFactors, m_v_invScaleFactors, m_v_scaleFactorSquares, m_v_invScaleFactorSquares;
    //map points that are associated to the points, nullptr means no association
    std::vector<std::shared_ptr<MapPoint>> m_v_sptrMapPoints;
    //transform from camera to world
    cv::Mat m_cvMat_T_c2w;
    //undistorted image bounds (compute only once)
    static float m_flt_minX, m_flt_maxX, m_flt_minY, m_flt_maxY;
    static bool m_b_isComputeInit;
    long int m_int_ID=-1;
    static long int m_int_reservedID;
    protected:
    //undistort key points given OpenCV distortion parameters.
    //only for the RGB-D. Stereo must be already rectified!
    //(called in the constructor).
    void undistortKeyPoints(std::vector<cv::KeyPoint> &_v_keyPoints, const cv::Mat &_cvMat_intParMat, const cv::Mat &_cvMat_imageDistCoef, const cv::Mat &_cvMat_P = cv::Mat());
    //if Stereo is not rectified
    void rectifyStereo();
    //compute image bounds for the undistorted image (called in the constructor).
    void computeImageBounds(const cv::Mat &_imageLeft);
    //assign keypoints to the grid to speed up feature (key point) matching (called in the constructor).
    void assignKeyPointsToGrid();
    // The following variables need to be accessed trough a mutex to be thread safe.
    //transform from world to camera
    cv::Mat m_cvMat_T_w2c;
    //rotation matrix
    cv::Mat m_cvMat_R_c2w, m_cvMat_R_w2c, m_cvMat_t_c2w;
    //camera origin coordinate in world
    cv::Mat m_cvMat_origin; //m_cvMat_t_w2c
    //stereo center coordinate in world
    cv::Mat m_cvMat_center;
    static std::mutex m_mutex_ID;
    std::mutex m_mutex_pose;
    std::mutex m_mutex_connections;
    std::mutex m_mutex_keyPoints;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_FRAME_HPP
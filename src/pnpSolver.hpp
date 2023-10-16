/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_PNP_HPP
#define YDORBSLAM_PNP_HPP

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include "mapPoint.hpp"
#include "frame.hpp"

namespace YDORBSLAM{
  class PnPsolver{
    public:
    PnPsolver(const Frame &_frame, const std::vector<std::shared_ptr<MapPoint>> &_v_matchedMapPoints);
    void setRansacParameters(const float &_flt_probability = 0.99, const int &_int_minInliersNum = 8, const int &_int_maxIterNum = 300, const int &_int_minSet = 4, const float &_flt_epsilon = 0.4, const float &_flt_thd = 5.991);
    cv::Mat find(std::vector<bool> &_v_isInliers, int &_int_inliersNum);
    cv::Mat iterate(const int &_int_iterNum, bool &_b_isNoMore, std::vector<bool> &_v_isInliers, int &_int_inliersNum);
    private:
    void checkInliers();
    bool refine();
    //
    void set_maximum_number_of_correspondences(const int _int_num);
    void reset_correspondences(void);
    void add_correspondence(const float &_flt_3DX, const float &_flt_3DY, const float &_flt_3DZ, const float &_flt_2DX, const float &_flt_2DY);
    void compute_pose(cv::Mat &_rotation, cv::Mat &_translation);
    float reprojection_error(const float _rotation[3][3], const float _translation[3]);
    void choose_control_points(void);
    void compute_barycentric_coordinates(void);
    void fill_M(cv::Mat &_M, const int &_row, const std::vector<float> &_alphas, const float &_u, const float &_v);
    void compute_ccs(const float *_betas, const float *_ut);
    void compute_pcs(void);
    void solve_for_sign(void);
    void find_betas_approx_1(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas);
    void find_betas_approx_2(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas);
    void find_betas_approx_3(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float *_betas);
    void qr_solve(cv::Mat &_A, cv::Mat &_b, cv::Mat &_X);
    float dot(const float *_v1, const float *_v2);
    float dist2(const std::vector<float>& _p1, const std::vector<float>& _p2);
    void compute_rho(float *_rho);
    void compute_L_6x10(const float *_u, float *_l_6x10);
    void gauss_newton(const cv::Mat &_L_6x10, const cv::Mat &_Rho, float _current_betas[4]);
    void compute_A_and_b_gauss_newton(const float *_l_6x10, const float *_rho, float _cb[4], cv::Mat &_A, cv::Mat &_b);
    float compute_R_and_t(const float *_ut, const float *_betas, float _rotation[3][3], float _translation[3]);
    void estimate_R_and_t(float _rotation[3][3], float _translation[3]);
    float m_flt_uc, m_flt_vc, m_flt_fu, m_flt_fv;
    std::vector<float> m_v_pws, m_v_us, m_v_alphas, m_v_pcs;
    int m_int_maxCorrespondencesNum = 0;
    int m_int_correspondencesNum = 0; 
    std::vector<std::vector<float>> m_vv_cws = std::vector<std::vector<float>>(4,std::vector<float>(3));
    std::vector<std::vector<float>> m_vv_ccs = std::vector<std::vector<float>>(4,std::vector<float>(3));
    float m_flt_cws_determinant;
    std::vector<std::shared_ptr<MapPoint>> m_v_matchedMapPoints;
    //2d points
    std::vector<cv::Point2f> m_v_point2D;
    std::vector<float> m_v_scaleFactorSquares;
    //3d points
    std::vector<cv::Point3f> m_v_point3D;
    //index in frame
    std::vector<int> m_v_keyPointsIndices;
    //current estimation
    cv::Mat m_cvMat_currentRotation;
    cv::Mat m_cvMat_currentTranslation;
    cv::Mat m_cvMat_T_c2w;
    std::vector<bool> m_v_isInliers;
    int m_int_inliersNum = 0;
    //current ransac state
    int m_int_currentRansacIterNum = 0;
    std::vector<bool> m_v_isBestInliers;
    int m_int_bestInliersNum = 0;
    cv::Mat m_cvMat_bestT_c2w;
    //refined
    cv::Mat m_cvMat_refinedT_c2w;
    std::vector<bool> m_v_isRefinedInliers;
    int m_int_refinedInliersNum = 0;
    //number of correspondences
    int m_int_correspondencesNum = 0;
    //indices for random selections [0 ... N-1]
    std::vector<int> m_v_allIndices;
    //ransac probability
    float m_flt_ransacProb;
    //ransac min inliers number
    int m_int_minRansacInliersNum;
    //ransac max iteration number
    int m_int_maxRansacIterNum;
    //ransac expected inliers/total ratio
    float m_flt_ransacEpsilon;
    //ransac threshold inlier/outlier ratio. Max error e = dist(P1,T_12*P2)^2
    float m_flt_ransacThd;
    //ransac min set used at each iteration
    int m_int_minRansacSet;
    //max square error associated with scale level. Max error 3 = thd*thd*sigma(level)*sigma(level)
    std::vector<float> m_v_maxErrors;
  };

} //namespace YDORBSLAM

#endif //YDORBSLAM_PNP_HPP
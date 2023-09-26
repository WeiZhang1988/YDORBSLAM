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
#include "mapPoint.h"
#include "frame.h"

namespace YDORBSLAM{
  class PnPsolver{
    public:
    PnPsolver(const Frame &_frame, const std::vector<std::shared_ptr<MapPoint>> &_v_matchedMapPoints);
    void setRansacParameters(const double &_dbl_probability = 0.99, const int &_int_minInliersNum = 8, const int &_int_maxIterNum = 300, const int &_int_minSet = 4, const float &_flt_epsilon = 0.4, const float &_flt_thd = 5.991);
    cv::Mat find(std::vector<bool> &_v_isInliers, int &_int_inliersNum);
    cv::Mat iterate(const int &_int_iterNum, bool &_b_isNoMore, std::vector<bool> &_v_isInliers, int &_int_inliersNum);
    private:
    void checkInliers();
    bool refine();
    //
    void set_maximum_number_of_correspondences(const int _int_num);
    void reset_correspondences(void);
    void add_correspondence(const double &_dbl_3DX, const double &_dbl_3DY, const double &_dbl_3DZ, const double &_dbl_2DX, const double &_dbl_2DY);
    void compute_pose(cv::Mat &_rotation, cv::Mat &_translation);
    void relative_error(double & _rotErr, double & _transErr, const cv::Mat _rotTrue, const cv::Mat _transTrue, const cv::Mat _rotEst,  const cv::Mat _transEst);
    void print_pose(const cv::Mat &_rotation, const cv::Mat &_translation);
    double reprojection_error(const cv::Mat &_rotation, const cv::Mat &_translation);
    void choose_control_points(void);
    void compute_barycentric_coordinates(void);
    void fill_M(cv::Mat &_M, const int &_row, const std::vector<double> &_alphas, const double &_u, const double &_v);
    void compute_ccs(const std::vector<double> &_betas, const std::vector<double> &_ut);
    void compute_pcs(void);
    void solve_for_sign(void);
    void find_betas_approx_1(const cv::Mat &_L_6x10, const cv::Mat &_Rho, std::vector<double> &_betas);
    void find_betas_approx_2(const cv::Mat &_L_6x10, const cv::Mat &_Rho, std::vector<double> &_betas);
    void find_betas_approx_3(const cv::Mat &_L_6x10, const cv::Mat &_Rho, std::vector<double> &_betas);
    void qr_solve(cv::Mat & _A, cv::Mat &_b, cv::Mat &_X);
    double dot(const std::vector<double> &_v1, const std::vector<double> &_v2);
    double dist2(const std::vector<double> &_p1, const std::vector<double> &_p2);
    void compute_rho(std::vector<double> &_rho);
    void compute_L_6x10(const std::vector<double> &_ut, std::vector<double> &_l_6x10);
    void gauss_newton(const cv::Mat &_L_6x10, const cv::Mat &_Rho, std::vector<double> &_current_betas);
    void compute_A_and_b_gauss_newton(const std::vector<double> &_l_6x10, const std::vector<double> &_rho, std::vector<double> &_cb, cv::Mat &_A, cv::Mat &_b);
    double compute_R_and_t(const std::vector<double> &_ut, const std::vector<double> &_betas, cv::Mat &_rotation, cv::Mat &_translation);
    void estimate_R_and_t(cv::Mat &_rotation, cv::Mat &_translation);
    void copy_R_and_t(const cv::Mat &_rotationDst, const cv::Mat &_transDst, cv::Mat &_rotationSrc, cv::Mat &_transSrc);
    void mat_to_quat(const cv::Mat &_rotation, std::<double> _q);
    double m_dbl_uc, m_dbl_vc, m_dbl_fu, m_dbl_fv;
    std::vector<double> m_v_pws, m_v_us, m_v_alphas, m_v_pcs;
    int m_int_maximum_number_of_correspondences = 0;
    int m_int_number_of_correspondences = 0; 
    std::vector<std::vector<double>> m_vv_cws = std::vector<std::vector<int>>(4,std::vector<int>(3));
    std::vector<std::vector<double>> m_vv_ccs = std::vector<std::vector<int>>(4,std::vector<int>(3));
    double m_dbl_cws_determinant;
    std::vector<std::shared_ptr<MapPoint>> m_v_matchedMapPoints;
    //2d points
    std::vector<cv::Point2f> m_v_point2D;
    std::vector<float> m_v_scaleFactorSquares;
    //3d points
    std::vector<cv::Point3f> m_v_point3D;
    //index in frame
    std::vector<int> m_v_keyPointsIndices;
    //current estimation
    std::vector<std::vector<double>> m_v_currentRotation = std::vector<std::vector<double>>(3,std::vector<double>(3));
    std::vector<double> m_v_currentTranslation = std::vector<double>(3);
    cv::Mat m_T_c2w;
    std::vector<bool> m_v_isInliers;
    int m_int_inliersNum = 0;
    //current ransac state
    int m_int_currentRansacIterNum = 0;
    std::vector<bool> m_v_isBestInliers;
    int m_int_bestInliersNum = 0;
    cv::Mat m_bestT_c2w;
    //refined
    cv::Mat m_cvMat_refinedT_c2w;
    std::vector<bool> m_v_isRefinedInliers;
    int refinedInliersNum = 0;
    //number of correspondences
    int m_int_correspondencesNum = 0;
    //indices for random selections [0 ... N-1]
    std::vector<int> m_v_allIndices;
    //ransac probability
    double m_dbl_ransacProb;
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
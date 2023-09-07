/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_CONVERTER_HPP
#define YDORBSLAM_CONVERTER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <g2o/config.h>
#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace YDORBSLAM{
  class Converter{
    public:
    static std::vector<cv::Mat> descriptors_cvMat_cvMatVector(const cv::Mat &_descriptors);
    static g2o::SE3Quat transform_cvMat_SE3Quat(const cv::Mat &_transform);
    static cv::Mat transform_SE3_cvMat(const g2o::SE3Quat &_transform);
    static cv::Mat transform_Sim3_cvMat(const g2o::Sim3 &_transform);
    static cv::Mat d4X4Matrix_eigen_cvMat(const Eigen::Matrix<double,4,4> &_matrix);
    static cv::Mat d3X3Matrix_eigen_cvMat(const Eigen::Matrix3d &_matrix);
    static cv::Mat d3X1Matrix_eigen_cvMat(const Eigen::Matrix<double,3,1> &_matrix);
    static cv::Mat transform_eigen_cvMat(const Eigen::Matrix<double,3,3> &_rotation, const Eigen::Matrix<double,3,1> &_translation);
    static Eigen::Matrix<double,3,1> d3X1Matrix_cvMat_eigen(const cv::Mat &_matrix);
    static Eigen::Matrix<double,3,1> d3X1Point_cvPoint_eigen(const cv::Point3f &_point);
    static Eigen::Matrix<double,3,3> d3X3Matrix_cvMat_eigen(const cv::Mat &_matrix);
    static std::vector<float> rotation_cvMat_eigenQuat(const cv::Mat &_rotation);
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_CONVERTER_HPP
#include "converter.hpp"

namespace YDORBSLAM{
  std::vector<cv::Mat> Converter::descriptors_cvMat_cvMatVector(const cv::Mat &_descriptors){
    std::vector<cv::Mat> vDescriptors;
    vDescriptors.reserve(_descriptors.rows);
    for(int i=0;i<_descriptors.rows;i++){
      vDescriptors.push_back(_descriptors.row(i));
    }
    return vDescriptors;
  }
  g2o::SE3Quat Converter::transform_cvMat_SE3Quat(const cv::Mat &_transform){
    Eigen::Matrix<double,3,3> rotation;
    rotation << _transform.at<float>(0,0), _transform.at<float>(0,1), _transform.at<float>(0,2),
                _transform.at<float>(1,0), _transform.at<float>(1,1), _transform.at<float>(1,2),
                _transform.at<float>(2,0), _transform.at<float>(2,1), _transform.at<float>(2,2);
    Eigen::Matrix<double,3,1> translation(_transform.at<float>(0,3), _transform.at<float>(1,3), _transform.at<float>(2,3));
    return g2o::SE3Quat(rotation,translation);
  }
  cv::Mat Converter::transform_SE3_cvMat(const g2o::SE3Quat &_transform){
    Eigen::Matrix<double,4,4> eigenMatrix = _transform.to_homogeneous_matrix();
    return d4X4Matrix_eigen_cvMat(eigenMatrix);
  }
  cv::Mat Converter::transform_Sim3_cvMat(const g2o::Sim3 &_transform){
    Eigen::Matrix3d eigenRotation = _transform.rotation().toRotationMatrix();
    Eigen::Vector3d eigenTranslation = _transform.translation();
    double scale = _transform.scale();
    return transform_eigen_cvMat(scale*eigenRotation,eigenTranslation);
  }
  cv::Mat Converter::d4X4Matrix_eigen_cvMat(const Eigen::Matrix<double,4,4> &_matrix){
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++){
      for(int j=0; j<4; j++){
        cvMat.at<float>(i,j)=_matrix(i,j);
      }
    }
    return cvMat.clone(); 
  }
  cv::Mat Converter::d3X3Matrix_eigen_cvMat(const Eigen::Matrix3d &_matrix){
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++){
      for(int j=0; j<3; j++){
        cvMat.at<float>(i,j)=_matrix(i,j);
      }
    }
    return cvMat.clone();
  }
  cv::Mat Converter::d3X1Matrix_eigen_cvMat(const Eigen::Matrix<double,3,1> &_matrix){
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++){
      cvMat.at<float>(i)=_matrix(i);
    }
    return cvMat.clone();
  }
  cv::Mat Converter::transform_eigen_cvMat(const Eigen::Matrix<double,3,3> &_rotation, const Eigen::Matrix<double,3,1> &_translation){
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=_rotation(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=_translation(i);
    }

    return cvMat.clone();
  }
  Eigen::Matrix<double,3,1> Converter::d3X1Matrix_cvMat_eigen(const cv::Mat &_matrix){
    Eigen::Matrix<double,3,1> eigenMatrix;
    eigenMatrix << _matrix.at<float>(0), _matrix.at<float>(1), _matrix.at<float>(2);
    return eigenMatrix;
  }
  Eigen::Matrix<double,3,1> Converter::d3X1Point_cvPoint_eigen(const cv::Point3f &_point){
    Eigen::Matrix<double,3,1> eigenMatrix;
    eigenMatrix << _point.x, _point.y, _point.z;
    return eigenMatrix;
  }
  Eigen::Matrix<double,3,3> Converter::d3X3Matrix_cvMat_eigen(const cv::Mat &_matrix){
    Eigen::Matrix<double,3,3> eigenMatrix;
    eigenMatrix << _matrix.at<float>(0,0), _matrix.at<float>(0,1), _matrix.at<float>(0,2),
                   _matrix.at<float>(1,0), _matrix.at<float>(1,1), _matrix.at<float>(1,2),
                   _matrix.at<float>(2,0), _matrix.at<float>(2,1), _matrix.at<float>(2,2);
    return eigenMatrix;
  }
  std::vector<float> Converter::rotation_cvMat_eigenQuat(const cv::Mat &_rotation){
    Eigen::Matrix<double,3,3> eigenMatrix = d3X3Matrix_cvMat_eigen(_rotation);
    Eigen::Quaterniond eigenQuaternion(eigenMatrix);
    std::vector<float> vRet(4);
    vRet[0] = eigenQuaternion.x();
    vRet[1] = eigenQuaternion.y();
    vRet[2] = eigenQuaternion.z();
    vRet[3] = eigenQuaternion.w();
    return vRet;
  }
}//namespace YDORBSLAM
#include "tracking.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "optimizer.hpp"
#include "pnpSolver.hpp"

namespace YDORBSLAM{
  Tracking::Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
  std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, const int _sensor, const string &_strSettingPath) : \
  m_sptr_system(_sptrSys), m_sptr_orbVocabulary(_sptrVoc), m_sptr_frameDrawer(_sptrFrameDrawer), m_sptr_mapDrawer(_sptrMapDrawer), m_sptr_map(_sptrMap), \
  m_sptr_keyFrameDataBase(_sptrKeyFrameDatabase), m_int_sensor(_sensor){
    //load camera parameters from setting file
    cv::FileStorage settingFile(_strSettingPath, cv::FileStorage::READ);
    float fx = settingFile["Camera.fx"];
    float fy = settingFile["Camera.fy"];
    float cx = settingFile["Camera.cx"];
    float cy = settingFile["Camera.cy"];
    cv::Mat intParMat = cv::Mat::eye(3,3,CV_32F);
    intParMat.at<float>(0,0) = fx;
    intParMat.at<float>(1,1) = fy;
    intParMat.at<float>(0,2) = cx;
    intParMat.at<float>(1,2) = cy;
    intParMat.copyTo(m_cvMat)//stop here
  }

}//namespace YDORBSLAM

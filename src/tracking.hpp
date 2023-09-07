/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_TRACKING_HPP
#define YDORBSLAM_TRACKING_HPP

#include <opencv2/opencv.hpp>
#include "DBoW3.h"
#include "map.hpp"
#include "frame.hpp"
#include "keyFrameDatabase.hpp"
#include "orbExtractor.hpp"
#include "system.hpp"
#include "Viewer.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include <string>
#include <memory>
#include <mutex>

namespace YDORBSLAM{
  class Viewer;
  class FrameDrawer;
  class Map;
  class LocalMapping;
  class LoopClosing;
  class System;
  class Tracking{
    public:
    Tracking(std::shared_ptr<System> _sptrSys, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Map> _sptrMap, \
    std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase, const string &_strSettingPath, const int sensor);
    //preprocess the input and call track(), extract key points and perform stereo matching.
    cv::Mat grabImageStereo(const cv::Mat &_leftImageRect, const cv::Mat &_rightImageRect, const double &_timestamp);
  };

}//namespace YDORBSLAM

#endif //YDORBSLAM_TRACKING_HPP
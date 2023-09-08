#ifndef LOOPCLOSING_HPP
#define LOOPCLOSING_HPP

#include "keyFrame.hpp"
//#include "localMapping.hpp"
#include "map.hpp"
//#include "tracking.hpp"
#include "keyFrameDatabase.hpp"
#include "DBoW3.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/config.h>
#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <thread>
#include <mutex>
#include <map>
#include <memory>

namespace YDORBSLAM{
  class LoopClosing{
    public:
    LoopClosing();
    typedef std::map<std::shared_ptr<KeyFrame>,g2o::Sim3,std::less<std::shared_ptr<KeyFrame>>,Eigen::aligned_allocator<std::pair<const std::shared_ptr<KeyFrame>, g2o::Sim3>>> KeyFrameAndPose;
  };
}//namespace YDORBSLAM

#endif //LOOPCLOSING_HPP
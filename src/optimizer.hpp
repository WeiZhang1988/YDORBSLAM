/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_OPTIMIZER_HPP
#define YDORBSLAM_OPTIMIZER_HPP

#include <vector>
#include <memory>
#include "mapPoint.hpp"
#include "frame.hpp"
#include "keyFrame.hpp"
#include "map.hpp"
#include "loopClosing.hpp"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"


namespace YDORBSLAM{
  class LoopClosing;
  class Optimizer{
    public:
    static void bundleAdjust(const std::vector<std::shared_ptr<KeyFrame>> &_vSptrKeyFrames, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, const int _iterNum = 5, bool &_bIsStopping = *(bool*)(nullptr), const long int _loopKeyFrameID = 0, const bool _bIsRobust = true);
    static void localBundleAdjust(std::shared_ptr<KeyFrame> _sptrKeyFrame, std::shared_ptr<Map> _sptrMap, bool &_bIsStopping = *(bool*)(nullptr));
    static void globalBundleAdjust(std::shared_ptr<Map> _sptrMap, const int _iterNum = 5, bool &_bIsStopping = *(bool*)(nullptr), const long int _loopKeyFrameID = 0, const bool _bIsRobust = true);
    static int optimizePose(Frame &_frame);
    //if bIsScaleFixed is true, execute 6DoF optimization (for stereo and rgbd), otherwise 7DoF optimization (for mono)
    static void optimizeEssentialGraph(std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrame> _sptrLoopKeyFrame, std::shared_ptr<KeyFrame> _sptrCurrentKeyFrame, const LoopClosing::KeyFrameAndPose &_nonCorrectedSim3, const LoopClosing::KeyFrameAndPose &_correctedSim3, const std::map<std::shared_ptr<KeyFrame>,std::set<std::shared_ptr<KeyFrame>>> &_loopConnections, const bool &_bIsScaleFixed);
    //if bIsScaleFixed is true, optimize SE3 (for stereo and rgbd), otherwise Sim3 (for mono)
    static int optimizeSim3(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrFirstKeyFrameMatchedMapPoints, g2o::Sim3 &_g2oSim3_first2second, const float _thd, const bool _bIsScaleFixed);
  };
}//namespace YDORBSLAM

#endif //YDORBSLAM_OPTIMIZER_HPP
/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_MAPDRAWER_HPP
#define YDORBSLAM_MAPDRAWER_HPP

#include "map.hpp"
#include "mapPoint.hpp"
#include "keyFrame.hpp"
#include <memory>
#include <pangolin/pangolin.h>
#include <mutex>

namespace YDORBSLAM{
class MapDrawer{
    public:
    MapDrawer(std::shared_ptr<Map> _sptrMap, const std::string& _strConfigurationPath);
    void drawMapPoints();
    void drawKeyFrames(const bool& _bdrawKF, const bool& _bdrawGraph);
    void drawCurrentCamera(pangolin::OpenGlMatrix& _glTw2c);
    void setCurrentCameraPose(const cv::Mat& _cvMatTc2w);
    void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& _glCameraPose);
    std::shared_ptr<Map> m_sptr_map;
    private:
    float m_flt_keyFrameSize;
    float m_flt_keyFrameLineWidth;
    float m_flt_graphLineWidth;
    float m_flt_pointSize;
    float m_flt_cameraSize;
    float m_flt_cameraLineWidth;
    cv::Mat m_cvMat_cameraPoseTc2w;
    std::mutex m_mutex_camera;
};
} //namespace YDORBSLAM

#endif // YDORBSLAM_MAPDRAWER_H
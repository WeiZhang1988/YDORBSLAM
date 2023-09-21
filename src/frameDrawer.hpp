/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_FRAMEDRAWER_HPP
#define YDORBSLAM_FRAMEDRAWER_HPP

#include "tracking.hpp"
#include "mapPoint.hpp"
#include "map.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <mutex>

namespace YDORBSLAM{
class Tracking;
class viewer;
class FrameDrawer{
    public:
    FrameDrawer(std::shared_ptr<Map> _sptrMap);
    // Update info from the last processed frame.
    void update(std::shared_ptr<Tracking> _sptrTracker);
    // Draw last processed frame.
    cv::Mat drawFrame();
    protected:
    void drawTextInfo(cv::Mat &_image, int _nState, cv::Mat &_imageText);
    // Info of the frame to be drawn
    cv::Mat m_cvMat_image;
    int m_int_CurrentKeyPointNUM;
    vector<cv::KeyPoint> m_v_cvMatCurrentKeyPoints;
    vector<bool> m_v_bOwnObservation, m_v_bNonObservation;
    bool m_b_onlyLocalization;
    int m_int_ownObservationMPNums, m_int_nonObservationMPNums;
    int m_int_state;
    std::shared_ptr<Map> m_sptrMap;
    std::mutex m_copyMutex;
};
} //namespace YDORBSLAM

#endif //YDORBSLAM_FRAMEDRAWER_HPP
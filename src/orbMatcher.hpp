/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_ORBMATCHER_HPP
#define YDORBSLAM_ORBMATCHER_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "mapPoint.hpp"
#include "keyFrame.hpp"
#include "frame.hpp"

namespace YDORBSLAM{
  class MapPoint;
  class Frame;
  class KeyFrame;
  class OrbMatcher{
    public:
    OrbMatcher(float _bestSecondRatio=0.6, bool _isToCheckOrientation=true) : m_flt_bestSecondRatio(_bestSecondRatio), m_b_isToCheckOrientation(_isToCheckOrientation){}
    //compute hamming distance between two ORB descriptors
    static int computeDescriptorsDistance(const cv::Mat &_firstDescriptor, const cv::Mat &_secondDescriptor);
    //search matches between frame key points and projected map points.
    //return number of matches
    //used to track the local map in (Trakcing)
    int searchByProjectionInFrameAndMapPoint(Frame &_frame, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, const float _thd=3);
    //project map points tracked in last frame into the current frame and search matches
    //used to track from previous frame in (Tracking)
    int searchByProjectionInLastAndCurrentFrame(Frame &_currentFrame, Frame &_lastFrame, const float _thd=3);
    //project map points seen in key frame into the current frame and search matches
    //used by relocalization in (Tracking)
    int searchByProjectionInKeyFrameAndCurrentFrame(Frame &_currentFrame, std::shared_ptr<KeyFrame> _sptrKeyFrame, const std::set<std::shared_ptr<MapPoint>> &_setFoundMapPoints, const float _thd, const int _orbDist);
    //project map points using similarity transform and search matches
    //used by loop detection in (Loop Closing)
    int searchByProjectionInSim(std::shared_ptr<KeyFrame> _sptrKeyFrame, cv::Mat &_sim_c2w, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const int _thd);
    //search matches between map points in a key frame and orb in a frame
    //brutal force constrained to orb that belongs to the same vocabulary node at a certain level
    //used by relocalization and loop detection in (Tracking and Loop Closing)
    int searchByBowInKeyFrameAndFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame, Frame &_frame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints);
    int searchByBowInTwoKeyFrames(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints);
    //match to triangulate new map point
    //check epipolar constraint
    int searchForTriangulation(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, cv::Mat &_fMatrix_first2second, std::vector<std::pair<int,int>> &_vMatchedPairs, const bool _bIsStereoOnly);
    //search matches map points seen in first key frame and second key frame transforming by Sim3 [s12*R12|t12]
    //in stereo and rgb-d case, s12 = 1
    int searchBySim3(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, const float &_scale_first2second, const cv::Mat &_rotation_first2second, const cv::Mat &_translation_first2second, const float _thd);
    //project map points into key frame and search for duplicated map points
    int fuseByProjection(std::shared_ptr<KeyFrame> _sptrKeyFrame, const std::vector<std::shared_ptr<MapPoint>> _vSptrMapPoints, const float _thd=3.0);
    //project map points into key frame using a given Sim3 and search for duplicated map points
    int fuseBySim3(std::shared_ptr<KeyFrame> _sptrKeyFrame, cv::Mat &_sim_c2w, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, float _thd, std::vector<std::shared_ptr<MapPoint>> &_vSptrReplacement);
    static const int m_int_highThd;
    static const int m_int_lowThd;
    static const int m_int_histLen;
    protected:
    bool isEpipolarLineDistCorrect(const cv::KeyPoint &_firstKeyPoint, const cv::KeyPoint &_secondKeyPoint, const cv::Mat &_fMatrix_first2second, const std::shared_ptr<KeyFrame> _sptrKeyFrame);
    float getRadiusByViewCos(const float &_viewCos);
    void computeThreeMaxima(std::vector<std::vector<int>> &_hist, const int &_len, int &_idx1, int &_idx2, int &_idx3);
    float m_flt_bestSecondRatio;
    bool m_b_isToCheckOrientation;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_ORBMATCHER_HPP
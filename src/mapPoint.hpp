/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_MAPPOINT_HPP
#define YDORBSLAM_MAPPOINT_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include "frame.hpp"
#include "keyFrame.hpp"
#include "map.hpp"

namespace YDORBSLAM{
  class KeyFrame;
  class Map;
  class Frame;

  class MapPoint : public std::enable_shared_from_this<MapPoint>{
    public:
    MapPoint(const cv::Mat &_posInWorld, std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrame> _sptrRefKeyFrame);
    MapPoint(const cv::Mat &_posInWorld, std::shared_ptr<Map> _sptrMap, Frame &_frame, const int &_idxInKeyPoints);
    void setPosInWorld(const cv::Mat &_posInWorld);
    void setBadFlag();
    inline cv::Mat getPosInWorld(){
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      return m_cvMat_posInWorld.clone();
    }
    inline cv::Mat getNormal(){
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      return m_cvMat_normalVector.clone();
    }
    inline std::shared_ptr<KeyFrame> getReferenceKeyFrame(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_sptr_refKeyFrame;
    }
    inline std::map<std::shared_ptr<KeyFrame>,int> getObservations(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_dic_observations;
    }
    inline int getObservationsNum(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_int_observationsNum;
    }
    inline std::shared_ptr<MapPoint> getReplacement(){
      std::unique_lock<std::mutex> lock1(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_sptr_replacement;
    }
    inline int getFoundNum(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_int_foundNum;
    }
    inline float getFoundRatio(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return static_cast<float>(m_int_foundNum)/static_cast<float>(m_int_visibleNum);
    }
    inline cv::Mat getDescriptor(){
      std::unique_lock<std::mutex> lock(m_mutex_descriptor);
      return m_cvMat_descriptor.clone();
    }
    inline float getMinDistanceInvariance(){
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      return 0.8f*m_flt_minDistance;
    }
    inline float getMaxDistanceInvariance(){
      std::unique_lock<std::mutex> lock(m_mutex_PosDistNorm);
      return 1.2f*m_flt_maxDistance;
    }
    int getIdxInKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    bool isInKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    inline bool isBad(){
      std::unique_lock<std::mutex> lock(m_mutex_badObsRefKeyFrmReplVsbFnd);
      return m_b_bad;
    }
    void addObservation(std::shared_ptr<KeyFrame> _sptrKeyFrame,const int &_idx);
    void eraseObservation(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void beReplacedBy(std::shared_ptr<MapPoint> _sptrMapPoint);
    void increaseVisible(const int &_num = 1);
    void increaseFound(const int &_num = 1);
    void computeDistinctiveDescriptors();
    void updateNormalAndDepth();
    int predictScaleLevel(const float &_currentDist, std::shared_ptr<KeyFrame> _sptrKeyFrame);
    int predictScaleLevel(const float &_currentDist, const Frame &_frame);
    //variables used by the tracking
    float m_flt_trackProjX = 0.0;
    float m_flt_trackProjY = 0.0;
    float m_flt_trackProjRightX = 0.0;
    float m_flt_trackProjRightY = 0.0;
    bool m_b_isTrackInView = false;
    int m_int_trackScaleLevel = 1;
    float m_flt_trackViewCos = 1;
    long int m_int_trackRefForFrameID=0;
    long int m_int_lastSeenInFrameID=0;
    //variables used by local mapping
    long int m_int_localBAForKeyFrameID=0;
    long int m_int_fuseCandidateForKeyFrameID=0;
    //variables used by loop closing
    long int m_int_loopPointForKeyFrameID=0;
    long int m_int_correctedByKeyFrameID=0;
    long int m_int_correctedRefKeyFrameID=0;
    long int m_int_globalBAforKeyFrameID=0;
    cv::Mat m_cvMat_posGlobalBA;
    static std::mutex m_mutex_global;
    long int m_int_ID=-1;
    protected:
    //bad flag
    bool m_b_bad=false;
    std::shared_ptr<MapPoint> m_sptr_replacement = std::shared_ptr<MapPoint>(nullptr);
    //number of observations, basically by key frame
    int m_int_observationsNum=0;
    //tracking number of being found
    int m_int_foundNum=1;
    //tracking number of being visible
    int m_int_visibleNum=1;
    //scale invariance distance
    float m_flt_minDistance=0;
    float m_flt_maxDistance=0;
    //the best descriptor to fast matching
    cv::Mat m_cvMat_descriptor;
    //keyframes that observe the point;
    //the index in each KeyFrame's vector which maintains all its observed points.
    std::map<std::shared_ptr<KeyFrame>,int> m_dic_observations;
    std::shared_ptr<KeyFrame> m_sptr_refKeyFrame = std::shared_ptr<KeyFrame>(nullptr);
    //position in world coordinates
    cv::Mat m_cvMat_posInWorld;
    //mean viewing direction, basically the direction from camera origin (optical center of camera) to the map point.
    cv::Mat m_cvMat_normalVector;
    long int m_int_firstKeyFrameID=-1;
    long int m_int_firstFrameID=-1;
    static long int m_int_reservedID;
    std::shared_ptr<Map> m_sptr_map = std::shared_ptr<Map>(nullptr);
    static std::mutex m_mutex_ID;
    std::mutex m_mutex_badObsRefKeyFrmReplVsbFnd;
    std::mutex m_mutex_PosDistNorm; 
    std::mutex m_mutex_descriptor; 
  };

}//namespace YDORBSLAM

#endif//YDORBSLAM_MAPPOINT_HPP
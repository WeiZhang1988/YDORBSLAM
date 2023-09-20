/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_KEYFRAME_HPP
#define YDORBSLAM_KEYFRAME_HPP

#include "DBoW3/DBoW3.h"
#include "orbExtractor.hpp"
#include "mapPoint.hpp"
#include "frame.hpp"
#include "keyFrameDatabase.hpp"

#include <mutex>
#include <memory>

namespace YDORBSLAM{
  class Map;
  class MapPoint;
  class Frame;
  class KeyFrameDatabase;
  class KeyFrame : public Frame, public std::enable_shared_from_this<KeyFrame>{
    public:
    //constructor by pass reference of Frame
    KeyFrame(const Frame &_frame, std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrameDatabase> _sptrKeyFrameDatabase);
    //covisibility graph functions#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void addConnection(std::shared_ptr<KeyFrame> _sptrKeyFrame, const int &_weight);
    void eraseConnection(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void updateConnections();
    void orderConnectionsByWeight();
    std::set<std::shared_ptr<KeyFrame>> getConnectedKeyFrames();
    std::vector<std::shared_ptr<KeyFrame>> getOrderedConnectedKeyFrames();
    std::vector<std::shared_ptr<KeyFrame>> getFirstNumOrderedConnectedKeyFrames(const int &_amount);
    std::vector<std::shared_ptr<KeyFrame>> getOrderedConnectedKeyFramesLargerThanWeight(const int &_weight);
    int getWeight(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####covisibility graph functions
    //spanning tree functions#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void addChild(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void eraseChild(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void changeParent(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    std::set<std::shared_ptr<KeyFrame>> getChildren();
    std::shared_ptr<KeyFrame> getParent();
    bool hasChild(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####spanning tree functions
    //map point observation functions#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void addMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint, const int &_idx);
    void eraseMatchedMapPoint(const int &_idx);
    void eraseMatchedMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint);
    void replaceMapPointMatch(std::shared_ptr<MapPoint> _sptrMapPoint, const int &_idx);
    std::set<std::shared_ptr<MapPoint>> getMatchedMapPointsSet();
    std::vector<std::shared_ptr<MapPoint>> getMatchedMapPointsVec();
    int trackedMapPointsNum(const int &_minObservationNum);
    std::shared_ptr<MapPoint> getMapPoint(const int &_idx);
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####map point observation functions
    //Enable/Disable bad flag changes#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void setEraseExemption();
    void cancelEraseExemption();
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####Enable/Disable bad flag changes
    //loop edges#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void addLoopEdge(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    std::set<std::shared_ptr<KeyFrame>> getLoopEdges();
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####loop edges
    //Set/check bad flag#####-#####-#####-#####-#####-#####-#####-#####-#####-#####
    void setBadFlag();
    bool isBad();
    //#####-#####-#####-#####-#####-#####-#####-#####-#####-#####Set/check bad flag
    //variables used by tracking
    long int m_int_trackRefForFrameID=0;
    long int m_int_fuseTargetForKeyFrameID=0;
    //variables used by local mapping
    long int m_int_localBAForKeyFrameID=0;
    long int m_int_fixedBAForKeyFrameID=0;
    //variables used by key frame database
    long int m_int_loopQueryID=0;
    int m_int_loopWordsNum=0;
    float m_flt_loopScore=0.0;
    long int m_int_relocalizationQueryID=0;
    int m_int_relocalizationWordsNum=0;
    float m_flt_relocalizationScore=0.0;
    //variables used by loop closing
    cv::Mat m_cvMat_T_c2w_GBA;
    cv::Mat m_cvMat_T_c2w_beforeGBA;
    long int m_int_globalBAForKeyFrameID=0;
    //pose relative to parent, represented by tranformation from current camera to parent camera, computed when bad flag is activated
    cv::Mat m_cvMat_T_c2p;
    long int m_int_keyFrameID=-1;
    static long int m_int_reservedKeyFrameID;
    //the following variables need to be accessed trough a mutex to be thread safe.
    protected:
    std::shared_ptr<KeyFrameDatabase> m_sptr_keyFrameDatabase;
    std::map<std::shared_ptr<KeyFrame>,int> m_dic_connectedKeyFrameWeights;
    std::vector<std::shared_ptr<KeyFrame>> m_v_orderedConnectedKeyFrames;
    std::vector<int> m_v_orderedWeights;
    //spanning tree and loop edges
    bool m_b_isFirstConnection = true;
    std::shared_ptr<KeyFrame> m_sptr_parent;
    std::set<std::shared_ptr<KeyFrame>> m_set_sptrChildren;
    std::set<std::shared_ptr<KeyFrame>> m_set_sptrLoopEdges;
    //bad flags
    bool m_b_isEraseExempted  = false;
    bool m_b_isEraseRequested = false;
    bool m_b_isBad = false;
    std::shared_ptr<Map> m_sptr_map;
    static std::mutex m_mutex_keyFrameID;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_KEYFRAME_HPP
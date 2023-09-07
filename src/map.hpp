/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_MAP_HPP
#define YDORBSLAM_MAP_HPP

#include "mapPoint.hpp"
#include "keyFrame.hpp"
#include <memory>
#include <set>
#include <mutex>

namespace YDORBSLAM{
  class MapPoint;
  class KeyFrame;
  class Map{
    public:
    Map() {}
    void addKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void addMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint);
    void eraseKeyFrame(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void eraseMapPoint(std::shared_ptr<MapPoint> _sptrMapPoint);
    void setReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints);
    void informNewBigChange();
    int getLastBigChangeIdx();
    std::vector<std::shared_ptr<KeyFrame>> getAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> getAllMapPoints();
    std::vector<std::shared_ptr<MapPoint>> getReferenceMapPoints();
    long int getKeyFramesNum();
    long int getMapPointsNum();
    long int getMaxKeyFrameID();
    void clearAll();
    std::vector<std::shared_ptr<KeyFrame>> m_v_sptrOriginalKeyFrames;
    std::mutex m_mutex_updateMap;
    protected:
    std::set<std::shared_ptr<KeyFrame>> m_set_sptrKeyFrames;
    std::set<std::shared_ptr<MapPoint>> m_set_sptrMapPoints;
    std::vector<std::shared_ptr<MapPoint>> m_v_sptrReferenceMapPoints;
    long int m_int_maxKeyFrameID = 0;
    //index related ot a big change in the map (i.e. loop closure and global BA)
    int m_int_bigChangeIdx = 0;
    std::mutex m_mutex_map;
  };

}//namespace YDORBSLAM

#endif //YDORBSLAM_MAP_HPP
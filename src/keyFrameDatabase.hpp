/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_KEYFRAMEDATABASE_HPP
#define YDORBSLAM_KEYFRAMEDATABASE_HPP

#include <vector>
#include <list>
#include <set>
#include <mutex>
#include "keyFrame.hpp"
#include "frame.hpp"
#include "DBoW3.h"

namespace YDORBSLAM{
  class Frame;
  class KeyFrame;
  class KeyFrameDatabase : public DBoW3::Vocabulary{
    public:
    KeyFrameDatabase(const DBoW3::Vocabulary &_vocab) : DBoW3::Vocabulary(_vocab) {m_v_invertedFile.resize(_vocab.size());}
    void add(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void erase(std::shared_ptr<KeyFrame> _sptrKeyFrame);
    void clear();
    //loop detection
    std::vector<std::shared_ptr<KeyFrame>> detectLoopCandidates(std::shared_ptr<KeyFrame> _sptrKeyFrame, const float &_minScore);
    //relocalization
    std::vector<std::shared_ptr<KeyFrame>> detectRelocalizationCandidates(Frame &_frame);
    protected:
    //inverted file
    std::vector<std::list<std::shared_ptr<KeyFrame>>> m_v_invertedFile;
    //mutex
    std::mutex m_mutex;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_KEYFRAMEDATABASE_HPP
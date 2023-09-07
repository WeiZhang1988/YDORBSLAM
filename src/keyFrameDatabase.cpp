#include "keyFrameDatabase.hpp"

namespace YDORBSLAM{
  void KeyFrameDatabase::add(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    std::unique_lock<std::mutex> lock(m_mutex);
    for(const auto &bow : _sptrKeyFrame->m_bow_wordVec){
      m_v_invertedFile[bow.first].push_back(_sptrKeyFrame);
    }
  }
  void KeyFrameDatabase::erase(std::shared_ptr<KeyFrame> _sptrKeyFrame){
    for(const auto &bow : _sptrKeyFrame->m_bow_wordVec){
      for(std::list<std::shared_ptr<KeyFrame>>::iterator lit=m_v_invertedFile[bow.first].begin();lit!=m_v_invertedFile[bow.first].end();lit++)
      {
          if(_sptrKeyFrame==*lit)
          {
              m_v_invertedFile[bow.first].erase(lit);
              break;
          }
      }
    }
  }
  void KeyFrameDatabase::clear(){
    m_v_invertedFile.clear();
    m_v_invertedFile.resize(size());
  }
  std::vector<std::shared_ptr<KeyFrame>> KeyFrameDatabase::detectLoopCandidates(std::shared_ptr<KeyFrame> _sptrKeyFrame, const float &_minScore){
    std::set<std::shared_ptr<KeyFrame>> setSptrConnectedKeyFrames = _sptrKeyFrame->getConnectedKeyFrames();
    std::list<std::shared_ptr<KeyFrame>> listWordsSharingKeyFrames;
    //search for all key frames that share a word with current key frame
    //discard key frames connected to the query key frame
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      for(const auto &bow : _sptrKeyFrame->m_bow_wordVec){
        for(std::shared_ptr<KeyFrame> &sptrKeyFrame : m_v_invertedFile[bow.first]){
          if(sptrKeyFrame->m_int_loopQueryID!=_sptrKeyFrame->m_int_keyFrameID){
            sptrKeyFrame->m_int_loopWordsNum=0;
            if(!setSptrConnectedKeyFrames.count(sptrKeyFrame)){
              sptrKeyFrame->m_int_loopQueryID=_sptrKeyFrame->m_int_keyFrameID;
              listWordsSharingKeyFrames.push_back(sptrKeyFrame);
            }
          }
          sptrKeyFrame->m_int_loopWordsNum++;
        }
      }
    }
    if(listWordsSharingKeyFrames.empty()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }
    //only compare against key frames that share enough words
    int maxCommonWordsNum = 0;
    for(std::shared_ptr<KeyFrame> &sptrKeyFrame : listWordsSharingKeyFrames){
      if(sptrKeyFrame->m_int_loopWordsNum>maxCommonWordsNum){
        maxCommonWordsNum = sptrKeyFrame->m_int_loopWordsNum;
      }
    }
    int minCommonWordsNum = maxCommonWordsNum*0.8;
    //compute similarity score. retain the matches whose score is higher than minScore
    std::list<std::pair<float,std::shared_ptr<KeyFrame>>> listScoreAndMatchPairs;
    for(std::shared_ptr<KeyFrame> &sptrKeyFrame : listWordsSharingKeyFrames){
      if(sptrKeyFrame->m_int_loopWordsNum>minCommonWordsNum){
        float similarity = score(_sptrKeyFrame->m_bow_wordVec,sptrKeyFrame->m_bow_wordVec);
        sptrKeyFrame->m_flt_loopScore = similarity;
        if(similarity>_minScore){
          listScoreAndMatchPairs.push_back(std::make_pair(similarity,sptrKeyFrame));
        }
      }
    }
    if(listScoreAndMatchPairs.empty()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }
    //accumulate score by covisibility
    std::list<std::pair<float,std::shared_ptr<KeyFrame>>> listAccumulatedScoreAndMatchPairs;
    float bestAccumulatedScore = _minScore;
    for(std::pair<float,std::shared_ptr<KeyFrame>> &scoreAndMatch : listScoreAndMatchPairs){
      float bestScore = scoreAndMatch.first;
      float accumulatedScore = scoreAndMatch.first;
      std::shared_ptr<KeyFrame> sptrBestKeyFrame = scoreAndMatch.second;
      for(std::shared_ptr<KeyFrame> &sptrKeyFrame : scoreAndMatch.second->getFirstNumOrderedConnectedKeyFrames(10)){
        if(sptrKeyFrame->m_int_loopQueryID==_sptrKeyFrame->m_int_keyFrameID && sptrKeyFrame->m_int_loopWordsNum>minCommonWordsNum){
          accumulatedScore+=sptrKeyFrame->m_flt_loopScore;
          if(sptrKeyFrame->m_flt_loopScore>bestScore){
            sptrBestKeyFrame = sptrKeyFrame;
            bestScore = sptrKeyFrame->m_flt_loopScore;
          }
        }
      }
      listAccumulatedScoreAndMatchPairs.push_back(std::make_pair(accumulatedScore,sptrBestKeyFrame));
      if(accumulatedScore>bestAccumulatedScore){
        bestAccumulatedScore = accumulatedScore;
      }
    }
    //return all key frames whose score is higher than 0.75*bestAccumulatedScore
    std::set<std::shared_ptr<KeyFrame>> setSptrExistedKeyFrames;
    std::vector<std::shared_ptr<KeyFrame>> vSptrLoopCandidates;
    vSptrLoopCandidates.reserve(listAccumulatedScoreAndMatchPairs.size());
    for(std::pair<float,std::shared_ptr<KeyFrame>> &scoreAndMatch : listAccumulatedScoreAndMatchPairs){
      if(scoreAndMatch.first>0.75*bestAccumulatedScore){
        if(!setSptrExistedKeyFrames.count(scoreAndMatch.second)){
          vSptrLoopCandidates.push_back(scoreAndMatch.second);
          setSptrExistedKeyFrames.insert(scoreAndMatch.second);
        }
      }
    }
    return vSptrLoopCandidates;
  }
  std::vector<std::shared_ptr<KeyFrame>> KeyFrameDatabase::detectRelocalizationCandidates(Frame &_frame){
    std::list<std::shared_ptr<KeyFrame>> listWordsSharingKeyFrames;
    //search for all key frames that share a word with current frame
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      for(const auto &bow : _frame.m_bow_wordVec){
        for(std::shared_ptr<KeyFrame> &sptrKeyFrame : m_v_invertedFile[bow.first]){
          if(sptrKeyFrame->m_int_relocalizationQueryID!=_frame.m_int_ID){
            sptrKeyFrame->m_int_relocalizationWordsNum = 0;
            sptrKeyFrame->m_int_relocalizationQueryID = _frame.m_int_ID;
            listWordsSharingKeyFrames.push_back(sptrKeyFrame);
          }
          sptrKeyFrame->m_int_relocalizationWordsNum++;
        }
      }
    }
    if(listWordsSharingKeyFrames.empty()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }
    //only compare against key frames that share enough words
    int maxCommonWordsNum = 0;
    for(std::shared_ptr<KeyFrame> &sptrKeyFrame : listWordsSharingKeyFrames){
      if(sptrKeyFrame->m_int_relocalizationWordsNum>maxCommonWordsNum){
        maxCommonWordsNum=sptrKeyFrame->m_int_relocalizationWordsNum;
      }
    }
    int minCommonWordsNum = maxCommonWordsNum*0.8;
    //compute similarity score
    std::list<std::pair<float,std::shared_ptr<KeyFrame>>> listScoreAndMatchPairs;
    for(std::shared_ptr<KeyFrame> &sptrKeyFrame : listWordsSharingKeyFrames){
      if(sptrKeyFrame->m_int_relocalizationWordsNum>minCommonWordsNum){
        float similarity = score(_frame.m_bow_wordVec,sptrKeyFrame->m_bow_wordVec);
        sptrKeyFrame->m_flt_relocalizationScore = similarity;
        listScoreAndMatchPairs.push_back(std::make_pair(similarity,sptrKeyFrame));
      }
    }
    if(listScoreAndMatchPairs.empty()){
      return std::vector<std::shared_ptr<KeyFrame>>();
    }
    //accumulate score by covisibility
    std::list<std::pair<float,std::shared_ptr<KeyFrame>>> listAccumulatedScoreAndMatchPairs;
    float bestAccumulatedScore = 0.0;
    for(std::pair<float,std::shared_ptr<KeyFrame>> &scoreAndMatch : listScoreAndMatchPairs){
      float bestScore = scoreAndMatch.first;
      float accumulatedScore = scoreAndMatch.first;
      std::shared_ptr<KeyFrame> sptrBestKeyFrame = scoreAndMatch.second;
      for(std::shared_ptr<KeyFrame> &sptrKeyFrame : scoreAndMatch.second->getFirstNumOrderedConnectedKeyFrames(10)){
        if(sptrKeyFrame->m_int_relocalizationQueryID!=_frame.m_int_ID){
          continue;
        }
        accumulatedScore+=sptrKeyFrame->m_flt_relocalizationScore;
        if(sptrKeyFrame->m_flt_relocalizationScore>bestScore){
          sptrBestKeyFrame = sptrKeyFrame;
          bestScore = sptrKeyFrame->m_flt_relocalizationScore;
        }
      }
      listAccumulatedScoreAndMatchPairs.push_back(std::make_pair(accumulatedScore,sptrBestKeyFrame));
      if(accumulatedScore>bestAccumulatedScore){
        bestAccumulatedScore = accumulatedScore;
      }
    }
    //return all key frames whose score is higher than 0.75*bestAccumulatedScore
    std::set<std::shared_ptr<KeyFrame>> setSptrExistedKeyFrames;
    std::vector<std::shared_ptr<KeyFrame>> vSptrRelocalizationCandidates;
    vSptrRelocalizationCandidates.reserve(listAccumulatedScoreAndMatchPairs.size());
    for(std::pair<float,std::shared_ptr<KeyFrame>> &scoreAndMatch : listAccumulatedScoreAndMatchPairs){
      if(scoreAndMatch.first>0.75*bestAccumulatedScore){
        if(!setSptrExistedKeyFrames.count(scoreAndMatch.second)){
          vSptrRelocalizationCandidates.push_back(scoreAndMatch.second);
          setSptrExistedKeyFrames.insert(scoreAndMatch.second);
        }
      }
    }
    return vSptrRelocalizationCandidates;
  }
}//namespace YDORBSLAM
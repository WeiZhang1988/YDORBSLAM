#include "loopClosing.hpp"
#include "sim3Solver.hpp"
#include "converter.hpp"
#include "optimizer.hpp"
#include "orbMatcher.hpp"

namespace YDORBSLAM{
  LoopClosing::LoopClosing(std::shared_ptr<Map> _sptr_map, std::shared_ptr<KeyFrameDatabase> _sptr_keyFrameDB, std::shared_ptr<DBoW3::Vocabulary> _sptr_voc, const bool &_b_isScaleFixed) : 
  m_sptr_map(_sptr_map),m_sptr_keyFrameDB(_sptr_keyFrameDB),m_sptr_voc(_sptr_voc),m_b_isScaleFixed(_b_isScaleFixed){}
  void LoopClosing::run(){
    m_b_isFinished = false;
    while(true){
      //check if there are key frames in the queue
      //detect loop candidates and check connection consistency
      //compute similarity transformation [sR|t] in the stereo/RGBD case s = 1
      if(checkNewKeyFrames() && detectLoop() && computeSim3()){
        //perform loop fusion and pos graph optimization
        correctLoop();
      }
      resetIfRequested();
      if(isFinished()){
        break;
      }
      usleep(5000);
    }
    setFinish();
  }
  void LoopClosing::insertKeyFrame(std::shared_ptr<KeyFrame> _sptr_keyFrame){
    std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
    if(_sptr_keyFrame->m_int_keyFrameID!=0){
      m_list_keyFrameBufferQueue.push_back(_sptr_keyFrame);
    }
  }
  bool LoopClosing::detectLoop(){
    {
      std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
      m_sptr_currentLoopKeyFrame = m_list_keyFrameBufferQueue.front();
      m_list_keyFrameBufferQueue.pop_front();
      //avoid key frame being erased while bing processed by this thread
      m_sptr_currentLoopKeyFrame->setEraseExemption();
    }
    //if the map contains less than 10 key frames or less than 10 key frames have passed from last loop detection
    if(m_sptr_currentLoopKeyFrame->m_int_keyFrameID < m_int_lastLoopKeyFrameID + 10){
      m_sptr_keyFrameDB->add(m_sptr_currentLoopKeyFrame);
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
    //compute reference BoW similarity score
    //this is the lowest score to a connected key frame in the connected graph
    //we will impose loop candidates to have a higher similarity than this
    float minScore = 1.0f;
    for(const std::shared_ptr<KeyFrame>& connectedKeyFrame : m_sptr_currentLoopKeyFrame->getOrderedConnectedKeyFrames()){
      if(connectedKeyFrame && !connectedKeyFrame->isBad()){
        float score = m_sptr_voc->score(m_sptr_currentLoopKeyFrame->m_bow_wordVec, connectedKeyFrame->m_bow_wordVec);
        if(score < minScore){
          minScore = score;
        }
      }
    }
    //query the database imposing the minimum score
    std::vector<std::shared_ptr<KeyFrame>> candidateKeyFrames = m_sptr_keyFrameDB->detectLoopCandidates(m_sptr_currentLoopKeyFrame, minScore);
    //if there is no loop candidate, just add new key frame and return false
    if(candidateKeyFrames.empty()){
      m_sptr_keyFrameDB->add(m_sptr_currentLoopKeyFrame);
      m_v_lastConsistentGroups.clear();
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
    //for each loop candidate check consistency with previous loop candidates
    //each candidate expands a connection group (key frames connected to the loop candidate in the connection graph )
    //a group is consistent with previous group is they share at least a key frame
    //to accept it detection of a consistent loop in servaral consectutive key frames is neccessary
    m_v_consistentCandidates.clear();
    std::vector<KeyFrameAndNum> currentConsistentGroups;
    std::vector<bool> isLastGroupConsistent(m_v_lastConsistentGroups.size(),false);
    for(std::shared_ptr<KeyFrame>& candidateKeyFrame : candidateKeyFrames){
      std::set<std::shared_ptr<KeyFrame>> setCandidateGroup = candidateKeyFrame->getConnectedKeyFrames();
      setCandidateGroup.insert(candidateKeyFrame);
      bool isConsistencyEnough = false;
      bool isConsistentForSomeGroup = false;
      for(int i=0;i<m_v_lastConsistentGroups.size();i++){
        for(const std::shared_ptr<KeyFrame>& candidateKeyFrameInGroup : setCandidateGroup){
          if(m_v_lastConsistentGroups[i].first.count(candidateKeyFrameInGroup)){
            isConsistentForSomeGroup = true;
            if(!isLastGroupConsistent[i]){
              KeyFrameAndNum consistentGroup = std::make_pair(setCandidateGroup, m_v_lastConsistentGroups[i].second+1);
              currentConsistentGroups.push_back(consistentGroup);
              isLastGroupConsistent[i] = true;//this avoid to include the same group more than once
            }
            if(m_v_lastConsistentGroups[i].second+1>=m_int_connectionConsistencyThd && !isConsistencyEnough){
              m_v_consistentCandidates.push_back(candidateKeyFrame);
              isConsistencyEnough = true;//this avoid to include the same group more than once
            }
            break;
          }
        }
      }
      //if the group is not consistent with any previous group insert with consistency counter set to zero
      if(!isConsistentForSomeGroup){
        KeyFrameAndNum consistentGroup = std::make_pair(setCandidateGroup, 0);
        currentConsistentGroups.push_back(consistentGroup);
      }
    }
    //update connection consistent groups
    m_v_lastConsistentGroups = currentConsistentGroups;
    //add current key frame to database
    m_sptr_keyFrameDB->add(m_sptr_currentLoopKeyFrame);
    if(m_v_consistentCandidates.empty()){
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }else{
      return true;
    }
  }
  bool LoopClosing::computeSim3(){
    //for each consistent loop candidate, try to compute a sim3
    //first compute ORB matches for each candidate
    //if matches are enough, setup a sim3 solver
    OrbMatcher matcher(0.75,true);
    std::vector<std::shared_ptr<Sim3Solver>> vSptrSim3Solvers;
    vSptrSim3Solvers.resize(m_v_consistentCandidates.size());
    std::vector<std::vector<std::shared_ptr<MapPoint>>> vvSptrMatchedMapPoints;
    vvSptrMatchedMapPoints.resize(m_v_consistentCandidates.size());
    std::vector<bool> vIsDiscardeds;
    vIsDiscardeds.resize(m_v_consistentCandidates.size());
    int validCandidatesNum = 0; //number of candidates that are of enough matches
    for(int i=0; i<m_v_consistentCandidates.size();i++){
      if(m_v_consistentCandidates[i]){
        m_v_consistentCandidates[i]->setEraseExemption();
        if(!m_v_consistentCandidates[i]->isBad() && matcher.searchByBowInTwoKeyFrames(m_sptr_currentLoopKeyFrame, m_v_consistentCandidates[i], vvSptrMatchedMapPoints[i])>=20){
          std::shared_ptr<Sim3Solver> sptrSolver = std::make_shared<Sim3Solver>(m_sptr_currentLoopKeyFrame, m_v_consistentCandidates[i], vvSptrMatchedMapPoints[i], m_b_isScaleFixed);
          sptrSolver->setRansacParameters(0.99,20,300);
          vsptrSim3Solvers[i] = sptrSolver;
          validCandidatesNum++;
        }else{
          vIsDiscardeds[i] = true;
        }
      }
    }
    bool bIsSim3OptSuccessful = false;
    //perform alternatively ransac iterations for each candidate
    //until one is successful or all fail
    while(validCandidatesNum>0 && !bIsSim3OptSuccessful){
      for(int i=0;i<m_v_consistentCandidates.size();i++){
        if(!vIsDiscardeds[i]){
          //perform 5 ransac iterations
          std::vector<bool> vIsInliers;
          int intInliersNum;
          bool bIsNoMore;
          std::shared_ptr<Sim3Solver> sptrSolver = vsptrSim3Solvers[i];
          cv::Mat sim3_cur2cand = sptrSolver->iterate(5,bIsNoMore,vIsInliers,intInliersNum);
          //if ransac reaches maximum, iterations discard key frame
          if(bIsNoMore){
            vIsDiscardeds[i] = true;
            validCandidatesNum--;
          }
          //if ransac returns a sim3, perform a guided matching and optimize with all correspondence
          if(!sim3_cur2cand.empty()){
            std::vector<std::shared_ptr<MapPoint>> vSptrMatchedMapPoints(vvSptrMatchedMapPoints[i].size(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
            for(int j=0;j<vIsInliers.size();j++){
              if(vIsInliers[j]){
                vSptrMatchedMapPoints[j] = vvSptrMatchedMapPoints[i][j];
              }
            }
            cv::Mat rotation = sptrSolver->getEstimationRotation();
            cv::Mat translation = sptrSolver->getEstimatedTranslation();
            const float scale = sptrSolver->getEstimatedScale();
            matcher.searchBySim3(m_sptr_currentLoopKeyFrame,m_v_consistentCandidates[i],vSptrMatchedMapPoints,scale,rotation,translation,7.5);
            g2o::Sim3 g2oSim3_cur2cand(Converter::d3X3Matrix_cvMat_eigen(rotation),Converter::d3X1Matrix_cvMat_eigen(translation),scale);
            const int intInliers = ;
            //if optimization is successful, stop ransacs and continue
            if((Optimizer::optimizeSim3(m_sptr_currentLoopKeyFrame,m_v_consistentCandidates[i],vSptrMatchedMapPoints,g2oSim3,10,m_b_isScaleFixed))>=20){
              bIsSim3OptSuccessful = true;
              m_sptr_currentLoopKeyFrame = m_v_consistentCandidates[i];
              g2o::Sim3 g2oSim3_cand2world(Converter::d3X3Matrix_cvMat_eigen(m_v_consistentCandidates[i]->getRotation_c2w()),Converter::d3X1Matrix_cvMat_eigen(m_v_consistentCandidates[i]->getTranslation_c2w()),1.0);
              m_g2o_sim3_c2w = g2oSim3_cur2cand * g2oSim3_cand2world;
              m_cvMat_sim3_c2w = Converter::transform_Sim3_cvMat(m_g2o_sim3_c2w);
              m_v_matchedMapPoints = vSptrMatchedMapPoints;
              break;
            }
          }
        }
      }
    }
    if(!bIsSim3OptSuccessful){
      for(std::shared_ptr<KeyFrame> &candidateKeyFrame : m_v_consistentCandidates){
        candidateKeyFrame->cancelEraseExemption();
      }
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
    //retrieve map points seen in loop key frame and connections
    //project all map points in loop key frame and connections into current key frame
    std::vector<std::shared_ptr<KeyFrame>> vSptrLoopConnectedKeyFrames = m_sptr_currentMatchedKeyFrame->getOrderedConnectedKeyFrames();
    vSptrLoopConnectedKeyFrames.push_back(m_sptr_currentMatchedKeyFrame);
    m_v_loopMapPoints.clear();
    for(std::shared_ptr<KeyFrame> &sptrLoopConnectedKeyFrame : vSptrLoopConnectedKeyFrames){
      std::vector<std::shared_ptr<MapPoint>> vSptrMapPoints = sptrLoopConnectedKeyFrame->getMatchedMapPointsVec();
      for(std::shared_ptr<MapPoint> &sptrMapPoint : vSptrMapPoints){
        if(sptrMapPoint && !sptrMapPoint->isBad() && sptrMapPoint->m_int_loopPointForKeyFrameID!=m_sptr_currentLoopKeyFrame->m_int_keyFrameID){
          m_v_loopMapPoints.push_back(sptrMapPoint);
          sptrMapPoint->m_int_loopPointForKeyFrameID = m_sptr_currentLoopKeyFrame->m_int_keyFrameID;
        }
      }
    }
    //find more matches projecting with the computed Sim3
    matcher.searchByProjectionInSim(m_sptr_currentLoopKeyFrame,m_cvMat_sim3_c2w,m_v_loopMapPoints,m_v_matchedMapPoints,10);
    //if matches are enough, then accept loop
    int intTotalMatchesNum = 0;
    for(std::shared_ptr<MapPoint> &sptrMatchedMapPoint : m_v_matchedMapPoints){
      if(sptrMatchedMP){
        intTotalMatchesNum++;
      }
    }
    if(intTotalMatchesNum >= 40){
      for(std::shared_ptr<KeyFrame> &candidateKeyFrame : m_v_consistentCandidates){
        if(candidateKeyFrame != m_sptr_currentMatchedKeyFrame){
          candidateKeyFrame->cancelEraseExemption();
        }
      }
      return true;
    }else{
      for(std::shared_ptr<KeyFrame>& candidateKeyFrame : m_v_consistentCandidates){
        candidateKeyFrame->cancelEraseExemption();
      }
      m_sptr_currentLoopKeyFrame->cancelEraseExemption();
      return false;
    }
  }
  void LoopClosing::correctLoop(){
    std::cout<<"loop detected!"<<std::endl;
    //send a stop signal to local mapping to avoid new key frame insertion while correcting
    m_sptr_localMapper->requestStop();
    //if a global adjustment is running, stop it
    if(isRunningGlobalBA()){
      std::unique_lock<std::mutex> lock(m_mutex_globalBA);
      m_b_isGlobalBAStopped = true;
      m_int_fullBAIndex++;
      if(m_sptr_globalBAThread){
        m_sptr_globalBAThread->detach();
        m_sptr_globalBAThread.reset();
      }
    }
    //wait until local mapping is effectively stopped
    while(!m_sptr_localMapper->isStopped()){
      usleep(1000);
    }
    //ensure rcurrent key frame is updated
    m_sptr_currentLoopKeyFrame->updateConnections();
    //retrieve key frames connected to the current key frame and compute corrected Sim3 pose by propagation
    m_v_connectedKeyFrames = m_sptr_currentLoopKeyFrame->getOrderedConnectedKeyFrames();
    m_v_connectedKeyFrames.push_back(m_sptr_currentLoopKeyFrame);
    KeyFrameAndPose correctedSim3s, nonCorrectedSim3s;
    correctedSim3s[m_sptr_currentLoopKeyFrame] = m_g2o_sim3_c2w;
    cv::Mat T_w2c = m_sptr_currentLoopKeyFrame->getInverseCameraPoseByTransform_w2c();
    {
      std::unique_lock<std::mutex> lock(m_sptr_map->m_mutex_updateMap);
      for(std::shared_ptr<KeyFrame> &sptrConnectedKeyFrame : m_v_connectedKeyFrames){
        cv::Mat T_cnct2w = sptrConnectedKeyFrame->getCameraPoseByTransrom_c2w();
        if(sptrConnectedKeyFrame != m_sptr_currentLoopKeyFrame){
          cv::Mat T_cnct2c = T_cnct2w * T_w2c;
          cv::Mat R_cnct2c = T_cnct2c.rowRange(0,3).colRange(0,3);
          cv::Mat t_cnct2c = T_cnct2c.rowRange(0,3).col(3);
          g2o::Sim3 g2oSim3_cnct2c(Converter::d3X3Matrix_cvMat_eigen(R_cnct2c),Converter::d3X1Matrix_cvMat_eigen(t_cnct2c),1.0);
          g2o::Sim3 g2oCorrectedSim3_cnct2w = g2oSim3_cnct2c * m_g2o_sim3_c2w;
          //pose corrected with the sim3 of the loop closure
          correctedSim3s[sptrConnectedKeyFrame] = g2oCorrectedSim3_cnct2w;
        }
        cv::Mat R_cnct2w = T_cnct2w.rowRange(0,3).colRange(0,3);
        cv::Mat t_cnct2w = T_cnct2w.rowRange(0,3).col(3);
        g2o::Sim3 g2oSim3_cnct2w(Converter::d3X3Matrix_cvMat_eigen(R_cnct2w),Converter::d3X1Matrix_cvMat_eigen(t_cnct2w),1.0);
        //Pose without correction
        nonCorrectedSim3s[sptrConnectedKeyFrame]=g2oSim3_cnct2w;
      }
      //correct all map points observed by current key frame and connections, so that they align with the other side of the loop
      for(const std::pair<std::shared_ptr<KeyFrame>,g2o::Sim3> &correctedSim3 : correctedSim3s){
        for(std::shared_ptr<MapPoint>& sptrMapPoint : correctedSim3.first->getMatchedMapPointsVec()){
          if(sptrMapPoint && !sptrMapPoint->isBad() && sptrMapPoint->m_int_correctedByKeyFrameID!=m_sptr_currentLoopKeyFrame->m_int_keyFrameID){
            //project with non-corrected pose and project back with corrected pose
            cv::Mat mapPointPosInWorld = sptrMapPoint->getPosInWorld();
            Eigen::Matrix<double,3,1> eigen_mapPointPosInWorld = Converter::d3X1Matrix_cvMat_eigen(mapPointPosInWorld);
            Eigen::Matrix<double,3,1> eigen_correctedMapPointPosInWorld = correctedSim3.second.inverse().map(nonCorrectedSim3s[correctedSim3.first].map())   map(g2oSiw.map(eigen_mapPointPosInWorld));
            cv::Mat cvMat_correctedMapPointPosInWorld = Converter::d3X1Matrix_eigen_cvMat(eigen_correctedMapPointPosInWorld);
            sptrMapPoint->setPosInWorld(cvMat_correctedMapPointPosInWorld);
            sptrMapPoint->m_int_correctedByKeyFrameID = m_sptr_currentLoopKeyFrame->m_int_keyFrameID;
            sptrMapPoint->m_int_correctedRefKeyFrameID = correctedSim3.first->m_int_keyFrameID;
            sptrMapPoint->updateNormalAndDepth();
          }
        }
      }
      //update keyframe pose
      //stop here
    }
  }
}
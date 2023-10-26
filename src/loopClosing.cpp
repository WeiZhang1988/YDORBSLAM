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
        //update keyframe pose
        correctedSim3.first->setCameraPoseByTransform_c2w(Converter::transform_eigen_cvMat(correctedSim3.second.rotation().toRotationMatrix(),correctedSim3.second.translation()/correctedSim3.second.scale()));
        correctedSim3.first->updateConnections();
      }
      //start loop function
      //update matched map points and replace if duplicated
      for(int i=0;i<m_v_matchedMapPoints.size();i+){
        if(m_v_matchedMapPoints[i]){
          if(m_sptr_currentLoopKeyFrame->getMapPoint(i)){
            m_sptr_currentLoopKeyFrame->getMapPoint(i)->beReplacedBy(m_v_matchedMapPoints[i]);
          }else{
            m_sptr_currentLoopKeyFrame->addMapPoint(m_v_matchedMapPoints[i],i);
            m_v_matchedMapPoints[i]->addObservation(m_sptr_currentLoopKeyFrame,i);
            m_v_matchedMapPoints[i]->computeDistinctiveDescriptors();
          }
        }
      }
    }
    //project map points observed in the connection of the loop key frame
    //into the current key frame and connections using corrected poses
    //fuse duplications
    searchAndFuse(correctedSim3s);
    //after the map point fused, new links in the connection graph will apprea attaching both sides of the loop
    std::map<std::shared_ptr<KeyFrame>,std::set<std::shared_ptr<KeyFrame>>> loopConnections;
    for(std::shared_ptr<KeyFrame> &connectedKeyFrame : m_v_connectedKeyFrames){
      //update connections. detect new links
      connectedKeyFrame->updateConnections();
      loopConnections[connectedKeyFrame] = connectedKeyFrame->getConnectedKeyFrames();
      for(std::shared_ptr<KeyFrame> &preConnectedKeyFrame : connectedKeyFrame->getOrderedConnectedKeyFrames()){
        loopConnections[connectedKeyFrame].erase(preConnectedKeyFrame);
      }
      for(std::shared_ptr<KeyFrame> &updatedConnectedKeyFrame : m_v_connectedKeyFrames){
        loopConnections[connectedKeyFrame].erase(updatedConnectedKeyFrame);
      }
    }
    //optimize graph
    Optimizer::optimizeEssentialGraph(m_sptr_map,m_sptr_currentMatchedKeyFrame,m_sptr_currentLoopKeyFrame,nonCorrectedSim3s,correctedSim3s,loopConnections,m_b_isScaleFixed);
    m_sptr_map->informNewBigChange();
    //add loop edge
    m_sptr_currentMatchedKeyFrame->addLoopEdge(m_sptr_currentLoopKeyFrame);
    m_sptr_currentLoopKeyFrame->addLoopEdge(m_sptr_currentMatchedKeyFrame);
    //launch a new thread to perform global bundle adjustment
    m_b_isRunningGlobalBA = true;
    m_b_isGlobalBAFinished = false;
    m_b_isGlobalBAStopped = false;
    m_sptr_globalBAThread = std::make_shared<thread>(&LoopClosing::runGlobalBundleAdjustment,shared_from_this(),m_sptr_currentLoopKeyFrame->m_int_keyFrameID);
    m_sptr_localMapper->release();
    m_int_lastLoopKeyFrameID = m_sptr_currentLoopKeyFrame->m_int_keyFrameID;
  }
  void LoopClosing::searchAndFuse(const KeyFrameAndPose &_correctedPosesMap){
    OrbMatcher matcher(0.8);
    for(const std::pair<std::shared_ptr<KeyFrame>,g2o::Sim3>& poseMap : _correctedPosesMap){
      std::vector<std::shared_ptr<MapPoint>> vBeingReplacedMapPoints(m_v_loopMapPoints.size(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
      matcher.fuseBySim3(poseMap.first,Converter::toCvMat(poseMap.second),m_v_loopMapPoints,4,vBeingReplacedMapPoints);
      //get map mutex
      std::unique_lock<std::mutex> lock(m_sptr_map->m_mutex_updateMap);
      for(int i=0;i<m_v_loopMapPoints.size();i++){
        if(vBeingReplacedMapPoints[i]){
          vBeingReplacedMapPoints[i]->beReplacedBy(m_v_loopMapPoints[i]);
        }
      }
    }
  }
  void LoopClosing::requestReset(){
    {
      std::unique_lock<std::mutex> lock(m_mutex_reset);
      m_b_isResetRequested = true;
    }
    while(true){
      {
        std::unique_lock<std::mutex> lock2(m_mutex_reset);
        if(!m_b_isResetRequested){
          break;
        }
      }
      usleep(5000);
    }
  }
  void LoopClosing::resetIfRequested(){
    std::unique_lock<std::mutex> lock(m_mutex_reset);
    if(m_b_isResetRequested)
    {
      m_list_keyFrameBufferQueue.clear();
      m_int_lastLoopKeyFrameID=0;
      m_b_isResetRequested=false;
    }
  }
  void LoopClosing::runGlobalBundleAdjustment(long int _int_loopKeyFrameID){
    std::cout << "starting global bundle adjustment..." << std::endl;
    int idx = m_int_fullBAIndex;
    Optimizer::globalBundleAdjust(m_sptr_map,10,m_b_isGlobalBAStopped,_int_loopKeyFrameID,false);
    //update all map points and key frames
    //local mapping was active during BA, which means that there might be new key frames not included in the global BA
    //and they are not consistent with the updated map
    //correction through the spanning tree needs to be propagated
    {
      std::unique_lock<std::mutex> lock(m_mutex_globalBA);
      if(idx == m_int_fullBAIndex){
        if(!m_b_isGlobalBAStopped){
          std::cout << "global bundle adjustment finished" << std::endl;
          std::cout << "updating map ..." << std::endl;
          m_sptr_localMapper->requestStop();
          //wait untile local mapping has effectively stopped
          while(!m_sptr_localMapper->isStopped() && !m_sptr_localMapper->isFinished()){
            usleep(1000);
          }
          //get map mutex
          std::unique_lock<std::mutex> lock((m_sptr_map->m_mutex_updateMap);
          //correct key frames starting at map first key frame
          std::list<std::shared_ptr<KeyFrame>> listKeyFramesToBeChecked(m_sptr_map->m_v_sptrOriginalKeyFrames.begin(),m_sptr_map->m_v_sptrOriginalKeyFrames.end());
          while(!listKeyFramesToBeChecked.empty()){
            std::shared_ptr<KeyFrame> firstKeyFrame = listKeyFramesToBeChecked.front();
            const std::set<std::shared_ptr<KeyFrame>> children = pKF->getChildren();
            for(const std::shared_ptr<KeyFrame> &childKeyFrame : children){
              if(childKeyFrame->m_int_globalBAForKeyFrameID!=_int_loopKeyFrameID){
                childKeyFrame->m_cvMat_T_c2w_GlobalBA = childKeyFrame->getCameraPoseByTransrom_c2w * firstKeyFrame->getInverseCameraPoseByTransform_w2c() * firstKeyFrame->m_cvMat_T_c2w_GlobalBA;
                childKeyFrame->m_int_globalBAForKeyFrameID = _int_loopKeyFrameID;
              }
              listKeyFramesToBeChecked.push_back(childKeyFrame);
            }
            firstKeyFrame->m_cvMat_T_c2w_beforeGlobalBA = firstKeyFrame->getCameraPoseByTransrom_c2w();
            firstKeyFrame->setCameraPoseByTransform_c2w(firstKeyFrame->m_cvMat_T_c2w_GlobalBA);
            listKeyFramesToBeChecked.pop_front();
          }
          //coorect map points
          for(const std::shared_ptr<MapPoint> &mapPoint : m_sptr_map->getAllMapPoints()){
            if(mapPoint && !mapPoint->isBad()){
              if(mapPoint->m_int_globalBAforKeyFrameID == _int_loopKeyFrameID){
                //if optimized by global BA just update
                mapPoint->setPosInWorld(mapPoint->m_cvMat_posGlobalBA);
              }else{
                //update according to the correction of its reference key frame
                std::shared_ptr<KeyFrame> referenceKeyFrame = mapPoint->getReferenceKeyFrame();
                if(referenceKeyFrame->m_int_globalBAForKeyFrameID!==_int_loopKeyFrameID){
                  // Map to non-corrected camera
                  cv::Mat rotation_c2w = referenceKeyFrame->m_cvMat_T_c2w_beforeGlobalBA.rowRange(0,3).colRange(0,3);
                  cv::Mat translation_c2w = referenceKeyFrame->m_cvMat_T_c2w_beforeGlobalBA.rowRange(0,3).col(3);
                  cv::Mat cameraInWorld = rotation_c2w*mapPoint->getPosInWorld()+translation_c2w;
                  // Backproject using corrected camera
                  cv::Mat Tansform_w2c = referenceKeyFrame->getInverseCameraPoseByTransform_w2c();
                  cv::Mat rotation_w2c = Tansform_w2c.rowRange(0,3).colRange(0,3);
                  cv::Mat translation_w2c = Tansform_w2c.rowRange(0,3).col(3)
                  mapPoint->setPosInWorld(rotation_w2c * cameraInWorld + translation_w2c);
                }
              }
            }
          }
          m_sptr_map->informNewBigChange();
          m_sptr_localMapper->release();
          std::cout << "map updated!" << std::endl;
        }
        m_b_isGlobalBAFinished = true;
        m_b_isRunningGlobalBA = false;
      }
    }
  }
}//namespace YDORBSLAM
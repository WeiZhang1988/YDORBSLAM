/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

# include "loopClosing.hpp"
# include "sim3Solver.hpp"
# include "converter.hpp"
# include "optimizer.hpp"
# include "orbMatcher.hpp"
# include <mutex>
# include <thread>
# include <unistd.h>

namespace YDORBSLAM{

LoopClosing::LoopClosing(std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrameDatabase> _sptrDB, std::shared_ptr<DBoW3::Vocabulary> _sptrVoc, const bool& _bFixScale):\
m_b_resetRequested(false){
  m_int_covisibilityConsistencyTh = 3;
}

void LoopClosing::run(){
  m_b_finished = false;
  while(1){
    // Check if there are keyframes in the queue
    // Detect loop candidates and check covisibility consistency
    // Compute similarity transformation [sR|t] in the stereo/RGBD case s=1
    if(checkNewKeyFrames() && detectLoop() && computeSim3()){
      // Perform loop fusion and pose graph optimization
      correctLoop();
    }
    resetIfRequested();
    if(checkFinish()){
      break;
    }
    usleep(5000);
  }
  setFinish();
}

bool LoopClosing::detectLoop(){
  {
    std::unique_lock<std::mutex> lock(m_mutex_loopQueue);
    m_sptrCurrentLoopKF = m_l_sptrLoopKeyFrameBufferQueue.front();
    m_l_sptrLoopKeyFrameBufferQueue.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this thread
    m_sptrCurrentLoopKF->setEraseExemption();
  }
  //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
  if(m_sptrCurrentLoopKF->m_int_keyFrameID < m_int_lastLoopKFid+10){
    m_sptrKeyFrameDB->add(m_sptrCurrentLoopKF);
    m_sptrCurrentLoopKF->cancelEraseExemption();
    return false;
  }
  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  float minScore = 1;
  const DBoW3::BowVector& currentBowVec = m_sptrCurrentLoopKF->m_bow_wordVec;
  for(const std::shared_ptr<KeyFrame>& sptrConnectedKF : m_sptrCurrentLoopKF->getOrderedConnectedKeyFrames()){
    if(!sptrConnectedKF->isBad()){
      const DBoW3::BowVector& connectedBowVec = sptrConnectedKF->m_bow_wordVec;
      float score = m_sptrORBVocabulary->score(currentBowVec, connectedBowVec);
      if(score < minScore){
        minScore = score;
      }
    }
  }
  // Query the database imposing the minimum score
  std::vector<std::shared_ptr<KeyFrame>> vsptrCandidateKFs = m_sptrKeyFrameDB->detectLoopCandidates(m_sptrCurrentLoopKF, minScore);
  // If there are no loop candidates, just add new keyframe and return false
  if(vsptrCandidateKFs.empty()){
    m_sptrKeyFrameDB->add(m_sptrCurrentLoopKF);
    m_v_lastConsistentGroups.clear();
    m_sptrCurrentLoopKF->cancelEraseExemption();
    return false;
  }
  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
  // A group is consistent with a previous group if they share at least a keyframe
  // We must detect a consistent loop in several consecutive keyframes to accept it
  m_v_sptrEnoughConsistentCandidates.clear();
  std::vector<setSptrKFAndLength> vCurrentConsistentGroups;
  std::vector<bool> vbLastConsistentGroup(m_v_lastConsistentGroups.size(),false);
  for(std::shared_ptr<KeyFrame>& sptrCandidateKF : vsptrCandidateKFs){
    std::set<std::shared_ptr<KeyFrame>> setCandidateGroup = sptrCandidateKF->getConnectedKeyFrames();
    setCandidateGroup.insert(sptrCandidateKF);
    bool bEnoughConsistent = false;
    bool bConsistentForSomeGroup = false;
    int ifor = 0;
    for(setSptrKFAndLength& lastGroup : m_v_lastConsistentGroups){
      for(const std::shared_ptr<KeyFrame>& candidateGroup : setCandidateGroup){
        if(lastGroup.first.count(candidateGroup)){
          bConsistentForSomeGroup = true;
          if(!vbLastConsistentGroup[ifor]){
            setSptrKFAndLength cGroup = std::make_pair(setCandidateGroup, lastGroup.second+1);
            vCurrentConsistentGroups.push_back(cGroup);
            //this avoid to include the same group more than once
            vbLastConsistentGroup[ifor] = true;
          }
          if((lastGroup.second+1)>=m_int_covisibilityConsistencyTh && !bEnoughConsistent){
            m_v_sptrEnoughConsistentCandidates.push_back(sptrCandidateKF);
            //this avoid to insert the same candidate more than once
            bEnoughConsistent = true;
          }
          break;
        }
      }
      ifor++;
    }
    // If the group is not consistent with any previous group insert with consistency counter set to zero
    if(!bConsistentForSomeGroup){
      setSptrKFAndLength cGroup = std::make_pair(setCandidateGroup, 0);
      vCurrentConsistentGroups.push_back(cGroup);
    }
  }
  // Update Covisibility Consistent Groups
  m_v_lastConsistentGroups = vCurrentConsistentGroups;
  // Add Current Keyframe to database
  m_sptrKeyFrameDB->add(m_sptrCurrentLoopKF);
  if(m_v_sptrEnoughConsistentCandidates.empty()){
    m_sptrCurrentLoopKF->cancelEraseExemption();
    return false;
  }else{
    return true;
  }
  m_sptrCurrentLoopKF->cancelEraseExemption();
  return false;
}

bool LoopClosing::computeSim3(){
  // For each consistent loop candidate we try to compute a Sim3
  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  OrbMatcher matcher(0.75,true);
  std::vector<std::shared_ptr<Sim3Solver>> vsptrSim3Solvers;
  vsptrSim3Solvers.resize(m_v_sptrEnoughConsistentCandidates.size());
  std::vector<std::vector<std::shared_ptr<MapPoint>>> vvsptrMatchedMPs;
  vvsptrMatchedMPs.resize(m_v_sptrEnoughConsistentCandidates.size());
  std::vector<bool> vbDiscarded;
  vbDiscarded.resize(m_v_sptrEnoughConsistentCandidates.size());
  int nCandidates = 0; //candidates with enough matches
  int ifor = 0;
  for(std::shared_ptr<KeyFrame>& candidateKF : m_v_sptrEnoughConsistentCandidates){
    candidateKF->setEraseExemption();
    if(candidateKF->isBad()){
      vbDiscarded[i] = true;
      continue;
    }
    int nBowMatches = matcher.searchByBowInTwoKeyFrames(m_sptrCurrentLoopKF, candidateKF, vvsptrMatchedMPs[ifor]);
    if(nBowMatches < 20){
      vbDiscarded[i] = true;
      continue;
    }else{
      std::shared_ptr<Sim3Solver> sptrSolver = std::make_shared<Sim3Solver>(m_sptrCurrentLoopKF, candidateKF, vvsptrMatchedMPs[ifor], m_b_fixScale);
      sptrSolver->setRansacParameters(0.99,20,300);
      vsptrSim3Solvers[ifor] = sptrSolver;
    }
    nCandidates++;
  }
  bool bSim3OpitmizeSuccess = false;
  // Perform alternatively RANSAC iterations for each candidate
  // until one is succesful or all fail
  while(nCandidates>0 && !bSim3OpitmizeSuccess){
    for(int i=0; i<m_v_sptrEnoughConsistentCandidates.size(); i++){
      if(vbDiscarded[i]){
        continue;
      }
      std::shared_ptr<KeyFrame> candidateKF = m_v_sptrEnoughConsistentCandidates[i];
      // Perform 5 Ransac Iterations
      std::vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;
      std::shared_ptr<Sim3Solver> sptrSolver = vsptrSim3Solvers[i];
      cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
      // If Ransac reachs max. iterations discard keyframe
      if(bNoMore){
        vbDiscarded[i] = true;
        nCandidates--;
      }
      // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
      if(!Scm.empty()){
        std::vector<std::shared_ptr<MapPoint>> vsptrMatchedMPs(vvsptrMatchedMPs[i].size(), static_cast<std::shared_ptr<MapPoint>>(nullptr));
        for(size_t j=0; j<vbInliers.size(); j++){
          if(vbInliers[j]){
            vsptrMatchedMPs[j] = vvsptrMatchedMPs[i][j];
          }
        }
        cv::Mat R = pSolver->getEstimatedRotation();
        cv::Mat t = pSolver->getEstimatedTranslation();
        const float s = pSolver->getEstimatedScale();
        matcher.searchBySim3(m_sptrCurrentLoopKF,candidateKF,vsptrMatchedMPs,s,R,t,7.5);
        g2o::Sim3 gScm(Converter::d3X3Matrix_cvMat_eigen(R),Converter::d3X1Matrix_cvMat_eigen(t),s);
        const int nInliers = Optimizer::optimizeSim3(m_sptrCurrentLoopKF,candidateKF,vsptrMatchedMPs,gScm,10,m_b_fixScale);
        // If optimization is succesful stop ransacs and continue
        if(nInliers >= 20){
          bSim3OpitmizeSuccess = true;
          m_sptrCurrentClosedLoopMatchedKF = candidateKF;
          g2o::Sim3 gSmw(Converter::d3X3Matrix_cvMat_eigen(candidateKF->getRotation_c2w()),Converter::d3X1Matrix_cvMat_eigen(candidateKF->getTranslation_c2w()),1.0);
          m_g2oScw = gScm * gSmw;
          m_cvMatScw = Converter::transform_Sim3_cvMat(m_g2oScw);
          m_v_sptrCurrentMatchedPoints = vsptrMatchedMPs;
          break;
        }
      }
    }
  }
  if(!bSim3OpitmizeSuccess){
    for(std::shared_ptr<KeyFrame>& candidateKF : m_v_sptrEnoughConsistentCandidates){
      candidateKF->cancelEraseExemption();
      m_sptrCurrentLoopKF->cancelEraseExemption();
      return false;
    }
  }
  // Project all MapPoints in Loop Keyframe and neighbors to current KeyFrame
  std::vector<std::shared_ptr<KeyFrame>> vsptrLoopConnectedKFs = m_sptrCurrentClosedLoopMatchedKF->getOrderedConnectedKeyFrames();
  vsptrLoopConnectedKFs.push_back(m_sptrCurrentClosedLoopMatchedKF);
  m_v_sptrLoopMapPoints.clear();
  for(std::shared_ptr<KeyFrame>& sptrLoopConnectedKF : vsptrLoopConnectedKFs){
    std::vector<std::shared_ptr<MapPoint>> vsptrMPs = sptrLoopConnectedKF->getMatchedMapPointsVec();
    for(std::shared_ptr<MapPoint>& sptrMP : vsptrMPs){
      if(sptrMP && !sptrMP->isBad() && sptrMP->m_int_loopPointForKeyFrameID!=m_sptrCurrentLoopKF->m_int_keyFrameID){
        m_v_sptrLoopMapPoints.push_back(sptrMP);
        sptrMP->m_int_loopPointForKeyFrameID = m_sptrCurrentLoopKF->m_int_keyFrameID;
      }
    }
  }
  matcher.searchByProjectionInSim(m_sptrCurrentLoopKF,m_cvMatScw,m_v_sptrLoopMapPoints,m_v_sptrCurrentMatchedPoints,10);
  // If enough matches accept Loop
  int nTotalMatches = 0;
  for(std::shared_ptr<MapPoint>& sptrMatchedMP : m_v_sptrCurrentMatchedPoints){
    if(sptrMatchedMP){
      nTotalMatches++;
    }
  }
  if(nTotalMatches >= 40){
    for(std::shared_ptr<KeyFrame>& candidateKF : m_v_sptrEnoughConsistentCandidates){
      if(candidateKF != m_sptrCurrentClosedLoopMatchedKF){
        candidateKF->cancelEraseExemption();
      }
    }
    return true;
  }else{
    for(std::shared_ptr<KeyFrame>& candidateKF : m_v_sptrEnoughConsistentCandidates){
      candidateKF->cancelEraseExemption();
    }
    m_sptrCurrentLoopKF->cancelEraseExemption();
    return false;
  }
}

void LoopClosing::correctLoop(){
  std::cout<< "Loop detected!" <<std::endl;
  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop
  m_sptrLocalMapper->requestStop();
  // If a Global Bundle Adjustment is running, abort it
  if(isRunningGlobalBA()){
    std::unique_lock<std::mutex> lock(m_mutex_globalBA);
    m_b_stopGlobalBA = true;
    m_int_fullBAIdx++;
    if(m_sptrThreadGlobalBA){
      m_sptrThreadGlobalBA->detach();
      m_sptrThreadGlobalBA.reset();
    }
  }
  // Wait until Local Mapping has effectively stopped
  while(!m_sptrLocalMapper->isStopped()){
    usleep(1000);
  }
  // Ensure current keyframe is updated
  m_sptrCurrentLoopKF->updateConnections();
  // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
  m_v_sptrCurrentConnectedKFs = m_sptrCurrentLoopKF->getOrderedConnectedKeyFrames();
  m_v_sptrCurrentConnectedKFs.push_back(m_sptrCurrentLoopKF);
  sptrKeyFrameAndPose correctedSim3, nonCorrectedSim3;
  correctedSim3[m_sptrCurrentLoopKF] = m_g2oScw;
  cv::Mat Twc = m_sptrCurrentLoopKF->getInverseCameraPoseByTransform_w2c();
  {
    std::unique_lock<std::mutex> lock(m_sptrMap->m_mutex_updateMap);
    for(std::shared_ptr<KeyFrame>& sptrConnectedKFi : m_v_sptrCurrentConnectedKFs){
      cv::Mat Tiw = sptrConnectedKFi->getCameraPoseByTransrom_c2w();
      if(sptrConnectedKFi != m_sptrCurrentLoopKF){
        cv::Mat Tic = Tiw * Twc;
        cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
        cv::Mat tic = Tic.rowRange(0,3).col(3);
        g2o::Sim3 g2oSic(Converter::d3X3Matrix_cvMat_eigen(Ric),Converter::d3X1Matrix_cvMat_eigen(tic),1.0);
        g2o::Sim3 g2oCorrectedSiw = g2oSic * m_g2oScw;
        //Pose corrected with the Sim3 of the loop closure
        correctedSim3[sptrConnectedKFi] = g2oCorrectedSiw;
      }
      cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
      cv::Mat tiw = Tiw.rowRange(0,3).col(3);
      g2o::Sim3 g2oSiw(Converter::d3X3Matrix_cvMat_eigen(Riw),Converter::d3X1Matrix_cvMat_eigen(tiw),1.0);
      //Pose without correction
      nonCorrectedSim3[sptrConnectedKFi] = g2oSiw;
    }
    // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
    for(const std::pair<std::shared_ptr<KeyFrame>,g2o::Sim3>& corSim3 : correctedSim3){
      std::shared_ptr<KeyFrame> sptrKFi = corSim3.first;
      g2o::Sim3 g2oCorrectedSiw = corSim3.second;
      g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
      g2o::Sim3 g2oSiw = nonCorrectedSim3[sptrKFi];
      std::vector<std::shared_ptr<MapPoint>> vsptrMPs = sptrKFi->getMatchedMapPointsVec();
      for(std::shared_ptr<MapPoint>& sptrMPi : vsptrMPs){
        if(sptrMPi && !sptrMPi->isBad() && sptrMPi->m_int_correctedByKeyFrameID!=m_sptrCurrentLoopKF->m_int_keyFrameID){
          // Project with non-corrected pose and project back with corrected pose
          cv::Mat mapPointPosInWorld = sptrMPi->getPosInWorld();
          Eigen::Matrix<double,3,1> eigen_mapPointPosInWorld = Converter::d3X1Matrix_cvMat_eigen(mapPointPosInWorld);
          Eigen::Matrix<double,3,1> eigen_correctedMapPointPosInWorld = g2oCorrectedSwi.map(g2oSiw.map(eigen_mapPointPosInWorld));
          cv::Mat cvMat_correctedMapPointPosInWorld = Converter::d3X1Matrix_eigen_cvMat(eigen_correctedMapPointPosInWorld);
          sptrMPi->setPosInWorld(cvMat_correctedMapPointPosInWorld);
          sptrMPi->m_int_correctedByKeyFrameID = m_sptrCurrentLoopKF->m_int_keyFrameID;
          sptrMPi->m_int_correctedRefKeyFrameID = sptrKFi->m_int_keyFrameID;
          sptrMPi->updateNormalAndDepth();
        }
      }
      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
      Eigen::Matrix3d eigen_Riw = g2oCorrectedSiw.rotation().toRotationMatrix();
      Eigen::Vector3d eigen_tiw = g2oCorrectedSiw.translation();
      double siw = g2oCorrectedSiw.scale();
      eigen_tiw *=(1./s); //[R t/s;0 1]
      cv::Mat correctedTiw = Converter::transform_eigen_cvMat(eigen_Riw,eigen_tiw);
      sptrKFi->setCameraPoseByTransform_c2w(correctedTiw);
      // Make sure connections are updated
      sptrKFi->updateConnections();
    }
    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    int ifor = 0;
    for(std::shared_ptr<MapPoint>& sptrLoopMP : m_v_sptrCurrentMatchedPoints){
      if(sptrLoopMP){
        std::shared_ptr<MapPoint> sptrCurrentMP = m_sptrCurrentLoopKF->getMapPoint(ifor);
        if(sptrCurrentMP){
          sptrCurrentMP->beReplacedBy(sptrLoopMP);
        }else{
          m_sptrCurrentLoopKF->addMapPoint(sptrLoopMP,ifor);
          sptrLoopMP->addObservation(m_sptrCurrentLoopKF,ifor);
          sptrLoopMP->computeDistinctiveDescriptors();
        }
      }
      ifor++;
    }
  }
  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  searchAndFuse(correctedSim3);
  // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
  std::map<std::shared_ptr<KeyFrame>,std::set<std::shared_ptr<KeyFrame>>> loopConnections;
  for(std::shared_ptr<KeyFrame>& sptrConnectedKFi : m_v_sptrCurrentConnectedKFs){
    std::vector<std::shared_ptr<KeyFrame>> vsptrPreviousNeighbors = sptrConnectedKFi->getOrderedConnectedKeyFrames();
    // Update connections. Detect new links.
    sptrConnectedKFi->updateConnections();
    loopConnections[sptrConnectedKFi] = sptrConnectedKFi->getConnectedKeyFrames();
    for(std::shared_ptr<KeyFrame>& sptrNeighborKFi : vsptrPreviousNeighbors){
      loopConnections[sptrConnectedKFi].erase(sptrNeighborKFi);
    }
    for(std::shared_ptr<KeyFrame>& sptrUpdateConnectedKFi : m_v_sptrCurrentConnectedKFs){
      loopConnections[sptrConnectedKFi].erase(sptrUpdateConnectedKFi);
    }
  }
  // Optimize graph
  Optimizer::optimizeEssentialGraph(m_sptrMap,m_sptrCurrentClosedLoopMatchedKF,m_sptrCurrentLoopKF,nonCorrectedSim3,correctedSim3,loopConnections,m_b_fixScale);
  m_sptrMap->informNewBigChange();
  // Add loop edge
  m_sptrCurrentClosedLoopMatchedKF->addLoopEdge(m_sptrCurrentLoopKF);
  m_sptrCurrentLoopKF->addLoopEdge(m_sptrCurrentClosedLoopMatchedKF);
  // Launch a new thread to perform Global Bundle Adjustment
  m_b_runningGlobalBA = true;
  m_b_finishedGlobalBA = false;
  m_b_stopGlobalBA = false;
  m_sptrThreadGlobalBA = std::make_shared<thread>(&LoopClosing::runGlobalBundleAdjustment,shared_from_this(),m_sptrCurrentLoopKF->m_int_keyFrameID);
  m_sptrLocalMapper->release();
  m_int_lastLoopKFid = m_sptrCurrentLoopKF->m_int_keyFrameID;
}

void searchAndFuse(const sptrKeyFrameAndPose& _correctedPosesMap){
  OrbMatcher matcher(0.8);
  for(const std::pair<std::shared_ptr<KeyFrame>,g2o::Sim3>& poseMap : _correctedPosesMap){
    cv::Mat cvMatScw = Converter::transform_Sim3_cvMat(poseMap.second);
    std::vector<std::shared_ptr<MapPoint>> vsptrReplacePoints(m_v_sptrLoopMapPoints.size(),static_cast<std::shared_ptr<MapPoint>>(nullptr));
    matcher.FuseBySim3(poseMap.first,cvMatScw,m_v_sptrLoopMapPoints,4,vsptrReplacePoints);
    // Get Map Mutex
    std::unique_lock<std::mutex> lock(m_sptrMap->m_mutex_updateMap);
    int ifor = 0;
    for(std::shared_ptr<MapPoint>& sptrLoopMP : m_v_sptrLoopMapPoints){
      if(vsptrReplacePoints[ifor]){
        vsptrReplacePoints[ifor]->beReplacedBy(sptrLoopMP);
      }
      ifor++;
    }
  }
}

void LoopClosing::RequestReset(){
  {
    std::unique_lock<std::mutex> lock(m_mutex_reset);
    m_b_resetRequested = true;
  }
  while(1){
    {
      std::unique_lock<std::mutex> lock2(m_mutex_reset);
      if(!m_b_resetRequested){
        break;
      }
    }
    usleep(5000);
  }
}

void LoopClosing::resetIfRequested(){
  std::unique_lock<std::mutex> lock(m_mutex_reset);
  if(m_b_resetRequested){
    m_l_sptrLoopKeyFrameBufferQueue.clear();
    m_int_lastLoopKFid = 0;
    m_b_resetRequested = false;
  }
}

void LoopClosing::runGlobalBundleAdjustment(unsigned long _nLoopKF){
  std::cout << "Starting Global Bundle Adjustment" << std::endl;
  int idx = m_int_fullBAIdx;
  Optimizer::globalBundleAdjust(m_sptrMap,10,m_b_stopGlobalBA,_nLoopKF,false);
  // Update all MapPoints and KeyFrames
  // Local Mapping was active during BA, that means that there might be new keyframes
  // not included in the Global BA and they are not consistent with the updated map.
  // We need to propagate the correction through the spanning tree
  {
    std::unique_lock<std::mutex> lock(m_mutex_globalBA);
    if(idx != m_int_fullBAIdx){
      return;
    }
    if(!m_b_stopGlobalBA){
      std::cout << "Global Bundle Adjustment finished" << std::endl;
      std::cout << "Updating map ..." << std::endl;
      m_sptrLocalMapper->requestStop();
      // Wait until Local Mapping has effectively stopped
      while(!m_sptrLocalMapper->isStopped() && !m_sptrLocalMapper->isFinished()){
        usleep(1000);
      }
      // Get Map Mutex
      std::unique_lock<std::mutex> lock(m_sptrMap->m_mutex_updateMap);
      // Correct keyframes starting at map first keyframe
      std::list<std::shared_ptr<KeyFrame>> lsptrKF2Check(m_sptrMap->m_v_sptrOriginalKeyFrames.begin(),m_sptrMap->m_v_sptrOriginalKeyFrames.end());
      while(!lsptrKF2Check.empty()){
        std::shared_ptr<KeyFrame> sptrFisrtKF = lsptrKF2Check.front();
        cv::Mat firstKF_Tw2c = sptrFisrtKF->getInverseCameraPoseByTransform_w2c();
        for(const std::shared_ptr<KeyFrame>& sptrChildKF : sptrFisrtKF->getChildren()){
          if(sptrChildKF->m_int_globalBAForKeyFrameID != _nLoopKF){
            sptrChildKF->m_cvMat_T_c2w_GlobalBA = sptrChildKF->getCameraPoseByTransrom_c2w * firstKF_Tw2c * sptrFisrtKF->m_cvMat_T_c2w_GlobalBA;
            sptrChildKF->m_int_globalBAForKeyFrameID = _nLoopKF;
          }
          lsptrKF2Check.push_back(sptrChildKF);
        }
        sptrFisrtKF->m_cvMat_T_c2w_beforeGlobalBA = sptrFisrtKF->getCameraPoseByTransrom_c2w();
        sptrFisrtKF->setCameraPoseByTransform_c2w(sptrFisrtKF->m_cvMat_T_c2w_GlobalBA);
        lsptrKF2Check.pop_front();
      }
      // Correct MapPoints
      for(const std::shared_ptr<MapPoint>& sptrMP : m_sptrMap->getAllMapPoints()){
        if(sptrMP->isBad()){
          continue;
        }
        if(sptrMP->m_int_globalBAforKeyFrameID == _nLoopKF){
          // If optimized by Global BA, just update
          sptrMP->setPosInWorld(sptrMP->m_cvMat_posGlobalBA);
        }else{
          // Update according to the correction of its reference keyframe
          std::shared_ptr<KeyFrame> sptrRefKF = sptrMP->getReferenceKeyFrame();
          if(sptrRefKF->m_int_globalBAForKeyFrameID == _nLoopKF){
            // Map to non-corrected camera
            cv::Mat Rcw = sptrRefKF->m_cvMat_T_c2w_beforeGlobalBA.rowRange(0,3).colRange(0,3);
            cv::Mat tcw = sptrRefKF->m_cvMat_T_c2w_beforeGlobalBA.rowRange(0,3).col(3);
            cv::Mat Xc = Rcw * sptrMP->getPosInWorld() + tcw;
            // Backproject using corrected camera
            cv::Mat Twc = sptrRefKF->getInverseCameraPoseByTransform_w2c();
            cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
            cv::Mat twc = Twc.rowRange(0,3).col(3);
            sptrMP->setPosInWorld(Rwc * Xc + twc);
          }
        }
      }
      m_sptrMap->informNewBigChange();
      m_sptrLocalMapper->release();
      std::cout << "Map updated!" << std::endl;
    }
    m_b_finishedGlobalBA = true;
    m_b_runningGlobalBA = false;
  }
}

} //namespace YDORBSLAM
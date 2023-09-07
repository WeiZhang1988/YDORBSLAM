#include <mutex>
#include <Eigen/StdVector>
#include "converter.hpp"
#include "optimizer.hpp"

namespace YDORBSLAM{
  void Optimizer::bundleAdjust(const std::vector<std::shared_ptr<KeyFrame>> &_vSptrKeyFrames, const std::vector<std::shared_ptr<MapPoint>> &_vSptrMapPoints, const int _iterNum, bool &_bIsStopping, const long int _loopKeyFrameID, const bool _bIsRobust){
    std::vector<bool> vIsMapPointExcluded;
    vIsMapPointExcluded.resize(_vSptrMapPoints.size());
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    //BlockSolver_6_3 means that pose dim (aka. optimized variable num) is 6 and landmark dim (aka. error num) is 3
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    if(&_bIsStopping){
      optimizer.setForceStopFlag(&_bIsStopping);
    }
    long int maxKeyFrameID = 0;
    //set key frame vertices
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : _vSptrKeyFrames){
      if(!sptrKeyFrame->isBad()){
        g2o::VertexSE3Expmap *vertexSE3 = new g2o::VertexSE3Expmap();
        vertexSE3->setEstimate(Converter::transform_cvMat_SE3Quat(sptrKeyFrame->getCameraPoseByTransrom_c2w()));
        vertexSE3->setId(sptrKeyFrame->m_int_keyFrameID);
        vertexSE3->setFixed(sptrKeyFrame->m_int_keyFrameID==0);
        optimizer.addVertex(vertexSE3);
        if(sptrKeyFrame->m_int_keyFrameID>maxKeyFrameID){
          maxKeyFrameID = sptrKeyFrame->m_int_keyFrameID;
        }
      }
    }
    const float monoDelta = sqrt(5.99);
    const float stereoDelta = sqrt(7.815);
    //set map point vertices
    int i_for = 0;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : _vSptrMapPoints){
      if(!sptrMapPoint->isBad()){
        g2o::VertexSBAPointXYZ *vertexPoint = new g2o::VertexSBAPointXYZ();
        vertexPoint->setEstimate(Converter::d3X1Matrix_cvMat_eigen(sptrMapPoint->getPosInWorld()));
        const int ID = sptrMapPoint->m_int_ID+maxKeyFrameID+1;
        vertexPoint->setId(ID);
        vertexPoint->setMarginalized(true);
        optimizer.addVertex(vertexPoint);
        const std::map<std::shared_ptr<KeyFrame>,int> observations = sptrMapPoint->getObservations();
        int edgeNum = 0;
        //set edges
        for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : observations){
          if(!observation.first->isBad() && observation.first->m_int_keyFrameID<=maxKeyFrameID){
            edgeNum++;
            const cv::KeyPoint &keyPoint = observation.first->m_v_keyPoints[observation.second];
            if(observation.first->m_v_rightXcords[observation.second]<0){
              Eigen::Matrix<double,2,1> measure;
              measure<<keyPoint.pt.x, keyPoint.pt.y;
              g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
              edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ID)));
              edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(observation.first->m_int_keyFrameID)));
              edge->setMeasurement(measure);
              const float &invScaleFactorSquares = observation.first->m_v_invScaleFactorSquares[keyPoint.octave];
              edge->setInformation(Eigen::Matrix2d::Identity()*invScaleFactorSquares);
              if(_bIsRobust){
                g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
                robustKernel->setDelta(monoDelta);
                edge->setRobustKernel(robustKernel);
              }
              edge->fx = Frame::m_flt_fx;
              edge->fy = Frame::m_flt_fy;
              edge->cx = Frame::m_flt_cx;
              edge->cy = Frame::m_flt_cy;
              optimizer.addEdge(edge);
            }else {
              Eigen::Matrix<double,3,1> measure;
              measure<<keyPoint.pt.x, keyPoint.pt.y, observation.first->m_v_rightXcords[observation.second];
              g2o::EdgeStereoSE3ProjectXYZ *edge = new g2o::EdgeStereoSE3ProjectXYZ();
              edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ID)));
              edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(observation.first->m_int_keyFrameID)));
              edge->setMeasurement(measure);
              const float &invScaleFactorSquares = observation.first->m_v_invScaleFactorSquares[keyPoint.octave];
              edge->setInformation(Eigen::Matrix3d::Identity()*invScaleFactorSquares);
              if(_bIsRobust){
                g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
                robustKernel->setDelta(stereoDelta);
                edge->setRobustKernel(robustKernel);
              }
              edge->fx = Frame::m_flt_fx;
              edge->fy = Frame::m_flt_fy;
              edge->cx = Frame::m_flt_cx;
              edge->cy = Frame::m_flt_cy;
              edge->bf = Frame::m_flt_baseLineTimesFx;
              optimizer.addEdge(edge);
            }
          }
        }
        if(edgeNum==0){
          optimizer.removeVertex(vertexPoint);
          vIsMapPointExcluded[i_for]=true;
        }
        else{
          vIsMapPointExcluded[i_for]=false;
        }
      }
      i_for++;
    }
    optimizer.initializeOptimization();
    optimizer.optimize(_iterNum);
    //recover optimized data
    //key frames
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : _vSptrKeyFrames){
      if(!sptrKeyFrame->isBad()){
        g2o::VertexSE3Expmap *vertexSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID));
        if(_loopKeyFrameID==0){
          sptrKeyFrame->setCameraPoseByTransrom_c2w(Converter::transform_SE3_cvMat(vertexSE3->estimate()));
        }else{
          sptrKeyFrame->m_cvMat_T_c2w_GBA.create(4,4,CV_32F);
          Converter::transform_SE3_cvMat(vertexSE3->estimate()).copyTo(sptrKeyFrame->m_cvMat_T_c2w_GBA);
          sptrKeyFrame->m_int_globalBAForKeyFrameID = _loopKeyFrameID;
        }
      }
    }
    //map points
    i_for = 0;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : _vSptrMapPoints){
      if(!sptrMapPoint->isBad() && !vIsMapPointExcluded[i_for]){
        g2o::VertexSBAPointXYZ *vertexPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(sptrMapPoint->m_int_ID+maxKeyFrameID+1));
        if(_loopKeyFrameID==0){
          sptrMapPoint->setPosInWorld(Converter::d3X1Matrix_eigen_cvMat(vertexPoint->estimate()));
          sptrMapPoint->updateNormalAndDepth();
        }else{
          sptrMapPoint->m_cvMat_posGlobalBA.create(3,1,CV_32F);
          Converter::d3X1Matrix_eigen_cvMat(vertexPoint->estimate()).copyTo(sptrMapPoint->m_cvMat_posGlobalBA);
          sptrMapPoint->m_int_globalBAforKeyFrameID = _loopKeyFrameID;
        }
      }
      i_for++;
    }
  }
  void Optimizer::localBundleAdjust(std::shared_ptr<KeyFrame> _sptrKeyFrame, std::shared_ptr<Map> _sptrMap, bool &_bIsStopping){
    //local key frames: first breath search from current key frame
    std::list<std::shared_ptr<KeyFrame>> localKeyFrameList;
    localKeyFrameList.push_back(_sptrKeyFrame);
    _sptrKeyFrame->m_int_localBAForKeyFrameID = _sptrKeyFrame->m_int_keyFrameID;
    const std::vector<std::shared_ptr<KeyFrame>> vConnectedKeyFrames = _sptrKeyFrame->getOrderedConnectedKeyFrames();
    for(const std::shared_ptr<KeyFrame> &sptrConnectedKeyFrame : vConnectedKeyFrames){
      sptrConnectedKeyFrame->m_int_localBAForKeyFrameID = _sptrKeyFrame->m_int_keyFrameID;
      if(!sptrConnectedKeyFrame->isBad()){
        localKeyFrameList.push_back(sptrConnectedKeyFrame);
      }
    }
    //local map points seen in local key frames
    std::list<std::shared_ptr<MapPoint>> localMapPointList;
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : localKeyFrameList){
      const std::vector<std::shared_ptr<MapPoint>> vSptrMapPoints = sptrKeyFrame->getMatchedMapPointsVec();
      for(const std::shared_ptr<MapPoint> &sptrMapPoint : vSptrMapPoints){
        if(sptrMapPoint && !sptrMapPoint->isBad() && sptrMapPoint->m_int_localBAForKeyFrameID!=_sptrKeyFrame->m_int_keyFrameID){
          localMapPointList.push_back(sptrMapPoint);
          sptrMapPoint->m_int_localBAForKeyFrameID = _sptrKeyFrame->m_int_keyFrameID;
        }
      }
    }
    //fixed key frames. key frames that see local map points but not local key frames
    std::list<std::shared_ptr<KeyFrame>> fixedKeyFrameList;
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : localMapPointList){
      const std::map<std::shared_ptr<KeyFrame>,int> observations = sptrMapPoint->getObservations();
      for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : observations){
        if(observation.first->m_int_localBAForKeyFrameID!=_sptrKeyFrame->m_int_keyFrameID && observation.first->m_int_fixedBAForKeyFrameID!=_sptrKeyFrame->m_int_keyFrameID){
          observation.first->m_int_fixedBAForKeyFrameID = _sptrKeyFrame->m_int_keyFrameID;
          if(!observation.first->isBad()){
            fixedKeyFrameList.push_back(observation.first);
          }
        }
      }
    }
    //set up optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    if(&_bIsStopping){
      optimizer.setForceStopFlag(&_bIsStopping);
    }
    long int maxKeyFrameID = 0;
    //set local key frame vertices
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : localKeyFrameList){
      if(!sptrKeyFrame->isBad()){
        g2o::VertexSE3Expmap *vertexSE3 = new g2o::VertexSE3Expmap();
        vertexSE3->setEstimate(Converter::transform_cvMat_SE3Quat(sptrKeyFrame->getCameraPoseByTransrom_c2w()));
        vertexSE3->setId(sptrKeyFrame->m_int_keyFrameID);
        vertexSE3->setFixed(sptrKeyFrame->m_int_keyFrameID==0);
        optimizer.addVertex(vertexSE3);
        if(sptrKeyFrame->m_int_keyFrameID>maxKeyFrameID){
          maxKeyFrameID = sptrKeyFrame->m_int_keyFrameID;
        }
      }
    }
    //set fixed key frame vertices
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : fixedKeyFrameList){
      g2o::VertexSE3Expmap *vertexSE3 = new g2o::VertexSE3Expmap();
      vertexSE3->setEstimate(Converter::transform_cvMat_SE3Quat(sptrKeyFrame->getCameraPoseByTransrom_c2w()));
      vertexSE3->setId(sptrKeyFrame->m_int_keyFrameID);
      vertexSE3->setFixed(true);
      optimizer.addVertex(vertexSE3);
      if(sptrKeyFrame->m_int_keyFrameID>maxKeyFrameID){
        maxKeyFrameID = sptrKeyFrame->m_int_keyFrameID;
      }
    }
    //set map point vertices
    const int expectedEdgeNum = (localKeyFrameList.size()+fixedKeyFrameList.size())*localMapPointList.size();
    std::vector<g2o::EdgeSE3ProjectXYZ *> vMonoEdges;
    vMonoEdges.reserve(expectedEdgeNum);
    std::vector<std::shared_ptr<KeyFrame>> vMonoEdgeKeyFrames;
    vMonoEdgeKeyFrames.reserve(expectedEdgeNum);
    std::vector<std::shared_ptr<MapPoint>> vMonoEdgeMapPoints;
    vMonoEdgeMapPoints.reserve(expectedEdgeNum);
    std::vector<g2o::EdgeStereoSE3ProjectXYZ *> vStereoEdges;
    vStereoEdges.reserve(expectedEdgeNum);
    std::vector<std::shared_ptr<KeyFrame>> vStereoEdgeKeyFrames;
    vStereoEdgeKeyFrames.reserve(expectedEdgeNum);
    std::vector<std::shared_ptr<MapPoint>> vStereoEdgeMapPoints;
    vStereoEdgeMapPoints.reserve(expectedEdgeNum);
    const float monoDelta = sqrt(5.991);
    const float stereoDelta = sqrt(7.815);
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : localMapPointList){
      g2o::VertexSBAPointXYZ *vertexPoint = new g2o::VertexSBAPointXYZ();
      vertexPoint->setEstimate(Converter::d3X1Matrix_cvMat_eigen(sptrMapPoint->getPosInWorld()));
      const int ID = sptrMapPoint->m_int_ID+maxKeyFrameID+1;
      vertexPoint->setId(ID);
      vertexPoint->setMarginalized(true);
      optimizer.addVertex(vertexPoint);
      const std::map<std::shared_ptr<KeyFrame>,int> observations = sptrMapPoint->getObservations();
      int edgeNum = 0;
      //set edges
      for(const std::pair<std::shared_ptr<KeyFrame>,int> &observation : observations){
        if(!observation.first->isBad() && observation.first->m_int_keyFrameID<=maxKeyFrameID){
          edgeNum++;
          const cv::KeyPoint &keyPoint = observation.first->m_v_keyPoints[observation.second];
          if(observation.first->m_v_rightXcords[observation.second]<0){
            Eigen::Matrix<double,2,1> measure;
            measure<<keyPoint.pt.x, keyPoint.pt.y;
            g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ID)));
            edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(observation.first->m_int_keyFrameID)));
            edge->setMeasurement(measure);
            const float &invScaleFactorSquares = observation.first->m_v_invScaleFactorSquares[keyPoint.octave];
            edge->setInformation(Eigen::Matrix2d::Identity()*invScaleFactorSquares);
            g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
            robustKernel->setDelta(monoDelta);
            edge->setRobustKernel(robustKernel);
            edge->fx = Frame::m_flt_fx;
            edge->fy = Frame::m_flt_fy;
            edge->cx = Frame::m_flt_cx;
            edge->cy = Frame::m_flt_cy;
            optimizer.addEdge(edge);
            vMonoEdges.push_back(edge);
            vMonoEdgeKeyFrames.push_back(observation.first);
            vMonoEdgeMapPoints.push_back(sptrMapPoint);
          }else {
            Eigen::Matrix<double,3,1> measure;
            measure<<keyPoint.pt.x, keyPoint.pt.y, observation.first->m_v_rightXcords[observation.second];
            g2o::EdgeStereoSE3ProjectXYZ *edge = new g2o::EdgeStereoSE3ProjectXYZ();
            edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ID)));
            edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(observation.first->m_int_keyFrameID)));
            edge->setMeasurement(measure);
            const float &invScaleFactorSquares = observation.first->m_v_invScaleFactorSquares[keyPoint.octave];
            edge->setInformation(Eigen::Matrix3d::Identity()*invScaleFactorSquares);
            g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
            robustKernel->setDelta(stereoDelta);
            edge->setRobustKernel(robustKernel);
            edge->fx = Frame::m_flt_fx;
            edge->fy = Frame::m_flt_fy;
            edge->cx = Frame::m_flt_cx;
            edge->cy = Frame::m_flt_cy;
            edge->bf = Frame::m_flt_baseLineTimesFx;
            optimizer.addEdge(edge);
            vStereoEdges.push_back(edge);
            vStereoEdgeKeyFrames.push_back(observation.first);
            vStereoEdgeMapPoints.push_back(sptrMapPoint);
          }
        }
      }
    }
    if(&_bIsStopping && _bIsStopping){
      return;
    }
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    bool isToOptimizeWithOutOutliers = true;
    if(!&_bIsStopping || !_bIsStopping){
      //check inlier observations
      for(int i=0;i<vMonoEdges.size();i++){
        g2o::EdgeSE3ProjectXYZ *edge = vMonoEdges[i];
        std::shared_ptr<MapPoint> sptrMapPoint = vMonoEdgeMapPoints[i];
        if(!sptrMapPoint->isBad()){
          if(edge->chi2()>5.991 || !edge->isDepthPositive()){
            edge->setLevel(1);
          }
          edge->setRobustKernel(0);
        }
      }
      for(int i=0;i<vStereoEdges.size();i++){
        g2o::EdgeStereoSE3ProjectXYZ *edge = vStereoEdges[i];
        std::shared_ptr<MapPoint> sptrMapPoint = vStereoEdgeMapPoints[i];
        if(!sptrMapPoint->isBad()){
          if(edge->chi2()>7.815 || !edge->isDepthPositive()){
            edge->setLevel(1);
          }
          edge->setRobustKernel(0);
        }
      }
      //optimize again without the outliers
      optimizer.initializeOptimization(0);
      optimizer.optimize(10);
    }
    std::vector<std::pair<std::shared_ptr<KeyFrame>,std::shared_ptr<MapPoint>>> vToBeErased;
    vToBeErased.reserve(vMonoEdges.size()+vStereoEdges.size());
    //check inlier observations
    for(int i=0;i<vMonoEdges.size();i++){
      g2o::EdgeSE3ProjectXYZ *edge = vMonoEdges[i];
      std::shared_ptr<MapPoint> sptrMapPoint = vMonoEdgeMapPoints[i];
      if(!sptrMapPoint->isBad() && (edge->chi2()>5.991 || !edge->isDepthPositive())){
        std::shared_ptr<KeyFrame> sptrKeyFrame = vMonoEdgeKeyFrames[i];
        vToBeErased.push_back(std::make_pair(sptrKeyFrame,sptrMapPoint));
      }
    }
    for(int i=0;i<vStereoEdges.size();i++){
      g2o::EdgeStereoSE3ProjectXYZ *edge = vStereoEdges[i];
      std::shared_ptr<MapPoint> sptrMapPoint = vStereoEdgeMapPoints[i];
      if(!sptrMapPoint->isBad() && (edge->chi2()>7.815 || !edge->isDepthPositive())){
        std::shared_ptr<KeyFrame> sptrKeyFrame = vStereoEdgeKeyFrames[i];
        vToBeErased.push_back(std::make_pair(sptrKeyFrame,sptrMapPoint));
      }
    }
    //get map mutex
    std::unique_lock<std::mutex> lock(_sptrMap->m_mutex_updateMap);
    for(std::pair<std::shared_ptr<KeyFrame>,std::shared_ptr<MapPoint>> &pair : vToBeErased){
      pair.first->eraseMatchedMapPoint(pair.second);
      pair.second->eraseObservation(pair.first);
    }
    //recover optimized key frames data
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : localKeyFrameList){
      g2o::VertexSE3Expmap *vertexSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID));
      sptrKeyFrame->setCameraPoseByTransrom_c2w(Converter::transform_SE3_cvMat(vertexSE3->estimate()));
    }
    //recover optimized map points data
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : localMapPointList){
      g2o::VertexSBAPointXYZ *vertexPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(sptrMapPoint->m_int_ID+maxKeyFrameID+1));
      sptrMapPoint->setPosInWorld(Converter::d3X1Matrix_eigen_cvMat(vertexPoint->estimate()));
      sptrMapPoint->updateNormalAndDepth();
    }
  }
  void Optimizer::globalBundleAdjust(std::shared_ptr<Map> _sptrMap, const int _iterNum, bool &_bIsStopping, const long int _loopKeyFrameID, const bool _bIsRobust){
    const std::vector<std::shared_ptr<KeyFrame>> vSptrKeyFrames = _sptrMap->getAllKeyFrames();
    const std::vector<std::shared_ptr<MapPoint>> vSptrMapPoints = _sptrMap->getAllMapPoints();
    bundleAdjust(vSptrKeyFrames,vSptrMapPoints,_iterNum,_bIsStopping,_loopKeyFrameID,_bIsRobust);
  }
  int Optimizer::optimizePose(std::shared_ptr<Frame> _sptrFrame){
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    int initialCorrespondenceNum = 0;
    //set frame vertex
    g2o::VertexSE3Expmap *vertexSE3 = new g2o::VertexSE3Expmap();
    vertexSE3->setEstimate(Converter::transform_cvMat_SE3Quat(_sptrFrame->getCameraPoseByTransrom_c2w()));
    vertexSE3->setId(0);
    vertexSE3->setFixed(false);
    optimizer.addVertex(vertexSE3);
    //set map points vertices
    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vMonoEdges;
    std::vector<int> vMonoEdgeIndices;
    vMonoEdges.reserve(_sptrFrame->m_int_keyPointsNum);
    vMonoEdgeIndices.reserve(_sptrFrame->m_int_keyPointsNum);
    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vStereoEdges;
    std::vector<int> vStereoEdgeIndices;
    vStereoEdges.reserve(_sptrFrame->m_int_keyPointsNum);
    vStereoEdgeIndices.reserve(_sptrFrame->m_int_keyPointsNum);
    const float monoDelta = sqrt(5.991);
    const float stereoDelta = sqrt(7.815);
    {
      std::unique_lock<std::mutex> lock(MapPoint::m_mutex_global);
      for(int i=0;i<_sptrFrame->m_int_keyPointsNum;i++){
        const cv::KeyPoint &keyPoint = _sptrFrame->m_v_keyPoints[i];
        if(_sptrFrame->m_v_sptrMapPoints[i]){
          initialCorrespondenceNum++;
          _sptrFrame->m_v_isOutliers[i] = false;
          if(_sptrFrame->m_v_rightXcords[i]<0){
            Eigen::Matrix<double,2,1> measure;
            measure<<keyPoint.pt.x, keyPoint.pt.y;
            g2o::EdgeSE3ProjectXYZOnlyPose *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            edge->setMeasurement(measure);
            const float &invScaleFactorSquares =_sptrFrame->m_v_invScaleFactorSquares[keyPoint.octave];
            edge->setInformation(Eigen::Matrix2d::Identity()*invScaleFactorSquares);
            g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
            robustKernel->setDelta(monoDelta);
            edge->setRobustKernel(robustKernel);
            edge->fx = Frame::m_flt_fx;
            edge->fy = Frame::m_flt_fy;
            edge->cx = Frame::m_flt_cx;
            edge->cy = Frame::m_flt_cy;
            cv::Mat Xw = _sptrFrame->m_v_sptrMapPoints[i]->getPosInWorld();
            edge->Xw[0] = Xw.at<float>(0);
            edge->Xw[1] = Xw.at<float>(1);
            edge->Xw[2] = Xw.at<float>(2);
            optimizer.addEdge(edge);
            vMonoEdges.push_back(edge);
            vMonoEdgeIndices.push_back(i);
          }else{
            Eigen::Matrix<double,3,1> measure;
            measure<<keyPoint.pt.x, keyPoint.pt.y, _sptrFrame->m_v_rightXcords[i];
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            edge->setMeasurement(measure);
            const float &invScaleFactorSquares =_sptrFrame->m_v_invScaleFactorSquares[keyPoint.octave];
            edge->setInformation(Eigen::Matrix3d::Identity()*invScaleFactorSquares);
            g2o::RobustKernelHuber *robustKernel = new g2o::RobustKernelHuber();
            robustKernel->setDelta(stereoDelta);
            edge->setRobustKernel(robustKernel);
            edge->fx = Frame::m_flt_fx;
            edge->fy = Frame::m_flt_fy;
            edge->cx = Frame::m_flt_cx;
            edge->cy = Frame::m_flt_cy;
            edge->bf = Frame::m_flt_baseLineTimesFx;
            cv::Mat Xw = _sptrFrame->m_v_sptrMapPoints[i]->getPosInWorld();
            edge->Xw[0] = Xw.at<float>(0);
            edge->Xw[1] = Xw.at<float>(1);
            edge->Xw[2] = Xw.at<float>(2);
            optimizer.addEdge(edge);
            vStereoEdges.push_back(edge);
            vStereoEdgeIndices.push_back(i);
          }
        }
      }
    }
    if(initialCorrespondenceNum<3){
      return 0;
    }
    const int episodeNum = 4;
    //episodeNum=4 episodes of optimizations are performed, each episode performs 10 iterations
    //observations are classified as inlier or outlier after each episode
    //outliers are excluded in the next episode, but can be classified as inlier at the end
    const std::vector<float> monoChi2(episodeNum,5.991);
    const std::vector<float> stereoChi2(episodeNum,7.815);
    const std::vector<int> iterNums(episodeNum,10);
    int badNum=0;
    for(int epi=0;epi<episodeNum;epi++){
      vertexSE3->setEstimate(Converter::transform_cvMat_SE3Quat(_sptrFrame->getCameraPoseByTransrom_c2w()));
      optimizer.initializeOptimization(0);
      optimizer.optimize(iterNums[epi]);
      badNum=0;
      for(int i=0;i<vMonoEdges.size();i++){
        g2o::EdgeSE3ProjectXYZOnlyPose *edge = vMonoEdges[i];
        const int idx = vMonoEdgeIndices[i];
        if(_sptrFrame->m_v_isOutliers[idx]){
          edge->computeError();
        }
        const float chi2 = edge->chi2();
        if(chi2>monoChi2[epi]){
          _sptrFrame->m_v_isOutliers[idx] = true;
          edge->setLevel(1);
          badNum++;
        }else{
          _sptrFrame->m_v_isOutliers[idx] = false;
          edge->setLevel(0);
        }
        if(epi==2){
          edge->setRobustKernel(0);
        }
      }
      for(int i=0;i<vStereoEdges.size();i++){
        g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = vStereoEdges[i];
        const int idx = vStereoEdgeIndices[i];
        if(_sptrFrame->m_v_isOutliers[idx]){
          edge->computeError();
        }
        const float chi2 = edge->chi2();
        if(chi2>stereoChi2[epi]){
          _sptrFrame->m_v_isOutliers[idx]=true;
          edge->setLevel(1);
          badNum++;
        }else {
          edge->setLevel(0);
          _sptrFrame->m_v_isOutliers[idx]=false;
        }
        if(epi==2){
          edge->setRobustKernel(0);
        }
      }
      if(optimizer.edges().size()<10){
        break;
      }
    }
    //recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *recoveredVertexSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    _sptrFrame->setCameraPoseByTransrom_c2w(Converter::transform_SE3_cvMat(recoveredVertexSE3->estimate()));
    return initialCorrespondenceNum - badNum;
  }
  void Optimizer::optimizeEssentialGraph(std::shared_ptr<Map> _sptrMap, std::shared_ptr<KeyFrame> _sptrLoopKeyFrame, std::shared_ptr<KeyFrame> _sptrCurrentKeyFrame, const LoopClosing::KeyFrameAndPose &_inCorrectedSim3, const LoopClosing::KeyFrameAndPose &_correctedSim3, const std::map<std::shared_ptr<KeyFrame>,std::set<std::shared_ptr<KeyFrame>>> &_loopConnections, const bool &_bIsScaleFixed){
    //setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    //BlockSolver_7_3 means that essential matrix dim of freedom (aka. optimized variable num) is 7 and landmark dim (aka. error num) is 3
    typedef g2o::BlockSolver_7_3 BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);
    const std::vector<std::shared_ptr<KeyFrame>> vSptrKeyFrames = _sptrMap->getAllKeyFrames();
    const std::vector<std::shared_ptr<MapPoint>> vSptrMapPoints = _sptrMap->getAllMapPoints();
    const long int maxKeyFrameID = _sptrMap->getMaxKeyFrameID();
    std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>> vSims_c2w(maxKeyFrameID+1);
    std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSims_w2c(maxKeyFrameID+1);
    std::vector<g2o::VertexSim3Expmap *> vVertices(maxKeyFrameID+1);
    const int minKeyPointNum = 100;
    //set key frame vertices
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : vSptrKeyFrames){
      if(!sptrKeyFrame->isBad()){
        g2o::VertexSim3Expmap *vertexSim3 = new g2o::VertexSim3Expmap();
        LoopClosing::KeyFrameAndPose::const_iterator iter = _correctedSim3.find(sptrKeyFrame);
        if(iter!=_correctedSim3.end()){
          vSims_c2w[sptrKeyFrame->m_int_keyFrameID] = iter->second;
          vertexSim3->setEstimate(iter->second);
        }else{
          Eigen::Matrix<double,3,3> rotation_c2w = Converter::d3X3Matrix_cvMat_eigen(sptrKeyFrame->getRotation_c2w());
          Eigen::Matrix<double,3,1> translation_c2w = Converter::d3X1Matrix_cvMat_eigen(sptrKeyFrame->getTranslation_c2w());
          g2o::Sim3 similarity_c2w(rotation_c2w,translation_c2w,1.0);
          vSims_c2w[sptrKeyFrame->m_int_keyFrameID] = similarity_c2w;
          vertexSim3->setEstimate(similarity_c2w);
        }
        if(sptrKeyFrame==_sptrLoopKeyFrame){
          vertexSim3->setFixed(true);
        }
        vertexSim3->setId(sptrKeyFrame->m_int_keyFrameID);
        vertexSim3->setMarginalized(false);
        vertexSim3->_fix_scale = _bIsScaleFixed;
        optimizer.addVertex(vertexSim3);
        vVertices[sptrKeyFrame->m_int_keyFrameID]=vertexSim3;
      }
    }
    std::set<std::pair<long int, long int>> setInsertedEdges;
    const Eigen::Matrix<double,7,7> informationMatrix = Eigen::Matrix<double,7,7>::Identity();
    //set loop edges
    for(const std::pair<std::shared_ptr<KeyFrame>,std::set<std::shared_ptr<KeyFrame>>> &pair : _loopConnections){
      const g2o::Sim3 firstSimilarity_w2c = vSims_c2w[pair.first->m_int_keyFrameID].inverse();
      for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : pair.second){
        if((pair.first->m_int_keyFrameID==_sptrCurrentKeyFrame->m_int_keyFrameID && sptrKeyFrame->m_int_keyFrameID==_sptrLoopKeyFrame->m_int_keyFrameID) || \
        (pair.first->getWeight(sptrKeyFrame)>=minKeyPointNum)){
          const g2o::Sim3 measure = vSims_c2w[sptrKeyFrame->m_int_keyFrameID] * firstSimilarity_w2c;
          g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
          edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pair.first->m_int_keyFrameID)));
          edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID)));
          edge->setMeasurement(measure);
          edge->setInformation(informationMatrix);
          optimizer.addEdge(edge);
          setInsertedEdges.insert(std::make_pair(std::min(pair.first->m_int_keyFrameID,sptrKeyFrame->m_int_keyFrameID),std::max(pair.first->m_int_keyFrameID,sptrKeyFrame->m_int_keyFrameID)));
        }
      }
    }
    //set normal edges
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : vSptrKeyFrames){
      g2o::Sim3 firstSimilarity_w2c;
      if(_inCorrectedSim3.find(sptrKeyFrame)!=_inCorrectedSim3.end()){
        firstSimilarity_w2c = _inCorrectedSim3.find(sptrKeyFrame)->second.inverse();
      }else {
        firstSimilarity_w2c = vSims_c2w[sptrKeyFrame->m_int_keyFrameID].inverse();
      }
      std::shared_ptr<KeyFrame> sptrParentKeyFrame = sptrKeyFrame->getParent();
      //spanning tree edge
      if(sptrParentKeyFrame){
        g2o::Sim3 secondSimilarity_c2w;
        if(_inCorrectedSim3.find(sptrParentKeyFrame)!=_inCorrectedSim3.end()){
          secondSimilarity_c2w = _inCorrectedSim3.find(sptrParentKeyFrame)->second;
        }else {
          secondSimilarity_c2w = vSims_c2w[sptrParentKeyFrame->m_int_keyFrameID];
        }
        g2o::Sim3 measure = secondSimilarity_c2w * firstSimilarity_w2c;
        g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID)));
        edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrParentKeyFrame->m_int_keyFrameID)));
        edge->setMeasurement(measure);
        edge->setInformation(informationMatrix);
        optimizer.addEdge(edge);
      }
      //loop edges
      const std::set<std::shared_ptr<KeyFrame>> setLoopEdges = sptrKeyFrame->getLoopEdges();
      for(const std::shared_ptr<KeyFrame> &sptrLoopKeyFrame : setLoopEdges){
        if(sptrLoopKeyFrame->m_int_keyFrameID<sptrKeyFrame->m_int_keyFrameID){
          g2o::Sim3 secondSimilarity_c2w;
          if(_inCorrectedSim3.find(sptrLoopKeyFrame)!=_inCorrectedSim3.end()){
            secondSimilarity_c2w = _inCorrectedSim3.find(sptrLoopKeyFrame)->second;
          }else {
            secondSimilarity_c2w = vSims_c2w[sptrKeyFrame->m_int_keyFrameID];
          }
          g2o::Sim3 measure = secondSimilarity_c2w * firstSimilarity_w2c;
          g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
          edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID)));
          edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrLoopKeyFrame->m_int_keyFrameID)));
          edge->setMeasurement(measure);
          edge->setInformation(informationMatrix);
          optimizer.addEdge(edge);
        }
      }
      //covisibility graph edges
      const std::vector<std::shared_ptr<KeyFrame>> vConnectedKeyFrames = sptrKeyFrame->getOrderedConnectedKeyFramesLargerThanWeight(minKeyPointNum);
      for(const std::shared_ptr<KeyFrame> &sptrConnectedKeyFrame : vConnectedKeyFrames){
        if(sptrConnectedKeyFrame && !sptrConnectedKeyFrame->isBad() && \
        sptrConnectedKeyFrame!=sptrParentKeyFrame && !sptrKeyFrame->hasChild(sptrConnectedKeyFrame) && \
        !setLoopEdges.count(sptrConnectedKeyFrame) && sptrConnectedKeyFrame->m_int_keyFrameID<sptrKeyFrame->m_int_keyFrameID && \
        !setInsertedEdges.count(std::make_pair(std::min(sptrKeyFrame->m_int_keyFrameID,sptrConnectedKeyFrame->m_int_keyFrameID),std::max(sptrKeyFrame->m_int_keyFrameID,sptrConnectedKeyFrame->m_int_keyFrameID)))){
          g2o::Sim3 secondSimilarity_c2w;
          if(_inCorrectedSim3.find(sptrConnectedKeyFrame)!=_inCorrectedSim3.end()){
            secondSimilarity_c2w = _inCorrectedSim3.find(sptrConnectedKeyFrame)->second;
          }else {
            secondSimilarity_c2w = vSims_c2w[sptrConnectedKeyFrame->m_int_keyFrameID];
          }
          g2o::Sim3 measure = secondSimilarity_c2w * firstSimilarity_w2c;
          g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
          edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID)));
          edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(sptrConnectedKeyFrame->m_int_keyFrameID)));
          edge->setMeasurement(measure);
          edge->setInformation(informationMatrix);
          optimizer.addEdge(edge);
        }
      }
    }
    //optimize
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    std::unique_lock<std::mutex> lock(_sptrMap->m_mutex_updateMap);
    //SE3 Pose recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1] Should the t be divided by s? 
    for(const std::shared_ptr<KeyFrame> &sptrKeyFrame : vSptrKeyFrames){
      g2o::VertexSim3Expmap *vertexSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(sptrKeyFrame->m_int_keyFrameID));
      g2o::Sim3 correctedSim3_c2w = vertexSim3->estimate();
      vCorrectedSims_w2c[sptrKeyFrame->m_int_keyFrameID] = correctedSim3_c2w.inverse();
      Eigen::Matrix3d eigen_rotation_c2w = correctedSim3_c2w.rotation().toRotationMatrix();
      Eigen::Vector3d eigen_translation_c2w =correctedSim3_c2w.translation();
      double scale_c2w = correctedSim3_c2w.scale();
      sptrKeyFrame->setCameraPoseByTransrom_c2w(Converter::transform_eigen_cvMat(eigen_rotation_c2w,eigen_translation_c2w / scale_c2w));
    }
    //correct points. transform to "non-optimized" reference key frame pose and transform back with optimized pose
    for(const std::shared_ptr<MapPoint> &sptrMapPoint : vSptrMapPoints){
      if(sptrMapPoint && !sptrMapPoint->isBad()){
        int referenceKeyFrameID = -1;
        if(sptrMapPoint->m_int_correctedByKeyFrameID == _sptrCurrentKeyFrame->m_int_keyFrameID){
          referenceKeyFrameID = sptrMapPoint->m_int_correctedRefKeyFrameID;
        }else {
          referenceKeyFrameID = sptrMapPoint->getReferenceKeyFrame()->m_int_keyFrameID;
        }
        g2o::Sim3 referenceKeyFrameSim3_c2w = vSims_c2w[referenceKeyFrameID];
        g2o::Sim3 correctedReferenceKeyFrameSim3_w2c = vCorrectedSims_w2c[referenceKeyFrameID];
        Eigen::Matrix<double,3,1> eigen_mapPointPosInWorld = Converter::d3X1Matrix_cvMat_eigen(sptrMapPoint->getPosInWorld());
        Eigen::Matrix<double,3,1> eigen_correctedMapPointPosInWorld = correctedReferenceKeyFrameSim3_w2c.map(referenceKeyFrameSim3_c2w.map(eigen_mapPointPosInWorld));
        sptrMapPoint->setPosInWorld(Converter::d3X1Matrix_eigen_cvMat(eigen_correctedMapPointPosInWorld));
        sptrMapPoint->updateNormalAndDepth();
      }
    }
  }
  int Optimizer::optimizeSim3(std::shared_ptr<KeyFrame> _sptrFirstKeyFrame, std::shared_ptr<KeyFrame> _sptrSecondKeyFrame, std::vector<std::shared_ptr<MapPoint>> &_vSptrMatchedMapPoints, g2o::Sim3 &_g2oSim3_first2second, const float _thd, const bool _bIsScaleFixed){
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    //camera poses
    const cv::Mat firstKeyFrameRotation_c2w = _sptrFirstKeyFrame->getRotation_c2w();
    const cv::Mat firstKeyFrameTranslation_c2w = _sptrFirstKeyFrame->getTranslation_c2w();
    const cv::Mat secondKeyFrameRotation_c2w = _sptrSecondKeyFrame->getRotation_c2w();
    const cv::Mat secondKeyFrameTranslation_c2w = _sptrSecondKeyFrame->getTranslation_c2w();
    //set Sim3 Vertex
    g2o::VertexSim3Expmap *vertexSim3 = new g2o::VertexSim3Expmap();
    vertexSim3->_fix_scale = _bIsScaleFixed;
    vertexSim3->setEstimate(_g2oSim3_first2second);
    vertexSim3->setId(0);
    vertexSim3->setFixed(false);
    vertexSim3->_principle_point1[0] = Frame::m_cvMat_intParMat.at<float>(0,2);
    vertexSim3->_principle_point1[1] = Frame::m_cvMat_intParMat.at<float>(1,2);
    vertexSim3->_focal_length1[0] = Frame::m_cvMat_intParMat.at<float>(0,0);
    vertexSim3->_focal_length1[1] = Frame::m_cvMat_intParMat.at<float>(1,1);
    vertexSim3->_principle_point2[0] = Frame::m_cvMat_intParMat.at<float>(0,2);
    vertexSim3->_principle_point2[1] = Frame::m_cvMat_intParMat.at<float>(1,2);
    vertexSim3->_focal_length2[0] = Frame::m_cvMat_intParMat.at<float>(0,0);
    vertexSim3->_focal_length2[1] = Frame::m_cvMat_intParMat.at<float>(1,1);
    optimizer.addVertex(vertexSim3);
    //set map point vertices
    const std::vector<std::shared_ptr<MapPoint>> vSptrFirstKeyFrameMatchedMapPoints = _sptrFirstKeyFrame->getMatchedMapPointsVec();
    std::vector<g2o::EdgeSim3ProjectXYZ *> vSptrEdges_first2second;
    std::vector<g2o::EdgeInverseSim3ProjectXYZ *> vSptrEdges_second2first;
    std::vector<int> vEdgeIndices;
    int matchedMapPointNum = _vSptrMatchedMapPoints.size();
    vSptrEdges_first2second.reserve(2*matchedMapPointNum);
    vSptrEdges_second2first.reserve(2*matchedMapPointNum);
    vEdgeIndices.reserve(2*matchedMapPointNum);
    const float delta = sqrt(_thd);
    int correspondenceNum = 0;
    for(int i=0;i<matchedMapPointNum;i++){
      if(_vSptrMatchedMapPoints[i]){
        const int idx_odd   = 2*i+1;
        const int idx_even  = 2*(i+1);
        const int secondKeyFrameIdx = vSptrFirstKeyFrameMatchedMapPoints[i]->getIdxInKeyFrame(_sptrSecondKeyFrame);
        if(_vSptrMatchedMapPoints[i] && \
        vSptrFirstKeyFrameMatchedMapPoints[i] && \
        !_vSptrMatchedMapPoints[i]->isBad() && \
        !vSptrFirstKeyFrameMatchedMapPoints[i]->isBad() && \
        secondKeyFrameIdx>=0){
          g2o::VertexSBAPointXYZ *vertexFirstMapPoint = new g2o::VertexSBAPointXYZ();
          cv::Mat firstMapPointPosInWorld = vSptrFirstKeyFrameMatchedMapPoints[i]->getPosInWorld();
          cv::Mat firstMapPointPosInCamera = firstKeyFrameRotation_c2w * firstMapPointPosInWorld + firstKeyFrameTranslation_c2w;
          vertexFirstMapPoint->setEstimate(Converter::d3X1Matrix_cvMat_eigen(firstMapPointPosInCamera));
          vertexFirstMapPoint->setId(idx_odd);
          vertexFirstMapPoint->setFixed(true);
          optimizer.addVertex(vertexFirstMapPoint);
          g2o::VertexSBAPointXYZ *vertexSecondMapPoint = new g2o::VertexSBAPointXYZ();
          cv::Mat secondMapPointPosInWorld = _vSptrMatchedMapPoints[i]->getPosInWorld();
          cv::Mat secondMapPointPosInCamera = secondKeyFrameRotation_c2w * secondMapPointPosInWorld + secondKeyFrameTranslation_c2w;
          vertexSecondMapPoint->setEstimate(Converter::d3X1Matrix_cvMat_eigen(secondMapPointPosInCamera));
          vertexSecondMapPoint->setId(idx_even);
          vertexSecondMapPoint->setFixed(true);
          optimizer.addVertex(vertexSecondMapPoint);
          correspondenceNum++;
          //set edge first = Sim3_first2second * second
          Eigen::Matrix<double,2,1> firstMeasure;
          firstMeasure<<_sptrFirstKeyFrame->m_v_keyPoints[i].pt.x, _sptrFirstKeyFrame->m_v_keyPoints[i].pt.y;
          g2o::EdgeSim3ProjectXYZ *edge_first2second = new g2o::EdgeSim3ProjectXYZ();
          edge_first2second->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
          edge_first2second->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx_even)));
          edge_first2second->setMeasurement(firstMeasure);
          edge_first2second->setInformation(Eigen::Matrix2d::Identity()*_sptrFirstKeyFrame->m_v_invScaleFactorSquares[_sptrFirstKeyFrame->m_v_keyPoints[i].octave]);
          g2o::RobustKernelHuber *firstRobustKernel = new g2o::RobustKernelHuber();
          firstRobustKernel->setDelta(delta);
          edge_first2second->setRobustKernel(firstRobustKernel);
          optimizer.addEdge(edge_first2second);
          //set edge second = Sim3_second2first * first
          Eigen::Matrix<double,2,1> secondMeasure;
          secondMeasure<<_sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].pt.x, _sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].pt.y;
          g2o::EdgeInverseSim3ProjectXYZ *edge_second2first = new g2o::EdgeInverseSim3ProjectXYZ();
          edge_second2first->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
          edge_second2first->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx_odd)));
          edge_second2first->setMeasurement(secondMeasure);
          edge_second2first->setInformation(Eigen::Matrix2d::Identity()*_sptrSecondKeyFrame->m_v_invScaleFactorSquares[_sptrSecondKeyFrame->m_v_keyPoints[secondKeyFrameIdx].octave]);
          g2o::RobustKernelHuber *secondRobustKernel = new g2o::RobustKernelHuber();
          secondRobustKernel->setDelta(delta);
          edge_second2first->setRobustKernel(secondRobustKernel);
          optimizer.addEdge(edge_second2first);
          vSptrEdges_first2second.push_back(edge_first2second);
          vSptrEdges_second2first.push_back(edge_second2first);
          vEdgeIndices.push_back(i);
        }
      }
    }
    //optimize
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    //check inliners
    int badNum = 0;
    for(int i=0;i<vSptrEdges_first2second.size();i++){
      g2o::EdgeSim3ProjectXYZ *edge_first2second = vSptrEdges_first2second[i];
      g2o::EdgeInverseSim3ProjectXYZ *edge_second2first = vSptrEdges_second2first[i];
      if(edge_first2second && edge_second2first && \
      (edge_first2second->chi2()>_thd || edge_second2first->chi2()>_thd)){
        _vSptrMatchedMapPoints[vEdgeIndices[i]] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
        optimizer.removeEdge(edge_first2second);
        optimizer.removeEdge(edge_second2first);
        vSptrEdges_first2second[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(nullptr);
        vSptrEdges_second2first[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(nullptr);
        badNum++;
      }
    }
    int extraIterNum = 0;
    if(badNum>0){
      extraIterNum=10;
    }else {
      extraIterNum=5;
    }
    if(correspondenceNum - badNum < 10){
      return 0;
    }
    //optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(extraIterNum);
    int inlinerNum = 0;
    for(int i=0;i<vSptrEdges_first2second.size();i++){
      g2o::EdgeSim3ProjectXYZ *edge_first2second = vSptrEdges_first2second[i];
      g2o::EdgeInverseSim3ProjectXYZ *edge_second2first = vSptrEdges_second2first[i];
      if(edge_first2second && edge_second2first){
        if(edge_first2second->chi2()>_thd || edge_second2first->chi2()>_thd){
          _vSptrMatchedMapPoints[vEdgeIndices[i]] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
        }else {
          inlinerNum++;
        }
      }
    }
    //recover optimized Sim3
    g2o::VertexSim3Expmap *recoveredVertexSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    _g2oSim3_first2second = recoveredVertexSim3->estimate();
    return inlinerNum;
  }
}//namespace YDORBSLAM
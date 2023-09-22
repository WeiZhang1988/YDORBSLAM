/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "viewer.hpp"
#include <pangolin/pangolin.h>
#include <mutex>
#include <memory>
#include <unistd.h>

namespace YDORBSLAM{
Viewer::Viewer(std::shared_ptr<System> _sptrSystem, std::shared_ptr<FrameDrawer> _sptrFrameDrawer, std::shared_ptr<MapDrawer> _sptrMapDrawer, std::shared_ptr<Tracking> _sptrTracking, const string& _strConfigurationPath):\
m_sptr_system(_sptrSystem), m_sptr_frameDrawer(_sptrFrameDrawer), m_sptr_mapDrawer(_sptrMapDrawer), m_sptr_tracker(_sptrTracking), mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false){
    cv::FileStorage fileConfigs(strSettingPath, cv::FileStorage::READ);
    if(fileConfigs["Camera.fps"]<1){
        m_db_period = 1e3/30;
    }else{
        m_db_period = 1e3/fileConfigs["Camera.fps"];
    }
    m_flt_imageWidth = fileConfigs["Camera.width"];
    m_flt_imageHeight = fileConfigs["Camera.height"];
    if(m_flt_imageWidth<1 || m_flt_imageHeight<1)
    {
        m_flt_imageWidth = 640;
        m_flt_imageHeight = 480;
    }
    m_flt_viewpointX = fileConfigs["Viewer.ViewpointX"];
    m_flt_viewpointY = fileConfigs["Viewer.ViewpointY"];
    m_flt_viewpointZ = fileConfigs["Viewer.ViewpointZ"];
    m_flt_viewpointF = fileConfigs["Viewer.ViewpointF"];
}
void Viewer::run(){
    m_b_finished = false;
    m_b_stopped = false;
    pangolin::CreateWindowAndBind("YDORBSLAM: Map Viewer",1024,768);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState glRenderObj(pangolin::ProjectionMatrix(1024,768,m_flt_viewpointF,m_flt_viewpointF,512,389,0.1,1000),\
                                            pangolin::ModelViewLookAt(m_flt_viewpointX,m_flt_viewpointY,m_flt_viewpointZ, 0,0,0,0.0,-1.0, 0.0));
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& glDisplay = pangolin::CreateDisplay().SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)\
                                .SetHandler(std::make_shared<pangolin::Handler3D>(glRenderObj));
    pangolin::OpenGlMatrix glTw2c;
    glTw2c.SetIdentity();
    cv::namedWindow("YDORBSLAM: Current Frame");
    bool bFollow = true;
    bool bLocalizationMode = false;
    while(1){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        m_sptr_mapDrawer->getCurrentOpenGLCameraMatrix(glTw2c);
        if(menuFollowCamera && bFollow){
            glRenderObj.Follow(glTw2c);
        }
        else if(menuFollowCamera && !bFollow){
            glRenderObj.SetModelViewMatrix(pangolin::ModelViewLookAt(m_flt_viewpointX,m_flt_viewpointY,m_flt_viewpointZ, 0,0,0,0.0,-1.0, 0.0));
            glRenderObj.Follow(glTw2c);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow){
            bFollow = false;
        }
        if(menuLocalizationMode && !bLocalizationMode){
            m_sptr_system->activateLocalizationMode(); //no define
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode){
            m_sptr_system->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }
        glDisplay.Activate(glRenderObj);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        m_sptr_mapDrawer->drawCurrentCamera(glTw2c);
        if(menuShowKeyFrames || menuShowGraph){
            m_sptr_mapDrawer->drawKeyFrames(menuShowKeyFrames,menuShowGraph);
        }
        if(menuShowPoints){
            m_sptr_mapDrawer->drawMapPoints();
        }
        pangolin::FinishFrame();
        cv::imshow("YDORBSLAM: Current Frame", m_sptr_frameDrawer->drawFrame());
        cv::waitKey(m_db_period);
        if(menuReset){
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                m_sptr_system->deactivateLocalizationMode(); //no define
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            m_sptr_system->reset(); //no define
            menuReset = false;
        }
        if(stop()){
            while(isStopped()){
                usleep(3000);
            }
        }
        if(checkFinish()){
            break;
        }
    }
    setFinish();
}
bool Viewer::stop(){
    std::unique_lock<std::mutex> lock(m_mutex_stop);
    std::unique_lock<std::mutex> lock2(m_mutex_finish);
    if(m_b_finishRequested){
        return false;
    }
    else if(m_b_stopRequested){
        m_b_stopped = true;
        m_b_stopRequested = false;
        return true;
    }
    return false;
}

} // namespace YDORBSLAM
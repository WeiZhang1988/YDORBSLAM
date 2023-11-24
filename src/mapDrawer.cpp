/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "mapDrawer.hpp"
#include <mutex>

namespace YDORBSLAM{

MapDrawer::MapDrawer(std::shared_ptr<Map> _sptrMap, const std::string& _strConfigurationPath):m_sptr_map(_sptrMap){
    cv::FileStorage fileConfigs(_strConfigurationPath, cv::FileStorage::READ);
    m_flt_keyFrameSize = fileConfigs["Viewer.KeyFrameSize"];           //0.05
    m_flt_keyFrameLineWidth = fileConfigs["Viewer.KeyFrameLineWidth"]; //1
    m_flt_graphLineWidth = fileConfigs["Viewer.GraphLineWidth"];       //0.9
    m_flt_pointSize = fileConfigs["Viewer.PointSize"];                 //2
    m_flt_cameraSize = fileConfigs["Viewer.CameraSize"];               //0.08
    m_flt_cameraLineWidth = fileConfigs["Viewer.CameraLineWidth"];     //3
}
void MapDrawer::drawMapPoints(){
    const std::vector<std::shared_ptr<MapPoint>>& vSptrRefMPs = m_sptr_map->getReferenceMapPoints();
    std::set<std::shared_ptr<MapPoint>> sSptrRefMPs(vSptrRefMPs.begin(),vSptrRefMPs.end());
    if(m_sptr_map->getAllMapPoints().empty()){
        return;
    }
    glPointSize(m_flt_pointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    for(const std::shared_ptr<MapPoint>& sptrMP : m_sptr_map->getAllMapPoints()){
        if(!sptrMP->isBad() && !sSptrRefMPs.count(sptrMP)){
            glVertex3f(sptrMP->getPosInWorld().at<float>(0),sptrMP->getPosInWorld().at<float>(1),sptrMP->getPosInWorld().at<float>(2));
        }
    }
    glEnd();

    glPointSize(m_flt_pointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for(const std::shared_ptr<MapPoint>& sptrRefMP : sSptrRefMPs){
        if(!sptrRefMP->isBad()){
            glVertex3f(sptrRefMP->getPosInWorld().at<float>(0),sptrRefMP->getPosInWorld().at<float>(1),sptrRefMP->getPosInWorld().at<float>(2));
        }
    }
    glEnd();
}
void MapDrawer::drawKeyFrames(const bool& _bdrawKF, const bool& _bdrawGraph){
    const float &w = m_flt_keyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;
    if(_bdrawKF){
        for(const std::shared_ptr<KeyFrame>& sptrKF : m_sptr_map->getAllKeyFrames()){
            cv::Mat cvMatTw2c_t = sptrKF->getInverseCameraPoseByTransform_w2c().t();
            glPushMatrix();
            glMultMatrixf(cvMatTw2c_t.ptr<GLfloat>(0));
            glLineWidth(m_flt_keyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(w,-h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(-w,h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);
            glEnd();

            glPopMatrix();
        }
    }
    if(_bdrawGraph){
        glLineWidth(m_flt_graphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);
        for(const std::shared_ptr<KeyFrame>& sptrKF : m_sptr_map->getAllKeyFrames()){
            // Covisibility Graph
            if(!sptrKF->getOrderedConnectedKeyFramesLargerThanWeight(100).empty()){
                for(const std::shared_ptr<KeyFrame>& covisibleKF : sptrKF->getOrderedConnectedKeyFramesLargerThanWeight(100)){
                    if(covisibleKF->m_int_keyFrameID > sptrKF->m_int_keyFrameID){
                        glVertex3f(sptrKF->getCameraOriginInWorld().at<float>(0),sptrKF->getCameraOriginInWorld().at<float>(1),sptrKF->getCameraOriginInWorld().at<float>(2));
                        glVertex3f(covisibleKF->getCameraOriginInWorld().at<float>(0),covisibleKF->getCameraOriginInWorld().at<float>(1),covisibleKF->getCameraOriginInWorld().at<float>(2));
                    }
                }
            }
            // Spanning tree
            if(sptrKF->getParent()){
                glVertex3f(sptrKF->getCameraOriginInWorld().at<float>(0),sptrKF->getCameraOriginInWorld().at<float>(1),sptrKF->getCameraOriginInWorld().at<float>(2));
                glVertex3f(sptrKF->getParent()->getCameraOriginInWorld().at<float>(0),sptrKF->getParent()->getCameraOriginInWorld().at<float>(1),sptrKF->getParent()->getCameraOriginInWorld().at<float>(2));
            }
            // Loops
            for(const std::shared_ptr<KeyFrame>& loopKF : sptrKF->getLoopEdges()){
                if(loopKF->m_int_keyFrameID > sptrKF->m_int_keyFrameID){
                    glVertex3f(sptrKF->getCameraOriginInWorld().at<float>(0),sptrKF->getCameraOriginInWorld().at<float>(1),sptrKF->getCameraOriginInWorld().at<float>(2));
                    glVertex3f(loopKF->getCameraOriginInWorld().at<float>(0),loopKF->getCameraOriginInWorld().at<float>(1),loopKF->getCameraOriginInWorld().at<float>(2));
                }
            }
        }
        glEnd();
    }
}
void MapDrawer::drawCurrentCamera(pangolin::OpenGlMatrix& _glTw2c){
    const float &w = m_flt_cameraSize;
    const float h = w*0.75;
    const float z = w*0.6;
    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(_glTw2c.m);
#else
        glMultMatrixd(_glTw2c.m);
#endif
    glLineWidth(m_flt_cameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(w,-h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glEnd();

    glPopMatrix();
}
void MapDrawer::setCurrentCameraPose(const cv::Mat& _cvMatTc2w){
    std::unique_lock<std::mutex> lock(m_mutex_camera);
    m_cvMat_cameraPoseTc2w = _cvMatTc2w;
}
void MapDrawer::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& _glCameraPose){
    if(!m_cvMat_cameraPoseTc2w.empty()){
        cv::Mat Rw2c(3,3,CV_32F);
        cv::Mat tw2c(3,1,CV_32F);
        {
            std::unique_lock<std::mutex> lock(m_mutex_camera);
            Rw2c = m_cvMat_cameraPoseTc2w.rowRange(0,3).colRange(0,3).t();
            tw2c = -Rw2c * m_cvMat_cameraPoseTc2w.rowRange(0,3).col(3);
        }
        _glCameraPose.m[0] = Rw2c.at<float>(0,0);
        _glCameraPose.m[1] = Rw2c.at<float>(1,0);
        _glCameraPose.m[2] = Rw2c.at<float>(2,0);
        _glCameraPose.m[3] = 0.0;

        _glCameraPose.m[4] = Rw2c.at<float>(0,1);
        _glCameraPose.m[5] = Rw2c.at<float>(1,1);
        _glCameraPose.m[6] = Rw2c.at<float>(2,1);
        _glCameraPose.m[7] = 0.0;

        _glCameraPose.m[8] = Rw2c.at<float>(0,2);
        _glCameraPose.m[9] = Rw2c.at<float>(1,2);
        _glCameraPose.m[10] = Rw2c.at<float>(2,2);
        _glCameraPose.m[11] = 0.0;

        _glCameraPose.m[12] = tw2c.at<float>(0);
        _glCameraPose.m[13] = tw2c.at<float>(1);
        _glCameraPose.m[14] = tw2c.at<float>(2);
        _glCameraPose.m[15] = 1.0;
    }else{
        _glCameraPose.SetIdentity();
    }
}

} //namespace YDORBSLAM
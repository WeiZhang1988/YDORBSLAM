#ifndef YDORBSLAM_ENUMCLASS_HPP
#define YDORBSLAM_ENUMCLASS_HPP

namespace YDORBSLAM{
enum class TrackingState{
    SYSTEM_NOT_READY,
    NO_IMAGE_YET,
    NOT_INITIALIZED,
    OK,
    LOST
};

enum class Sensor{
    MONOCULAR,
    STEREO,
    RGBD
};
} //namespace YDORBSLAM

#endif //YDORBSLAM_ENUMCLASS_HPP
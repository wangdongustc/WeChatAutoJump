#ifndef UTILS_H
#define UTILS_H

#include <glog/logging.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <string>
#include <vector>

std::string GetCommandOutput(const char* command);

inline std::string get_time_string() {
  time_t time_raw;
  time_raw = time(&time_raw);
  struct tm* timeinfo = localtime(&time_raw);
  return asctime(timeinfo);
}

template<class T>
void swap_if_larger(T* a, T* b) {
  if (*a <= *b) return;
  auto t = *a;
  *a = *b;
  *b = t;
}

inline void init_glog(std::string logname) {
  FLAGS_stderrthreshold = google::GLOG_INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(logname.c_str());
}

inline int cvt2gray(const cv::Mat& src, cv::Mat* dst) {
  if (src.channels() == 3) {
    cv::cvtColor(src, dst[0], CV_RGB2GRAY);
    return 0;
  } else if (src.channels() == 1) {
    dst[0] = src.clone();
    return 0;
  } else {
    LOG(WARNING) << "color conversion failed";
    return -1;
  }
}

inline int cvt2color(const cv::Mat& src, cv::Mat* dst) {
  if (src.channels() == 3) {
    dst[0] = src.clone();
    return 0;
  } else if (src.channels() == 1) {
    cv::cvtColor(src, dst[0], CV_GRAY2BGR);
    return 0;
  } else {
    LOG(WARNING) << "color conversion failed";
    return -1;
  }
}

#endif  // UTILS_H

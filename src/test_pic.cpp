#include <vector>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace std;
using namespace cv;

#define IMAGE_PATH  "/home/wangdong/Projects/WeChatAutoJump/imgs/10.png"

int h_low = 110;
int h_high = 135;
int s_low = 55;
int s_high = 120;
int v_low = 0;
int v_high = 255;

int b_low = 0;
int b_high = 255;
int g_low = 0;
int g_high = 255;
int r_low = 0;
int r_high = 255;

Mat img, hsv;
int x, y, lorr;

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;

int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 50;

int detect_man(const Mat& src, int* x, int* y, int* lorr) {
  Mat color_match;
  inRange(hsv, Scalar(h_low, s_low, v_low), Scalar(h_high, s_high, v_high), color_match);


  int operation = morph_operator + 2;
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( color_match, color_match, operation, element );

  namedWindow("test", WINDOW_NORMAL);
  imshow("test", color_match);

  return 0;
}

void on_trackbar_change(int pos, void* userdata) {
  detect_man(img, &x, &y, &lorr);
}

int main() {
  init_glog("test");

  img = imread(IMAGE_PATH);
  cvtColor(img, hsv, CV_BGR2HSV);
  const char* window_name = "test";
  namedWindow(window_name, CV_WINDOW_NORMAL);
  createTrackbar( "h_low:", window_name, &h_low, 255, on_trackbar_change);
  createTrackbar( "h_high:", window_name, &h_high, 255, on_trackbar_change);
  createTrackbar( "s_low:", window_name, &s_low, 255, on_trackbar_change);
  createTrackbar( "s_high:", window_name, &s_high, 255, on_trackbar_change);
  createTrackbar( "v_low:", window_name, &v_low, 255, on_trackbar_change);
  createTrackbar( "v_high:", window_name, &v_high, 255, on_trackbar_change);


  createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, on_trackbar_change );
  createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
                    &morph_elem, max_elem,
                    on_trackbar_change );
  createTrackbar( "Kernel size:\n 2n +1", window_name,
                    &morph_size, max_kernel_size,
                    on_trackbar_change );

  detect_man(img, &x, &y, &lorr);
  waitKey();
  return 0;
}


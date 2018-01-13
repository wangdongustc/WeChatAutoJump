#include <vector>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

#define IMAGE_PATH  "/home/wangdong/Projects/WeChatAutoJump/imgs/"
#define ADB "/home/wangdong/Android/Sdk/platform-tools/adb"
#define LEFT 0
#define RIGHT 1
#define MIDDLE 2

using namespace std;
using namespace cv;

int analysis(const Mat& src);
int reduce_bg(const Mat src, int lorr, Mat* dst);
int canny_detect(const Mat& src);
int detect_man(const Mat& src, float* x, float* y, int* lorr);
int detect_stand(const Mat& src, const float man_x, const float man_y,
                 const int man_lorr, float* x, float* y);
int find_last_error(const Mat& src, const int man_x, const int man_y,
                    const int lorr, float* last_error);
int find_best_center(const vector<Point2f>& centers, Point2f* best);
int get_screenshot(Mat* screenshot);
int tap_distance(float dstance);

//float coeffs  = 1.596;
float coeffs  = 1.4;

int main() {
  init_glog("test");
  // get all the files in the image path
  string command(IMAGE_PATH);
  command = "ls " + command;
  stringstream files(GetCommandOutput(command.c_str()));
  string filename;
  static RNG rng;
  int count = 0;
  float last_error;
  while(/* getline(files, filename) */ 1) {
    LOG(INFO) << "\n\n***********************************";
    // filename = IMAGE_PATH + filename;
    // Mat src = imread(filename);
    Mat src;
    get_screenshot(&src);
    if (src.cols <= 0 || src.rows <= 0) break;
    // analysis(src);
    // canny_detect(src);
    // determine the position of the man
    float man_x, man_y;
    int lorr, last_lorr;
    last_lorr = lorr;
    detect_man(src, &man_x, &man_y, &lorr);
    LOG(INFO) << "man: " << man_x << ", " << man_y;
    // determine the position of the next stand
    float stand_x, stand_y;
    detect_stand(src, man_x, man_y, lorr, &stand_x, &stand_y);
    LOG(INFO) << "stand: " << stand_x << ", " << stand_y;
    // distance along the x axis
    float d = stand_x - man_x;
    /* // find the error of last jump
    if (count != 0) {
      last_error = 0;
      if (0 == find_last_error(src, man_x, man_y, last_lorr, &last_error)) {
        LOG(ERROR) << "last_error: " << last_error;
        if (abs(last_error) >  10) {
          coeffs *= 0.8 + 0.2 * d * 1.0 / (d + last_error) * 1.0;
        }
      }
    } */
    tap_distance(d);
    LOG(INFO) << "count: " << count++;
    if(waitKey(1200 + rng.next() % 500) == 27) break;
  }

  return 0;
}

int detect_man(const Mat& src, float *x, float *y, int* lorr) {
  Mat color_match, hsv, src_nobg, bg_area;
  reduce_bg(src, MIDDLE, &bg_area);
  src.copyTo(src_nobg, bg_area);
  // first match the color in HSV color space
  cvtColor(src_nobg, hsv, CV_BGR2HSV);
  inRange(hsv, Scalar(110, 55, 70), Scalar(135, 120, 105), color_match);
  // there might be some missmatch, do a closing operation to reduce them
  int morph_size = 15;
  Mat element = getStructuringElement(MORPH_ELLIPSE,
                                      Size(2*morph_size+1, 2*morph_size+1),
                                      Point(morph_size, morph_size));
  morphologyEx(color_match, color_match, MORPH_OPEN, element);
  // show the extracted "man"
  namedWindow("color_match", WINDOW_NORMAL);
  imshow("color_match", color_match);
  // get the position of the man
  int height = color_match.rows;
  int width  = color_match.cols;
  for (int row = height * 0.65; row > 0.5 * height; --row) {
   unsigned char* row_ptr = color_match.ptr(row);
   Mat arow(1, width, color_match.type(), row_ptr);
   if (mean(arow)[0] > 0) {
     *y = row;
     vector<int> nonzero_cols;
     for (int col = 0; col < width; ++col) {
       if (row_ptr[col] > 0) {
         nonzero_cols.push_back(col);
       }
     }
     *x = mean(nonzero_cols)[0];
     break;
   }
  }
  // add a little correction to the y pos
  *y -= 15;
  // determine whether the man is on the left or the right
  if (*x > width / 2) {
    *lorr = RIGHT;
  } else {
    *lorr = LEFT;
  }
  // show the pos to see if it is right
  Mat src_withpos = src.clone();
  line(src_withpos, Point(0, *y), Point(width, *y), Scalar(0, 255, 0), 3, LINE_4);
  line(src_withpos, Point(*x, 0), Point(*x, height), Scalar(0, 255, 0), 3, LINE_4);
  // draw a line of 30 degree from the man

  int end_x;
  if (*lorr == LEFT) {
    end_x = *x + 1732;
  } else {
    end_x = *x - 1732;
  }
  int end_y = *y - 1000;
  line(src_withpos, Point(*x, *y), Point(end_x, end_y), Scalar(0, 0, 255), 3, LINE_4);
  namedWindow("src_withpos", WINDOW_NORMAL);
  imshow("src_withpos", src_withpos);
  return 0;
}

int detect_stand(const Mat& src, const float man_x, const float man_y,
                 const int man_lorr, float *x, float *y) {
  // make a roi
  int side_bound = 0.2 * src.cols;
  int side_bound_small = 20;
  int roi_height = 0.577 * (src.cols - side_bound);
  int roi_width = src.cols - side_bound - side_bound_small;
  int high_y = man_y - roi_height;
  roi_height *= 1.2;
  Size roi_size(roi_width, roi_height);
  Rect roi_rect;
  Point origin;
  if (man_lorr == LEFT) {
    origin.x = side_bound;
    origin.y = high_y;
    roi_rect = Rect(origin, roi_size);
    // roi_rect = Rect(side_bound, low_y - roi_height, src.cols -1 , roi_height);
  } else {
    origin.x = side_bound_small;
    origin.y = high_y;
    roi_rect = Rect(origin, roi_size);
    // roi_rect = Rect(0, low_y - roi_height, src.cols - side_bound, roi_height);
  }
  LOG(INFO) << "roi_rect: " << roi_rect;
  Mat src_withroi = src.clone();
  rectangle(src_withroi, roi_rect, Scalar(255, 0, 0), 3);
  namedWindow("src_withroi", WINDOW_NORMAL);
  imshow("src_withroi", src_withroi);
  Mat roi_base = src.clone();
  Mat roi = roi_base(roi_rect);
  // reduce the background
  Mat nobg;
  reduce_bg(roi, man_lorr, &nobg);
  // remove the man
  line(nobg, Point(man_x - origin.x, man_y - origin.y),
       Point(man_x - origin.x, 0), 0, 60);
  namedWindow("nobg", CV_WINDOW_NORMAL);
  imshow("nobg", nobg);
  // find the first row with color
  for (int row = 2; row < nobg.rows; ++row) {
    unsigned char* row_ptr = nobg.ptr(row);
    Mat arow(1, nobg.cols, nobg.type(), row_ptr);
    if (sum(arow)[0] > 0) {
      *y = row;
      vector<int> nonzero_cols;
      stringstream s;
      for (int col = 0; col < nobg.cols; ++col) {
        s << (int)row_ptr[col] << ",   ";
        if (row_ptr[col] > 0) {
          nonzero_cols.push_back(col);
        }
      }
      *x = mean(nonzero_cols)[0];
      break;
    }
  }
  circle(roi, Point(*x, *y), 10, Scalar(0, 255, 0), 10);
  namedWindow("roi", CV_WINDOW_NORMAL);
  imshow("roi", roi);
  *x = *x + origin.x;
  *y = *y + origin.y;
  return 0;
}

int reduce_bg(const Mat src, int lorr, Mat* dst) {
  *dst = src.clone();
  Vec3i bg1, bg2;
  if (lorr == LEFT) {
    bg1 = dst->at<Vec3b>(0, 0);
    bg2 = dst->at<Vec3b>(dst->rows - 1, dst->cols - 1);
  } else {
    bg1 = dst->at<Vec3b>(0, dst->cols - 1);
    bg2 = dst->at<Vec3b>(dst->rows - 1, 0);
  }
  swap_if_larger(&bg1[0], &bg2[0]);
  swap_if_larger(&bg1[1], &bg2[1]);
  swap_if_larger(&bg1[2], &bg2[2]);
  Mat bgarea;
  inRange(*dst, bg1, bg2, bgarea);
  *dst = 255 - bgarea;
  namedWindow("bgarea", WINDOW_NORMAL);
  imshow("bgarea", *dst);
  LOG(INFO) << "OK";
  return 0;
}

int find_last_error(const Mat& src, const int man_x, const int man_y,
                    const int lorr, float* last_error) {
  Rect roi_rect;
  roi_rect.width = 120;
  roi_rect.height = 700;
  roi_rect.x = man_x - roi_rect.width / 2;
  roi_rect.y = man_y - roi_rect.height / 2;
  Mat nobg, src_error_roi = src.clone();
  reduce_bg(src, lorr, &nobg);
  rectangle(src_error_roi, roi_rect, Scalar(0, 255, 0), 3);
  Mat roi = nobg(roi_rect);
  int x, y;
  // find the first row with color
  for (int row = roi.rows - 1; row > 1; --row) {
    unsigned char* row_ptr = roi.ptr(row);
    Mat arow(1, roi.cols, roi.type(), row_ptr);
    if (sum(arow)[0] > 0) {
      if (row == roi.rows) {
        return -1;
      } else {
        if (row_ptr[0] != 0 || row_ptr[roi.cols - 1]) {
          return -1;
        }
      }
      y = row;
      vector<int> nonzero_cols;
      stringstream s;
      for (int col = 0; col < roi.cols; ++col) {
        s << (int)row_ptr[col] << ",   ";
        if (row_ptr[col] > 0) {
          nonzero_cols.push_back(col);
        }
      }
      x = mean(nonzero_cols)[0];
      break;
    }
  }
  circle(src_error_roi, Point(x + roi_rect.x, y + roi_rect.y), 3, Scalar(255, 255, 0), 3);
  if (lorr == LEFT) {
    *last_error = man_x - x - roi_rect.x;
  } else {
    *last_error = x + roi_rect.x - man_x;
  }
  namedWindow("src_error_roi", WINDOW_NORMAL);
  imshow("src_error_roi", src_error_roi);
  return 0;
}

int get_screenshot(Mat* screenshot) {
  GetCommandOutput(ADB " shell screencap -p /data/local/tmp/ss.png");
  GetCommandOutput(ADB " pull /data/local/tmp/ss.png /tmp/");
  *screenshot = imread("/tmp/ss.png");
  return 0;
}


int tap_distance(float distance) {
  char command[80];
  static RNG rng;
  LOG(INFO) << "coeffs: " << coeffs;
  float k = coeffs /* + (rng.next() % 100) / 10000.0 */;
  LOG(INFO) << "k: " << k;
  int time = abs(distance * coeffs);
  sprintf(command, ADB " shell input swipe %d %d %d %d %d",
          900 + rng.next() % 100, 800 + rng.next() % 100,
          800 + rng.next() % 100, 900 + rng.next() % 100, time);
  GetCommandOutput(command);
  LOG(INFO) << "command: " << command;
  return 0;
}

#include <vector>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

#define IMAGE_PATH  "/home/wangdong/Projects/WeChatAutoJump/imgs/"
#define ADB "/home/wangdong/Android/Sdk/platform-tools/adb"
#define LEFT 0
#define RIGHT 1

using namespace std;
using namespace cv;

int analysis(const Mat& src);
int reduce_bg(const Mat& src);
int canny_detect(const Mat& src);
int detect_man(const Mat& src, int* x, int* y, int* lorr);
int detect_stand(const Mat& src, const int man_x, const int man_y,
                 const int man_lorr, int* x, int* y);
int find_best_center(const vector<Point2f>& centers, Point2f* best);
int get_screenshot(Mat* screenshot);

int main() {
  init_glog("test");
  // get all the files in the image path
  string command(IMAGE_PATH);
  command = "ls " + command;
  stringstream files(GetCommandOutput(command.c_str()));
  string filename;
  while(/*getline(files, filename)*/ 1) {
    // filename = IMAGE_PATH + filename;
    // Mat src = imread(filename);
    Mat src;
    get_screenshot(&src);
    if (src.cols <= 0 || src.rows <= 0) break;
    // analysis(src);
    reduce_bg(src);
    // canny_detect(src);
    // determine the position of the man
    int man_x, man_y, lorr;
    detect_man(src, &man_x, &man_y, &lorr);
    // determine the position of the next stand
    int stand_x, stand_y;
    detect_stand(src, man_x, man_y, lorr, &stand_x, &stand_y);
    if(waitKey(100) == 27) break;
  }

  return 0;
}

vector<Mat> hsv_channals;
Mat hsv;
int analysis(const Mat& src)
{
  cvtColor(src, hsv, CV_BGR2HSV);
  split(hsv, hsv_channals);
  namedWindow("src", WINDOW_NORMAL);
  imshow("src", src);
  namedWindow("h", WINDOW_NORMAL);
  imshow("h", hsv_channals[0]);
  imwrite("./h.png", hsv_channals[0]);
  namedWindow("s", WINDOW_NORMAL);
  imshow("s", hsv_channals[1]);
  imwrite("./s.png", hsv_channals[1]);
  namedWindow("v", WINDOW_NORMAL);
  imshow("v", hsv_channals[2]);
  imwrite("./v.png", hsv_channals[2]);
  return 0;
}

int reduce_bg(const Mat& src) {
  int min = 50;
  Vec3i bgcolor = src.at<Vec3b>(src.cols * 0.05, src.rows * 0.05);
  Mat img = src.clone();
  circle(img, Point(src.cols * 0.05, src.rows * 0.05), 10, Scalar(255, 0, 255), 5);
  namedWindow("bgcolor", WINDOW_NORMAL);
  imshow("bgcolor", img);
  Mat diff(src.size(), CV_16UC1, Scalar(0));
  uint16_t max = 0;
  for (int i = 0.2 * src.rows; i < 0.9 * src.rows; ++i) {
    auto diff_rowptr = diff.ptr<uint16_t>(i);
    const auto src_rowptr = src.ptr<Vec3b>(i);
    for (int j = 0; j < src.cols; ++j) {
      Vec3i src_color = src_rowptr[j];
      diff_rowptr[j] = norm(src_color - bgcolor);
      if (diff_rowptr[j] > max) {
        max = diff_rowptr[j];
      }
      if (diff_rowptr[j] < min) {
        diff_rowptr[j] = 0;
      }
    }
  }
  diff.convertTo(diff, CV_8UC1, 255.0/max);
  Mat result(src.size(), CV_16UC3, Scalar(0, 0, 0));
  src.copyTo(result, diff);
  namedWindow("result", WINDOW_NORMAL);
  imshow("result", result);
  namedWindow("diff", WINDOW_NORMAL);
  imshow("diff", diff);
  analysis(result);
  LOG(INFO) << "OK";
  return 0;
}

int canny_detect(const Mat& src) {
  Mat blured, gray, edges;
  blur(src, blured, Size(2,2));
  cvtColor(blured, gray, CV_BGR2GRAY);
  namedWindow("gray", WINDOW_NORMAL);
  imshow("gray", gray);
  Mat hsv;
  cvtColor(blured, hsv, CV_BGR2HSV);
  Canny(hsv, edges, 20, 50);
  LOG(INFO) << "MEAN: " <<  mean(src)[0];
  // adaptiveThreshold(gray, edges, 255, BORDER_REPLICATE, THRESH_BINARY, 11, mean(src)[0]);
  int morph_size = 2;
  Mat element = getStructuringElement(MORPH_ELLIPSE,
                                      Size(2*morph_size+1, 2*morph_size+1),
                                      Point(morph_size, morph_size));
  morphologyEx(edges, edges, MORPH_DILATE, element);
  namedWindow("edges", WINDOW_NORMAL);
  imshow("edges", edges);

  vector<vector<Point>> contours;
  findContours(edges, contours, RETR_LIST, CHAIN_APPROX_NONE);
  RNG rgn;
  Mat src_withcontours;;
  cvtColor(gray, src_withcontours, CV_GRAY2BGR);
  vector<Point2f> possible_centers;
  for (int i = 0; i < contours.size(); ++i) {
    auto& ct = contours[i];
    if (ct.size() < 100) {
      continue;
    }
    else {
      // LOG(INFO) << "ct.size(): " << ct.size();
      RotatedRect rrect = fitEllipse(ct);
      float tolerance = 3;
      if (abs(fmod(rrect.angle + 180 ,180) - 90) > tolerance) {
        continue;
      }
      tolerance = 0.05;
      float ratio = rrect.size.width / rrect.size.height;
      LOG(INFO) << "ratio: " << ratio;
      if (ratio - 0.57 > tolerance) {
        continue;
      }
      possible_centers.push_back(rrect.center);
    }
    drawContours(src_withcontours, contours, i, Scalar(rgn.next(), rgn.next(), rgn.next()), 3);
  }
  Point2f center;
  find_best_center(possible_centers, &center);
//  center.x = mean(possible_centers)[0];
//  center.y = mean(possible_centers)[1];
  circle(src_withcontours, center, 4, Scalar(0, 255, 0), 5);
  namedWindow("edges_withcontours", WINDOW_NORMAL);
  imshow("edges_withcontours", src_withcontours);
  return 0;
}

int find_best_center(const vector<Point2f>& centers, Point2f* best) {
  if (centers.size() == 1) {
    *best = centers[0];
    return true;
  }
  Point2d sum(0, 0);
  Point2d sq_sum(0, 0);
  for (auto & p : centers) {
    sum = Point2d(sum.x + p.x, sum.y + p.y);
  }
  Point2d avg = Point2d(sum.x / centers.size(), sum.y / centers.size());
  for (auto & p : centers) {
    float xx = p.x - avg.x;
    float yy = p.y - avg.y;
    sq_sum = Point2d(sq_sum.x + xx * xx, sq_sum.y + yy * yy);
  }
  Point2d var = Point2d(sqrt(sq_sum.x / (centers.size() - 1)),
                        sqrt(sq_sum.y / (centers.size() - 1)));
  LOG(INFO) << "centers.size(): " << centers.size();
  LOG(INFO) << "avg, var: " << avg << ", " << var;
  int found_outlier = false;
  vector<Point2f> new_centers;
  for (auto & p : centers) {
    if (norm(Point(p.x - avg.x, p.y - avg.y)) > norm(var)) {
      found_outlier = true;
      continue;
    }
    new_centers.push_back(p);
  }
  if (found_outlier == false) {
    *best = avg;
    return true;
  }
  else {
    return find_best_center(new_centers, best);
  }
}

int detect_man(const Mat& src, int* x, int* y, int* lorr) {
  Mat color_match;
  // first match the color in HSV color space
  cvtColor(src, hsv, CV_BGR2HSV);
  inRange(hsv, Scalar(110, 55, 0), Scalar(135, 120, 255), color_match);
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

int detect_stand(const Mat& src, const int man_x, const int man_y,
                 const int man_lorr, int* x, int* y) {
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
  // Canny
  canny_detect(roi);
  return 0;
}

int get_screenshot(Mat* screenshot) {
  GetCommandOutput(ADB " shell screencap -p /data/local/tmp/ss.png");
  GetCommandOutput(ADB " pull /data/local/tmp/ss.png /tmp/");
  *screenshot = imread("/tmp/ss.png");
  return 0;
}

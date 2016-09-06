#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

static const float kScaleFactorForDisplay = 0.5;

// Find the angle between line [p1, p2] and [p2, p3].
float GetAngleBetweenThreePoints(Point3f p1, Point3f p2, Point3f p3) {
  float x1 = p1.x - p2.x;
  float y1 = p1.y - p2.y;
  float z1 = p1.z - p2.z;

  float x2 = p3.x - p2.x;
  float y2 = p3.y - p2.y;
  float z2 = p3.z - p2.z;

  return acos((x1 * x2 + y1 * y2 + z1 * z2) /
	      (sqrt(x1 * x1 + y1 * y1 + z1 * z1) *
	       sqrt(x2 * x2 + y2 * y2 + z2 * z2))) * 180 / M_PI;
}

float GetDepth(Point2f p1, Point2f p2) {
  static const int WIDTH = 1632;
  static const int HEIGHT = 1224;
  static const int GAP = 1500;
  static const int FOCAL_LEN = 200;

  Point3f p1_3d(p1.x, p1.y, 0);
  Point3f p2_3d(p2.x + WIDTH + GAP, p2.y, 0);

  Point3f f1(WIDTH / 2, HEIGHT / 2, -FOCAL_LEN);
  Point3f f2(WIDTH / 2 + WIDTH + GAP, HEIGHT / 2, -FOCAL_LEN);

  float angle1 = 180.0 - GetAngleBetweenThreePoints(f1, p1_3d, p2_3d);
  float angle2 = 180.0 - GetAngleBetweenThreePoints(f2, p2_3d, p1_3d);

  float distance = sqrt((p1_3d.x - p2_3d.x) * (p1_3d.x - p2_3d.x) +
			(p1_3d.y - p2_3d.y) * (p1_3d.y - p2_3d.y));

  return distance * sin(angle1 / 180.0 * M_PI) / sin(angle2 / 180.0 * M_PI);
}

void DrawLines(Mat& mat1,
	       Mat& mat2,
	       std::vector<Point2f>& points1,
	       std::vector<Point2f>& points2,
	       std::vector<cv::Vec<float, 3> >& lines1,
	       std::vector<cv::Vec<float, 3> >& lines2) {
  assert(points1.size() == points2.size());
  assert(points1.size() == lines1.size());
  assert(lines1.size() == lines2.size());

  for (int i = 0; i < lines1.size(); i++) {
    cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

    cv::line(mat1,
	     cv::Point(0, -lines1[i][2] / lines1[i][1]),
	     cv::Point(mat1.cols, -(lines1[i][2] + lines1[i][0] * mat1.cols) / lines1[i][1]),
	     color);
    cv::circle(mat1, points1[i], 3, color, -1, CV_AA);

    cv::line(mat2,
	     cv::Point(0, -lines2[i][2] / lines2[i][1]),
	     cv::Point(mat2.cols, -(lines2[i][2] + lines2[i][0] * mat2.cols) / lines2[i][1]),
	     color);
    cv::circle(mat2, points2[i], 3, color, -1, CV_AA); 
  }
}

static const int kImgWidth = 640;
static const int kImgHeight = 480;

void RealMain() {
  //Mat mat1 = imread("1.jpg");
  //Mat mat2 = imread("2.jpg");

  VideoCapture camera1(1);
  VideoCapture camera2(2);

  if (!camera1.isOpened() || !camera2.isOpened()) {
    std::cout << "Failed to open camera." << std::endl;
    return;
  }

  // set video capture properties for MacBook' iSight camera
  // Use a smaller buffer so camera frames get populated faster.
  camera1.set(CV_CAP_PROP_FRAME_WIDTH, kImgWidth);
  camera1.set(CV_CAP_PROP_FRAME_HEIGHT, kImgHeight);
  camera2.set(CV_CAP_PROP_FRAME_WIDTH, kImgWidth);
  camera2.set(CV_CAP_PROP_FRAME_HEIGHT, kImgHeight);

  std::cout << "Wait 1 sec for buffer to populate" << std::endl;
  usleep(1000000);

  while (true) {

  Mat mat1;
  Mat mat2;
  camera1 >> mat1;
  camera2 >> mat2;

  Mat main = mat1.clone();

  assert(mat1.rows == mat2.rows);
  assert(mat1.cols == mat2.cols);

  Size sz = mat1.size();
  Size target_size(sz.width * kScaleFactorForDisplay, sz.height * kScaleFactorForDisplay);

  Mat gray1(sz.height, sz.width, CV_8UC1);
  cvtColor(mat1, gray1, CV_BGR2GRAY);
  Mat gray2(sz.height, sz.width, CV_8UC1);
  cvtColor(mat2, gray2, CV_BGR2GRAY);

  /* StereoBM method
  Mat mat_disparity_16s = Mat(sz.height, sz.width, CV_16S);
  Mat mat_disparity_8u = Mat(sz.height, sz.width, CV_8UC1);

  int n_disparities = 16 * 5;  // Range of disparity.
  int sad_window_size = 21;  // Size of the block window. Must be odd.
  StereoBM sbm(StereoBM::BASIC_PRESET, n_disparities, sad_window_size);

  sbm(gray1, gray2, mat_disparity_16s);
  double min_val; double max_val;
  minMaxLoc(mat_disparity_16s, &min_val, &max_val);
  mat_disparity_16s.convertTo(mat_disparity_8u, CV_8UC1, 255 / (max_val - min_val));

  // Build output image.
  Mat mat3(target_size.height, target_size.width * 3, CV_8UC1);
  resize(gray1, Mat(mat3, Rect(0, 0, target_size.width, target_size.height)), target_size);
  resize(gray2, Mat(mat3, Rect(target_size.width, 0, target_size.width, target_size.height)), target_size);
  resize(mat_disparity_8u, Mat(mat3, Rect(target_size.width * 2, 0, target_size.width, target_size.height)), target_size);

  imshow("display", mat3);
  */

  // Key points.
  SurfFeatureDetector surf(400);
  std::vector<KeyPoint> keypoints1, keypoints2;
  surf.detect(gray1, keypoints1);
  surf.detect(gray2, keypoints2);

  // Descriptors.
  SurfDescriptorExtractor extractor;
  Mat descriptors1, descriptors2;
  extractor.compute(gray1, keypoints1, descriptors1);
  extractor.compute(gray2, keypoints2, descriptors2);

  // Flann matching.
  FlannBasedMatcher matcher;
  std::vector<std::vector<DMatch> > matches;
  matcher.knnMatch(descriptors1, descriptors2, matches, 2);

  std::vector<DMatch> good_matches;
  for (int i = 0; i < matches.size(); i++) {
    if (matches[i].size() < 2) {
      continue;
    }

    const DMatch& first = matches[i][0];
    const DMatch& second = matches[i][1];

    // Ratio test per Lowe's paper.
    if (first.distance < 0.7 * second.distance) {
      good_matches.push_back(first);
    }
  }

  Mat mat_matches;
  drawMatches(mat1, keypoints1, mat2, keypoints2, good_matches, mat_matches,
	      Scalar::all(-1), Scalar::all(-1), vector<char>(),
	      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  //imshow("matches", mat_matches);

  // Compute fundamental matrix.

  std::vector<Point2f> match_points1, match_points2;
  for (int i = 0; i < good_matches.size(); i++) {
    const DMatch& m = good_matches[i];
    match_points1.push_back(keypoints1[m.queryIdx].pt);
    match_points2.push_back(keypoints2[m.trainIdx].pt);
  }

  std::vector<uchar> mask;
  Mat fundamental_matrix =
    findFundamentalMat(match_points1, match_points2, FM_RANSAC, 3, 0.99, mask);

  std::vector<Point2f> inliners1, inliners2;
  for (int i = 0; i < mask.size(); i++) {
    if (mask[i] == 1) {
      inliners1.push_back(match_points1[i]);
      inliners2.push_back(match_points2[i]);
    }
  }

  /*
  std::vector<cv::Vec<float, 3> > epilines1, epilines2;
  computeCorrespondEpilines(inliners1, 1, fundamental_matrix, epilines2);
  computeCorrespondEpilines(inliners2, 2, fundamental_matrix, epilines1);
  */

  std::vector<float> depths(inliners1.size());
  for (int i = 0; i < inliners1.size(); i++) {
    depths[i] = GetDepth(inliners1[i], inliners2[i]);
  }
  std::vector<float> normalized_depths;
  normalize(depths, normalized_depths, 0, 255, NORM_MINMAX);

  for (int i = 0; i < inliners1.size(); i++) {
    // std::cout << normalized_depths[i] << std::endl;
    int gray = round(normalized_depths[i]);
    cv::Scalar color(gray, gray, gray);
    cv::circle(mat1, inliners1[i], 5, color, -1, CV_AA);
  }

  //imshow("mat 1", mat1);
  //imshow("mat 2", mat2);

  Mat mat1_e;
  equalizeHist(gray1, mat1_e);

  // dilate to remove small black spots
  Mat strel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
  Mat dilate1;
  dilate(mat1_e, dilate1, strel);

  // open and close to highlight objects
  strel = getStructuringElement(MORPH_ELLIPSE, Size(19, 19));
  Mat morph1;
  morphologyEx(dilate1, morph1, MORPH_OPEN, strel);
  morphologyEx(morph1, morph1, MORPH_CLOSE, strel);

  // adaptive threshold to create binary image
  Mat threshold1;
  adaptiveThreshold(morph1, threshold1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 105, 0);

  // erode binary image twice to separate regions
  Mat erode1;
  erode(threshold1, erode1, strel, Point(-1, -1), 2);

  // Find contours.
  vector<vector<Point> > contours, big_contours;
  vector<Vec4i> hierarchy;
  findContours(erode1, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

  // remove very small contours
  for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
    if(contourArea(contours[idx]) > 50) {
      big_contours.push_back(contours[idx]);
    }
  }

  Mat markers = cv::Mat::zeros(erode1.size(), CV_32SC1);
  for(int idx = 0; idx < big_contours.size(); idx++) {
    drawContours(markers, big_contours, idx, Scalar::all(idx + 1), -1, 8);
  }

  // Water shed.
  watershed(main, markers);

  // Find out the color for each water shed marker area.
  std::vector<std::vector<float> > depth_by_marker(big_contours.size());
  for (int i = 0; i < inliners1.size(); i++) {
    int index = markers.at<int>(inliners1[i].y, inliners1[i].x);
    if (index >= 1 && index <= big_contours.size()) {
      depth_by_marker[index - 1].push_back(normalized_depths[i]);
    }
  }
  
  std::vector<int> avg_depth(big_contours.size());
  for (int i = 0; i < depth_by_marker.size(); i++) {
    if (depth_by_marker[i].size() < 3) {
      avg_depth[i] = 0;
      continue;
    }

    double total_depth = 0.0;
    for (int j = 0; j < depth_by_marker[i].size(); j++) {
      total_depth += depth_by_marker[i][j];
    }
    avg_depth[i] = (int)(total_depth / depth_by_marker[i].size());
  }

  Mat wshed(markers.size(), CV_8UC1);
  for (int i = 0; i < markers.rows; i++) {
    for(int j = 0; j < markers.cols; j++) {
      int index = markers.at<int>(i, j);
      if (index >= 1 && index <= avg_depth.size()) {
	wshed.at<char>(i, j) = avg_depth[index - 1];
      }
    }
  }

  imshow("wshed", wshed);

  // Wait until any key is pressed.
  if (waitKey(100) > 0) {
    break;
  }
  }
}

int main(int argc, char** argv) {
  RealMain();
  return 0;
}

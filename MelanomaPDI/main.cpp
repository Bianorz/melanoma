/*
 *This code describes an example of how to segmentate a melanoma image
 *It's used the Otsu method for segmentation and for labelling it's used the distance between melanoma centroid and image center
 */
#include "opencv2/opencv.hpp"
#include "useful_dip_lib.h"
#include <sstream>
#include <ctime>
#include <stack>

using namespace cv;
using namespace std;

/*img: Used to store the melanoma image
 *backup: A backup from the melanoma img
 *gray: Grayscale conversion from img
 *binary: gray after segmentation
 *blurred: binary after median 3x3 filter
 *rows,cols: img's size
 *area: variable used to store the coutour's areas, it will used the function 'Moments' to get this value, details can be found here:
 *http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
 *resultBeforeCrop: black image with segmented melanoma but with same original size
 *resultAfterCrop: black image after bounding rectangle crop
 *HSV: image at HSV space
 *channels: use to split the hsv image in h,s,v
 *hue: image with only Hue plane
 * */

std::stack<clock_t> tictoc_stack;

void tic() {
	tictoc_stack.push(clock());
}

void toc() {
	std::cout << "Time elapsed: "
			<< ((double) (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
			<< " sec" << std::endl;
	tictoc_stack.pop();
}

int main(int argc, char** argv) {
	//detLine();
	//tic();
	String result;
	String resultTxt;
	for (int cont = 1; cont <= 114; cont++) {
		cout << cont <<endl;
		string ext = ".jpg";
		string inic = "";
		Size size(550, 450);
		Mat gray, binary, blurred, resultBeforeCrop, backup, resultAfterCrop,
				HSV, hue, binaryHue, blurredHue;
		stringstream path, pathTxt;
		path << inic << cont << ext;
		pathTxt << "good_results/" << cont << ".txt";

		result = path.str();
		resultTxt = pathTxt.str();
		Mat img = imread(result);
		//resize(img,img,size);

		if (!img.data) {
			cout << "File not found" << endl;
			//return -1;
		} else
		{
			Mat histogram;
			vector<Mat> channels;
			img.copyTo(backup);
			img.copyTo(HSV);
			ofstream file;

			file.open(resultTxt.c_str());
			//namedWindow("Original Image", WINDOW_NORMAL);
			//imshow("Original Image", img);
			//waitKey(0);
			int rows = img.rows;
			int cols = img.cols;
			int area = 0;
			cvtColor(img, gray, COLOR_BGR2GRAY);
			threshold(gray, binary, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
			medianBlur(binary, blurred, 3);

			/*cvtColor(img, HSV, COLOR_BGR2HSV);
			 split(HSV, channels);
			 channels[0].copyTo(hue);
			 histogram = histCalc(hue, 180);
			 histPlot(histogram,180,hue.cols,hue.rows);
			 for (int i = -1; i < 180; i++) {
			 threshold(hue, binaryHue, i, 255, THRESH_BINARY);
			 medianBlur(binaryHue, blurredHue, 3);
			 namedWindow("HUE - Image after segmentation + median filter",
			 WINDOW_NORMAL);
			 imshow("HUE - Image after segmentation + median filter", blurredHue);
			 cout << i << endl;
			 waitKey(0);
			 }*/
		//	namedWindow("Image after segmentation + median filter",
		//			WINDOW_NORMAL);
		//	imshow("Image after segmentation + median filter", blurred);
		//	waitKey(0);
			/*
			 *Contour detection using findContours function, details can be found here: http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
			 */
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(blurred, contours, hierarchy, CV_RETR_CCOMP,
					CV_CHAIN_APPROX_SIMPLE);
			/*
			 * Creation of a bouding box to enclose the segmented image
			 */
			vector<Rect> boundRect(contours.size());
			for (size_t i = 0; i < contours.size(); i++) {
				boundRect[i] = boundingRect(Mat(contours[i]));
			}
			float minDist = 9999;
			int myIndex;
			/*
			 * The hierarchy.size() is used to know if some contour is detected, after that a loop through the contour is initiated.
			 * This version excludes areas below 5000 pixels squared, to avoid contours that aren't significant (this need to be way improved)
			 * 'xb' and 'yb' are used to get the contour centroid, more details can be found here: http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
			 * 'distance' is the way that its decided what contour is the melanoma, usually the melanoma will be in the center of the image,
			 * so if the image centroid is closer than 900 pixels from the img centroid, then it's considered to be the melanoma contour (Need improvement too)
			 * */
			if (hierarchy.size() > 0) {
				for (int index = 0; index >= 0; index = hierarchy[index][0]) {
					Moments moment = moments((cv::Mat) contours[index]);
					area = moment.m00;
					if (area > 5000) {
						int xb = moment.m10 / area;
						int yb = moment.m01 / area;
						float distance = sqrt(
								pow((xb - cols / 2), 2)
										+ pow((yb - rows / 2), 2));
						//cout << "area = " << area << "distance = " << distance
						//	<< " n contornos: " << contours.size() << endl;
						//circle(img, Point(cols / 2, rows / 2), 5,
							//	Scalar(255, 255, 255), 5);
						//circle(img, Point(xb, yb), 5, Scalar(0, 255, 0), 5);
						//drawContours(img, contours, index, Scalar(0, 255, 0), 2,
							//	8, hierarchy, 0, Point());
						//namedWindow("Detected Points", WINDOW_NORMAL);
						//imshow("Detected Points", img);
						//waitKey(0);

						if (distance < minDist) {
							minDist = distance;
							myIndex = index;
						}
					}
				}

				Rect boxContour(boundRect[myIndex].tl(),
						boundRect[myIndex].br());
				Mat black(img.size(), CV_8UC3, Scalar(0, 0, 0));
				cout << myIndex << endl;
				drawContours(black, contours, myIndex, Scalar(255, 255, 255),
						-1, 8, hierarchy, 0, Point());
				//namedWindow("White filled contour", WINDOW_NORMAL);
				//imshow("White filled contour", black);
				bitwise_and(black, backup, resultBeforeCrop);
				resultAfterCrop = resultBeforeCrop(boxContour);
				//cout << index << " " << myIndex << endl;

				vector<Vec3b> color;
				Vec3b bgrPixel;
				for (int i = 0; i < resultAfterCrop.rows; i++) {
					for (int j = 0; j < resultAfterCrop.cols; j++) {
						bgrPixel = resultAfterCrop.at<Vec3b>(i, j);
						if (bgrPixel != Vec3b(0, 0, 0)) {
							Mat matRGB(1, 1, CV_8UC3);
							matRGB.at<Vec3b>(0, 0) = bgrPixel;
							Mat matHSV;
							cvtColor(matRGB, matHSV, COLOR_BGR2HSV);
							Vec3b hsv = matHSV.at<Vec3b>(0, 0);
							color.push_back(bgrPixel);
							//cout << hsv.val[0];
							file << (int)hsv.val[0] << "\n";
						}
					}
				}

			//	namedWindow("Final Result", WINDOW_NORMAL);
			//	imshow("Final Result", resultAfterCrop);
			//	waitKey(0);

				/****Acessing elements example, vec3b vector
				 *
				 *vector<Vec3b> dedColors;
				 *dedColors.push_back(Vec3b(1,2,3));
				 *dedColors.push_back(Vec3b(4,5,6));
				 *dedColors.push_back(Vec3b(7,8,9));
				 *int uVal = dedColors[2][1];  // reads '8'
				 *
				 ****Another way using one for
				 *
				 *Mat3b result = resultAfterCrop;
				 *for (Mat3b::iterator it = result.begin(); it != result.end(); it++) {
				 *	 if (*it != Vec3b(0, 0, 0)) {
				 *		bgrPixel = *it;
				 *		color.push_back(bgrPixel);
				 *		file << bgrPixel << "\n";
				 *		}
				 *}
				 *
				 */

			}
			//	file.close();
		}			//toc();
	}
	return 0;
}

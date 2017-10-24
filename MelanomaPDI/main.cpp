/*
 *This code describes an example of how to segmentate a melanoma image
 *It's used the Otsu method for segmentation and for labelling it's used the distance between melanoma centroid and image center
 */
#include <sstream>
#include <ctime>
#include <stack>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "melanoma_libs.h"

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

Mat invert_image(cv::Mat const& input) {
	return 255 - input;
}

int main() {
	VideoCapture capture;
	Mat img, crop;
	Mat R, G, B;
	Mat src, directResult, combinedResult, gray, negative, aux, backup,
			resultAfterCrop, resultBeforeCrop;
	double redThresh, greenThresh, blueThresh;
	vector<Mat> channels;
	int rows;
	int cols;
	double area;
	int n_crops = 4;
	double combinedThresh, directThresh;
	int cont = 0;
	int init_height = 150, init_width = 150;
	int xb, yb;
	int contador = 1;
	Point tl;
	Point br;
	Mat C =
			(Mat_<char>(5, 5) << 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 1, 0, 1);
	float descritor[5];
	textura(C, descritor);

	for (int w = 0; w < 5; w++) {
		cout << descritor[w] << endl;
	}

	return 0;
	while (1) {
		capture.open("melanoma_imgs/melanoma.mp4");

		if (!capture.isOpened()) {
			cout << "ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}
		while (capture.get(CV_CAP_PROP_POS_FRAMES)
				< capture.get(CV_CAP_PROP_FRAME_COUNT) - 1) {
			int height = init_height, width = init_width;
			cont = 0;
			capture.read(img);
			img.copyTo(src);
			//	imshow("without crop",img);
			//	waitKey(0);
			while (cont < n_crops) {
				cvtColor(img, gray, COLOR_BGR2GRAY);
				negative = invert_image(gray);
				Moments moment = moments(negative);
				area = moment.m00;
				xb = moment.m10 / area;
				yb = moment.m01 / area;
				tl.x = xb - width;
				if (tl.x < 0)
					tl.x = 0;
				tl.y = yb - height;
				if (tl.y < 0)
					tl.y = 0;
				br.x = xb + width;
				if (br.x > img.cols)
					br.x = img.cols;
				br.y = yb + height;
				if (br.y > img.rows)
					br.y = img.rows;
				Rect boxContour(tl, br);
				crop = img(boxContour);
				crop.copyTo(img);
				height = 0.9 * height;
				width = 0.9 * width;
				//	imshow("with crop",img);
				//	waitKey(0);
				cont++;
			}
			rows = img.rows;
			cols = img.cols;
			split(img, channels);
			cvtColor(img, gray, COLOR_BGR2GRAY);
			img.copyTo(backup);
			channels[0].copyTo(B);
			channels[1].copyTo(G);
			channels[2].copyTo(R);
			redThresh = threshold(R, aux, 0, 255,
					THRESH_BINARY_INV | CV_THRESH_OTSU);
			greenThresh = threshold(G, aux, 0, 255,
					THRESH_BINARY_INV | CV_THRESH_OTSU);
			blueThresh = threshold(B, aux, 0, 255,
					THRESH_BINARY_INV | CV_THRESH_OTSU);
			combinedThresh = round(
					0.299 * redThresh + 0.587 * greenThresh
							+ 0.114 * blueThresh);
			directThresh = threshold(gray, directResult, 0, 255,
					THRESH_BINARY_INV | CV_THRESH_OTSU);
			threshold(gray, combinedResult, combinedThresh, 255,
					THRESH_BINARY_INV);
			cout << "Combined value: " << combinedThresh << " Direct Value: "
					<< directThresh << endl;
			medianBlur(combinedResult, combinedResult, 3);
			medianBlur(directResult, directResult, 3);
			/*	imshow("Using 3 channels ponderation", combinedResult);
			 imshow("Applying directing in the gray scale", directResult);
			 imshow("Gray Image", gray);
			 waitKey(0);
			 */
			// CONTOUR DETECTION
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(combinedResult, contours, hierarchy, CV_RETR_CCOMP,
					CV_CHAIN_APPROX_SIMPLE);
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
					if (area > 0.05 * img.cols * img.rows) {
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
						drawContours(img, contours, index, Scalar(0, 255, 0), 2,
								8, hierarchy, 0, Point());
						namedWindow("Detected Points", WINDOW_NORMAL);
						imshow("Detected Points", img);

						if (distance < minDist) {
							minDist = distance;
							myIndex = index;
						}
					}
				}

				Rect boxContour(boundRect[myIndex].tl(),
						boundRect[myIndex].br());
				Mat black(img.size(), CV_8UC3, Scalar(0, 0, 0));
				//cout << myIndex << endl;
				drawContours(black, contours, myIndex, Scalar(255, 255, 255),
						-1, 8, hierarchy, 0, Point());
				bitwise_and(black, backup, resultBeforeCrop);
				resultAfterCrop = resultBeforeCrop(boxContour);
				namedWindow("White filled contour", WINDOW_NORMAL);
				imshow("White filled contour", resultAfterCrop);
				imshow("original", src);
				cout << contador << endl;
				contador++;
				waitKey(0);

			}
		}

		capture.release();

	}

	/*
	 *
	 *
	 * comandos linux
	 *
	 * renomear
	 *
	 * #!/bin/bash
	 counter=0
	 for file in *jpg; do
	 [[ -f $file ]] && mv -i "$file" $((counter+1)).jpg && ((counter++))
	 done
	 *
	 * gerar video 1 framerate
	 *
	 * ffmpeg -framerate 1 -pattern_type glob -i '*.jpg'     -c:v libx264 -r 1 -pix_fmt yuv420p out.mp4
	 *
	 * resize para 400x500
	 *
	 * convert '*.jpg[500x400!]' -set filename:base "%[base]" "new_folder/%[filename:base].jpg"
	 *
	 *
	 * renomear, dar resize, fazer video, separar bad e good results
	 *
	 *
	 */

	return 0;
}


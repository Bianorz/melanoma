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
	String melanomaFolder = "melanoma_imgs/";
	String nevoFolder = "nevos/";
	String img_extension = ".jpg";
	int num_mel_img = 120;
	int num_nev_img = 22;
	int rows;
	int cols;
	int cont = 0;
	int initH = 150;
	int initW = 150;
	int height;
	int width;
	int xb, yb;
	int contador;
	int n_crops = 5;
	vector<Mat> channels;
	Mat crop;
	Mat R, G, B;
	Mat src, directResult, combinedResult, gray, negative, aux, backup,
			resultAfterCrop, resultBeforeCrop;
	double redThresh, greenThresh, blueThresh;
	double area;
	double combinedThresh, directThresh;
	double reduction_percentage = 0.9; // for crops
	double ratio_lesion_area = 0.05;
	Point topLeft;
	Point bottonRight;
	Size size(500, 400);

	// Analise melanoma

	for (int i = 1; i <= num_mel_img; i++) {
		stringstream imgPath;
		imgPath << melanomaFolder << i << img_extension;
		src = imread(imgPath.str());
		if (!src.data) {
			cout << "File not found" << endl;
			//return -1;
		} else {
			resize(src, src, size);
			cont = 0;
			height = initH;
			width = initW;
			while (cont < n_crops) {
				cvtColor(src, gray, COLOR_BGR2GRAY);
				negative = invert_image(gray);
				Moments moment = moments(negative);
				area = moment.m00;
				xb = moment.m10 / area;
				yb = moment.m01 / area;
				topLeft.x = xb - width;
				if (topLeft.x < 0)
					topLeft.x = 0;
				topLeft.y = yb - height;
				if (topLeft.y < 0)
					topLeft.y = 0;
				bottonRight.x = xb + width;
				if (bottonRight.x > src.cols)
					bottonRight.x = src.cols;
				bottonRight.y = yb + height;
				if (bottonRight.y > src.rows)
					bottonRight.y = src.rows;
				Rect boxContour(topLeft, bottonRight);
				crop = src(boxContour);
				crop.copyTo(src);
				height = reduction_percentage * height;
				width = reduction_percentage * width;
				cont++;
			}
			src.copyTo(backup);
			split(src, channels);
			cvtColor(src, gray, COLOR_BGR2GRAY);
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
			threshold(gray, combinedResult, combinedThresh, 255,
					THRESH_BINARY_INV);
			medianBlur(combinedResult, combinedResult, 3);


			// Contour detection

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(combinedResult, contours, hierarchy, CV_RETR_CCOMP,
					CV_CHAIN_APPROX_SIMPLE);
			vector<Rect> boundRect(contours.size());
			for (size_t i = 0; i < contours.size(); i++) {
				boundRect[i] = boundingRect(Mat(contours[i]));
			}
			float minDist = 9999;
			int myIndex = 0;

			if (hierarchy.size() > 0) {
				for (int index = 0; index >= 0; index = hierarchy[index][0]) {
					Moments moment = moments((cv::Mat) contours[index]);
					area = moment.m00;
					if (area > ratio_lesion_area * src.cols * src.rows) {
						int xb = moment.m10 / area;
						int yb = moment.m01 / area;
						float distance = sqrt(
								pow((xb - src.cols / 2), 2)
										+ pow((yb - src.rows / 2), 2));

						if (distance < minDist) {
							minDist = distance;
							myIndex = index;
						}
					}
				}

				Rect boxContour(boundRect[myIndex].tl(),boundRect[myIndex].br());
				Mat black(src.size(), CV_8UC3, Scalar(0, 0, 0));
				//cout << myIndex << endl;
				drawContours(black, contours, myIndex, Scalar(255, 255, 255),
						-1, 8, hierarchy, 0, Point());
				bitwise_and(black, backup, resultBeforeCrop);
				resultAfterCrop = resultBeforeCrop(boxContour);
				//namedWindow("Result", WINDOW_NORMAL);
				imshow("Result", resultAfterCrop);
				imshow("original", backup);
				waitKey(0);
				cout << "Resultado aceitavel? (1/sim) (0/nao)"<<endl;
				cin >> contador;

				if (contador == 1){

					String good_res_folder = "good_results/";
					stringstream good_res_path;
					stringstream good_orig_path;
					good_res_path << good_res_folder << i << img_extension;
					//good_orig_path << good_res_folder << "orig"<<i
					imwrite(good_res_path.str(),resultAfterCrop);

				} else {

					String bad_res_folder = "bad_results/";
					stringstream bad_res_path;
					bad_res_path << bad_res_folder << i << img_extension;
					imwrite(bad_res_path.str(),resultAfterCrop);
				}

			}

		}
	}

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
 *Mat C =
			(Mat_<char>(5, 5) << 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 1, 0, 1);
	float descritor[5];
	textura(C, descritor);
 *
 * renomear, dar resize, fazer video, separar bad e good results
 *
 *
 */


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

Mat invert_image(cv::Mat const& input) {
	return 255 - input;
}

int main(int argc, char** argv) {
	Size size(500, 400);
	vector<Mat> channels;
	Mat R, G, B;
	Mat Rh, Gh, Bh, result, resultOtsu, gray, negative;
	double redThresh, greenThresh, blueThresh;
	int combinedThresh;
	Mat img = imread("bad_results/94.jpg");
	if (!img.data) {
		cout << "file not found";
		return -1;
	}
	resize(img, img, size);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	negative = invert_image(gray);
	Moments moment = moments(negative);
	double area = moment.m00;
	// Get x and y position of the object
	int xb, yb;
	xb = moment.m10 / area;
	yb = moment.m01 / area;
	int altura = 100, largura = 100;
	Point tl(xb - largura, yb - altura);
	Point br(xb + largura, yb + altura);
	Rect boxContour(tl, br);
	B = img(boxContour);
	imshow("corte",B);
	for (int i = 2; i <= 400; i = i + 2) {

		cvtColor(B, gray, COLOR_BGR2GRAY);
		negative = invert_image(gray);
		moment = moments(negative);
		area = moment.m00;
		xb = moment.m10 / area;
		yb = moment.m01 / area;
		altura = altura/2;
		largura = largura/2;
		tl.x = xb - largura;
		tl.y = yb - altura;
		br.x = xb + largura;
		br.y = yb + altura;
		Rect box(tl,br);
		B = B(box);
		imshow("corte2",B);
		waitKey(0);
		cout << i << " "<<area<<endl;
	}



	/*split(img, channels);
	 channels[0].copyTo(B);
	 channels[1].copyTo(G);
	 channels[2].copyTo(R);
	 redThresh = threshold(R, resultOtsu, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	 greenThresh = threshold(G, resultOtsu, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	 blueThresh = threshold(B, resultOtsu, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	 cout << redThresh<<" " << greenThresh << " "<< blueThresh<< endl;
	 combinedThresh = round(0.299*redThresh + 0.587*greenThresh + 0.114*blueThresh);
	 cout << "Combined value: "<<combinedThresh << endl;
	 cout << "direct value: "<<threshold(gray, resultOtsu, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU) << endl;
	 threshold(gray, result, 129, 255, THRESH_BINARY_INV);
	 medianBlur(result, result, 3);
	 medianBlur(resultOtsu, resultOtsu, 3);
	 imshow("Using 3 channels ponderation",result);
	 imshow("Applying directing in the gray scale",resultOtsu);
	 imshow("Gray Image",gray);
	 Rh = histCalc(gray,256);
	 histPlot(Rh,256,gray.cols,gray.rows,"histrograma gray");*/
	cout << xb << " " << yb << endl;
	waitKey(0);
	return 0;
}

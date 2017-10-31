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



int main() {
	String melanomaFolder = "melanoma_imgs/";
	String nevoFolder = "nevo_test/";
	String img_extension = ".jpg";
	int num_mel_img = 120;
	int num_nev_img = 29;

	vector<Mat> channels;
	Mat crop;
	Mat src, original;
	Point topLeft;
	Point bottonRight;
	Size size(500, 400);
	float* descTxt;
	// Analise melanoma

	for (int i = num_nev_img+1; i <= 100; i++) {
		stringstream imgPath;
		Mat C =(Mat_<char>(2, 3) << 1,1,0,0,1,0);
		imgPath << nevoFolder << i << img_extension;
		src = imread(imgPath.str());
		if (!src.data) {
			cout << "File not found" << endl;
			//return -1;
		} else {
			src.copyTo(original);
			crop = segment(src);
			descTxt = textura(C);
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


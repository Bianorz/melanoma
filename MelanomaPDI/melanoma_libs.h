using namespace cv;
using namespace std;


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


Mat histCalc(Mat image, int hsize) {
	Mat hist;
	float range[] = { 0, float(hsize) }; // range de cores
	const float* histRange = { range }; // variavel do histograma
	calcHist(&image, 1, 0, Mat(), hist, 1, &hsize, &histRange); // cálculo o histograma de cada vaga e salvo em backHist
	return hist;
}

void histPlot(Mat hist, int histSize, int hist_w, int hist_h, string janela) {
	//int hist_w = hist.cols; int hist_h = hist.rows;
	int bin_w = cvRound((double) hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++) {
		line(histImage,
				Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(255, 255, 255), 3, 8, 0);
	}
	namedWindow(janela.c_str(), CV_WINDOW_NORMAL);
	imshow(janela.c_str(), histImage);
	waitKey(0);
}

Mat detLine() {
	Mat img = imread("images/apple.png");
	if (!img.data) {
		cout << "Arquivo nao encontrado" << endl;
		//return -1;
	}
	Mat cinza, edges;
	cvtColor(img, cinza, COLOR_BGR2GRAY);
	Canny(cinza, edges, 50, 200, 3);
	//namedWindow("Canny", CV_WINDOW_NORMAL);
	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 30, 10);
	for (size_t i = 0; i < lines.size(); i++) {
		line(img, Point(lines[i][0], lines[i][1]),
				Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 5, 8);
		//circle(img, Point(lines[i][0], lines[i][1]), 25, Scalar(0, 255, 0));
		//circle(img, Point(lines[i][2], lines[i][3]), 25, Scalar(0, 255, 0));
		namedWindow("linhas", CV_WINDOW_NORMAL);
		imshow("linhas", img);
		waitKey(0);
	}
	Mat3b src2 = img;
	for (Mat3b::iterator it = src2.begin(); it != src2.end(); it++) {
		if (*it == Vec3b(0, 0, 255)) {
			*it = Vec3b(255, 255, 255);
		}
	}
	namedWindow("linhas apagadas", CV_WINDOW_NORMAL);
	imshow("linhas apagadas", src2);
	waitKey(0);
	return edges;

}

float *textura(Mat C) {
	//cout << "C = " << endl << " " << C << endl << endl;
	float *descritor=new float[5];
	double min, max;
	minMaxLoc(C, &min, &max);
	float contraste = 0, dissimilaridade = 0, homogeneidade = 0, energia = 0,
			entropia = 0, smo = 0;
	float P[int(max) + 1][int(max) + 1] = { 0 };
	int pixelAtual;
	int pixelTarget;
	int soma = 0;
	int height = max + 1;
	int width = height;
	for (int i = 0; i < C.cols - 1; i++) {
		for (int j = 0; j < C.rows - 1; j++) {

			pixelAtual = C.at<char>(Point(j, i));
			pixelTarget = C.at<char>(Point(j + 1, i + 1));

			for (int k = 0; k <= max; k++) {
				for (int m = 0; m <= max; m++) {
					if (pixelAtual == k && pixelTarget == m) {
						P[k][m] = P[k][m] + 1;
					}

				}
			}

		}

	}

	for (int k = 0; k <= max; k++) {
		for (int m = 0; m <= max; m++) {
			soma = soma + P[k][m];
		}
	}
	for (int k = 0; k <= max; k++) {
		for (int m = 0; m <= max; m++) {
			P[k][m] = P[k][m] / soma;
			if (P[k][m] < 0.0000001) {
				P[k][m] = 0;
			}
			cout << P[k][m] << endl;
		}
	}

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			contraste = contraste + P[h][w] * (h - w) * (h - w);
			dissimilaridade = dissimilaridade + P[h][w] * abs(h - w);
			homogeneidade = homogeneidade + P[h][w] / (1 + (h - w) * (h - w));
			smo = smo + P[h][w] * P[h][w];
			if (P[h][w] != 0) {
				entropia = entropia + P[h][w] * -1 * log(P[h][w]);
			}
		}
	}
	energia = sqrt(smo);
	descritor[0] = contraste;
	descritor[1] = dissimilaridade;
	descritor[2] = homogeneidade;
	descritor[3] = energia;
	descritor[4] = entropia;
	return descritor;
}



Mat segment(Mat src){
	Mat directResult, combinedResult, gray, negative, aux, backup,
			resultAfterCrop, resultBeforeCrop, original,crop,B,G,R;

	int cont = 0;
	int initH = 150;
	int initW = 150;
	int height;
	int width;
	int xb, yb;
	int n_crops = 5;
	vector<Mat> channels;

	double redThresh, greenThresh, blueThresh;
	double area;
	double combinedThresh;
	double reduction_percentage = 0.9; // for crops
	double ratio_lesion_area = 0.05;

	Point topLeft;
	Point bottonRight;
	Size size(500, 400);

	src.copyTo(original);
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
		drawContours(black, contours, myIndex, Scalar(255, 255, 255),
				-1, 8, hierarchy, 0, Point());
		bitwise_and(black, backup, resultBeforeCrop);
		resultAfterCrop = resultBeforeCrop(boxContour);



	}
	return resultAfterCrop;
}
 /* KNN RASCUNHO
// GenData.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	/*
	 cv::Mat matClassificationInts;
	 cv::Mat matTrainingImagesAsFlattenedFloats;

	 std::vector<float> opa1;
	 std::vector<float> opa2;

	 opa1.push_back(7);
	 opa1.push_back(8);
	 opa1.push_back(9);
	 opa1.push_back(7);
	 opa1.push_back(6);

	 opa2.push_back(20);
	 opa2.push_back(21);
	 opa2.push_back(18);
	 opa2.push_back(17);
	 opa2.push_back(19);

	 cv::Mat C = (cv::Mat_<float>(1, 4) << 1, 2, 3, 4);
	 cv::Mat D = (cv::Mat_<float>(1, 4) << 2, 4, 6, 8);
	 matClassificationInts.push_back(1);
	 matTrainingImagesAsFlattenedFloats.push_back(C);
	 matClassificationInts.push_back(2);
	 matTrainingImagesAsFlattenedFloats.push_back(D);
	 std::cout << "training complete\n\n";

	 cv::FileStorage fsClassifications("classifications.xml",
	 cv::FileStorage::WRITE); // open the classifications file

	 if (fsClassifications.isOpened() == false) { // if the file was not opened successfully
	 std::cout
	 << "error, unable to open training classifications file, exiting program\n\n"; // show error message
	 return (0); // and exit program
	 }

	 fsClassifications << "classifications" << matClassificationInts; // write classifications into classifications section of classifications file
	 fsClassifications.release(); // close the classifications file

	 // save training images to file ///////////////////////////////////////////////////////

	 cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE); // open the training images file

	 if (fsTrainingImages.isOpened() == false) { // if the file was not opened successfully
	 std::cout
	 << "error, unable to open training images file, exiting program\n\n"; // show error message
	 return (0); // and exit program
	 }

	 fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats; // write training images into images section of images file
	 fsTrainingImages.release();// close the training images file

FIM TREINAMENTO


	cv::Mat matClassificationInts; // we will read the classification numbers into this variable as though it is a vector

	cv::FileStorage fsClassifications("classifications.xml",
			cv::FileStorage::READ); // open the classifications file

	if (fsClassifications.isOpened() == false) { // if the file was not opened successfully
		std::cout
				<< "error, unable to open training classifications file, exiting program\n\n"; // show error message
		return (0); // and exit program
	}

	fsClassifications["classifications"] >> matClassificationInts; // read classifications section into Mat classifications variable
	fsClassifications.release(); // close the classifications file

	cv::Mat matTrainingImagesAsFlattenedFloats; // we will read multiple images into this single image variable as though it is a vector

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ); // open the training images file

	if (fsTrainingImages.isOpened() == false) { // if the file was not opened successfully
		std::cout
				<< "error, unable to open training images file, exiting program\n\n"; // show error message
		return (0); // and exit program
	}

	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats; // read images section into Mat training images variable
	fsTrainingImages.release(); // close the traning images file

	cv::Ptr<cv::ml::KNearest> kNearest(cv::ml::KNearest::create()); // instantiate the KNN object

	// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
	// even though in reality they are multiple images / numbers
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE,
			matClassificationInts);

  //  cv::Mat matROIFlattenedFloat = (cv::Mat_<float>(1, 1) << 3);

  //  cv::Mat matCurrentChar(0, 0, CV_32F);

   // kNearest->findNearest(matROIFlattenedFloat, 3, matCurrentChar);     // finally we can call find_nearest !!!

   // float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

   // std::cout << "classe: " << fltCurrentChar << std::endl;


	return 0;
}

 *
 */



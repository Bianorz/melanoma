using namespace cv;
using namespace std;

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

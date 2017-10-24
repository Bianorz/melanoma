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

void textura(Mat C, float* descritor) {
	//cout << "C = " << endl << " " << C << endl << endl;
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
	int n = height;
	float** array2D = 0;
	array2D = new float*[height];
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

}

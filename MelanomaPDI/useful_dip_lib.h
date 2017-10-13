// Fun��es para o estacionamento inteligente
#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>   
#include <fstream>

using namespace cv;
using namespace std;

Mat pegarImagemCortada(Mat src, Point2f centro, Size2f tamanho, int angulo) {
	Mat imagemCortada;
	Size tamImagemOriginal = src.size();
	RotatedRect rRect = RotatedRect(centro, tamanho, angulo);
	Point2f vertices[4];
	rRect.points(vertices);
	vector<Point> tri2CroppedInt;
	for (int i = 0; i < 4; ++i) {
		tri2CroppedInt.push_back(vertices[i]);
	}
	if (src.type() != 0) {
		Mat preta(tamImagemOriginal, CV_8UC3, Scalar(0, 0, 0));
		fillConvexPoly(preta, tri2CroppedInt, Scalar(255, 255, 255));
		bitwise_and(src, preta, imagemCortada);
	} else {
		Mat preta(tamImagemOriginal, CV_8U, Scalar(0));
		fillConvexPoly(preta, tri2CroppedInt, Scalar(255));
		bitwise_and(src, preta, imagemCortada);

	}

	Rect brect = rRect.boundingRect();
	imagemCortada = imagemCortada(brect);
	return imagemCortada;
}

void apenasParaCompletarParametros(int, void*) {
	// � chamado quando se usa os sliders, necess�rio passar uma fun��o como par�metro
}

void salvarLocalVagas(String nomeImagem, int nVagas, String nomeArqTxt) {
	int centro_x = 1;
	int centro_y = 1;
	int largura = 1;
	int altura = 1;
	int inclinacao = 0;
	int fim = 0;
	const string nomeJanela = "Interface de Marca��o";
	namedWindow(nomeJanela, 0);
	createTrackbar("horizontal", nomeJanela, &centro_x, 1000,
			apenasParaCompletarParametros);
	createTrackbar("vertical", nomeJanela, &centro_y, 1000,
			apenasParaCompletarParametros);
	createTrackbar("largura", nomeJanela, &largura, 1000,
			apenasParaCompletarParametros);
	createTrackbar("altura", nomeJanela, &altura, 1000,
			apenasParaCompletarParametros);
	createTrackbar("inclinacao", nomeJanela, &inclinacao, 1000,
			apenasParaCompletarParametros);
	createTrackbar("fim", nomeJanela, &fim, 1, apenasParaCompletarParametros);
	Mat imagem = imread(nomeImagem, 1);
	if (!imagem.data) {
		cout << "ERRO - ARQUIVO NAO ENCONTRADO" << endl;
		system("pause");
	}
	ofstream myfile;
	myfile.open(nomeArqTxt.c_str());
	printf(
			"Use o slider para realizar a marcacao, mude 'fim' para '1'quando acabar\n");
	for (int i = 1; i <= nVagas; i++) {
		while (fim == 0) {
			imagem = imread(nomeImagem, 1);
			RotatedRect rRect = RotatedRect(Point2f(centro_x, centro_y),
					Size2f(largura, altura), inclinacao);
			Point2f vertices[4];
			rRect.points(vertices);
			for (int i = 0; i < 4; i++)
				line(imagem, vertices[i], vertices[(i + 1) % 4],
						Scalar(0, 255, 0));

			imshow("rectangles", imagem);
			waitKey(15);
		}
		fim = 0;
		if (myfile.is_open()) {
			myfile << centro_x << " " << centro_y << " " << largura << " "
					<< altura << " " << inclinacao << "\n";
			cout << "Vaga " << i << " salva\n";
		} else {
			cout << "ERRO DE ACESSO AO ARQUIVO TXT \n";
			system("pause");
		}
	}
	destroyAllWindows();
	// Checo se arquivo est� aberto e escrevo os valores HSV
	if (myfile.is_open()) {
		myfile << "\ncentro_x centro_y largura altura inclinacao";
		myfile.close();
	} else {
		cout << "ERRO DE ACESSO AO ARQUIVO TXT \n";
		system("pause");
	}
	cout << "Marca��o concluida com sucesso, valores determinados salvos em "
			<< nomeArqTxt << endl;
	system("pause");

}

struct Vaga {
	Point2f posicao;
	int largura;
	int altura;
	int inclinacao;
};

void carregarParametros(String nomeArqTxt, Vaga vaga[], int nVagas) {
	ifstream file;
	file.open(nomeArqTxt.c_str());
	for (int i = 0; i < nVagas; i++) {
		file >> vaga[i].posicao.x >> vaga[i].posicao.y >> vaga[i].largura
				>> vaga[i].altura >> vaga[i].inclinacao;
	}
	file.close();
}

void mostrarMapeamento(Mat imagem, Vaga vaga[], int nVagas) {
	//Mat imagem = imread(nomeImagem, 1);
	if (!imagem.data) {
		cout << "ERRO - ARQUIVO NAO ENCONTRADO" << endl;
		system("pause");
	}
	for (int i = 0; i < nVagas; i++) {
		RotatedRect rRect = RotatedRect(
				Point2f(vaga[i].posicao.x, vaga[i].posicao.y),
				Size2f(vaga[i].largura, vaga[i].altura), vaga[i].inclinacao);
		Point2f vertices[4];
		rRect.points(vertices);
		for (int i = 0; i < 4; i++)
			line(imagem, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
	}

	imshow("mapa", imagem);
	waitKey(15);
}

int numeroContornos(Mat imagem) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat imagemBorrada, bordas;
	GaussianBlur(imagem, imagemBorrada, Size(5, 5), 0, 0);
	Canny(imagemBorrada, bordas, 35, 125);
	findContours(bordas, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_SIMPLE);
	int numeroBordas = hierarchy.size();
	return numeroBordas;
}

float mediaArray(float vetor[], int buff) {
	float sum = 0, media = 0;
	for (int i = 0; i < buff; i++) {
		sum = sum + vetor[i];
	}
	media = ((float) sum) / buff;
	return media;
}

Mat histCalc(Mat image, int hsize) {
	Mat hist;
	int histSize = hsize;  // número de cores do meu histograma
	float range[] = { 0, hsize }; // range de cores
	const float* histRange = { range }; // variavel do histograma
	calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange); // cálculo o histograma de cada vaga e salvo em backHist
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
	//waitKey(0);
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
		line(img, Point(lines[i][0], lines[i][1]),Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 5, 8);
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

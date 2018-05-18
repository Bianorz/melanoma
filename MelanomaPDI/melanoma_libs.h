#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

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

float *get_texture_descriptors(Mat C) {
	//cout << "C = " << endl << " " << C << endl << endl;
	Mat grayImage;
	cvtColor(C, grayImage, COLOR_BGR2GRAY);
	grayImage.copyTo(C);
	float *descritor = new float[5];
	double min, max;
	minMaxLoc(C, &min, &max);
	float contraste = 0, dissimilaridade = 0, homogeneidade = 0, energia = 0,
			entropia = 0, smo = 0;
	float co_ocurrence_matrix[int(max) + 1][int(max) + 1] = { 0 };
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
						co_ocurrence_matrix[k][m] = co_ocurrence_matrix[k][m] + 1;
					}

				}
			}

		}

	}

	for (int k = 1; k <= max; k++) {
		for (int m = 1; m <= max; m++) {
			soma = soma + co_ocurrence_matrix[k][m];
		}
	}
	for (int k = 1; k <= max; k++) {
		for (int m = 1; m <= max; m++) {
			co_ocurrence_matrix[k][m] = co_ocurrence_matrix[k][m] / soma;
			if (co_ocurrence_matrix[k][m] < 0.00001) {
				co_ocurrence_matrix[k][m] = 0;
			}
		}
	}
	// beginning in '1' because I don't want to process the black(0) color
	for (int h = 1; h < height; h++) {
		for (int w = 1; w < width; w++) {
			contraste = contraste + co_ocurrence_matrix[h][w] * (h - w) * (h - w);
			dissimilaridade = dissimilaridade + co_ocurrence_matrix[h][w] * abs(h - w);
			homogeneidade = homogeneidade + co_ocurrence_matrix[h][w] / (1 + (h - w) * (h - w));
			smo = smo + co_ocurrence_matrix[h][w] * co_ocurrence_matrix[h][w];
			if (co_ocurrence_matrix[h][w] != 0) {
				entropia = entropia + co_ocurrence_matrix[h][w] * -1 * log(co_ocurrence_matrix[h][w]);
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

Mat get_segmented_image(Mat src) {
	Mat directResult, combinedResult, gray, negative, aux, backup,
			resultAfterCrop, resultBeforeCrop, original, crop, B, G, R;

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
	Size size(625, 500);

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
	redThresh = threshold(R, aux, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	greenThresh = threshold(G, aux, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	blueThresh = threshold(B, aux, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	combinedThresh = round(
			0.299 * redThresh + 0.587 * greenThresh + 0.114 * blueThresh);
	threshold(gray, combinedResult, combinedThresh, 255, THRESH_BINARY_INV);
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

		Rect boxContour(boundRect[myIndex].tl(), boundRect[myIndex].br());
		Mat black(src.size(), CV_8UC3, Scalar(0, 0, 0));
		drawContours(black, contours, myIndex, Scalar(255, 255, 255), -1, 8,
				hierarchy, 0, Point());
		bitwise_and(black, backup, resultBeforeCrop);
		resultAfterCrop = resultBeforeCrop(boxContour);

	}
	return resultAfterCrop;
}

int file_number(String folder_path) {
	DIR *dp;
	int i = 0;
	struct dirent *ep;
	dp = opendir(folder_path.c_str());

	if (dp != NULL) {
		while ((ep = readdir(dp)))
			i++;

		(void) closedir(dp);
	} else
		perror("Couldn't open the directory");
	return (i - 2); // the fuction count the number of files + 2,
	//I don't know why
}


void rename_files(String path) {
	stringstream imgPath;
	imgPath << "./rename.sh " << path;
	system(imgPath.str().c_str());
}


/**
    Used to get the image features for the KNN classifier

    @param nevFolder Folder with the nevo images.
    @param melFolder Folder with the melanoma images.
    @param s_perc Percentage of the total amount of images used for training.
    @param trainDataFile File where the texture descriptors will be stored.
    @param responsesFile File where the responses(0 for nevo|1 for melanoma) will be stored.
    @param training True to get texture for training, False to get texture for testing.
*/
void get_texture_data(String nevFolder, String melFolder, int s_perc,
		String trainDataFile, String responsesFile, bool training) {
	// Variable used to store the image texture descriptors
	float* texture_descriptors;

	// The images inside both folders are renamed to numbers in crescent order, ex: 1.jpg, 2.jpg, 3.jpg...
	rename_files(nevFolder);
	rename_files(melFolder);

	// Matrix used to store the source and segmented images
	Mat src_image, segmented_image;

	// Get the number of images inside each folder
	int number_of_nevos = file_number(nevFolder);
	int number_of_melanomas = file_number(melFolder);

	// Get the percentual amount of images used for training
	int nevos_for_training = round(number_of_nevos * s_perc / 100);
	int melanomas_for_training = round(number_of_melanomas * s_perc / 100);

	// Open the files where the traindata and responses will be stored
	ofstream trainData, responses;
	trainData.open(trainDataFile.c_str());
	responses.open(responsesFile.c_str());

	// Variable used to index the images
	int begin_nevos = 0, end_nevos = 0, begin_melanoma = 0, end_melanoma = 0;

	if (training) {
		// Show the amount of images used for training
		cout << nevos_for_training + melanomas_for_training
				<< " images used to train the KNN classifier:" << endl;
		cout << nevos_for_training << " nevos and " << melanomas_for_training
				<< " melanoma images." << endl;

		// Save this info in a txt file called training_info.txt in the following way: total_of_images nevos melanomas
		ofstream train_file;
		train_file.open("training_info.txt");
		train_file << nevos_for_training + melanomas_for_training << " "<<nevos_for_training <<" "<< melanomas_for_training;
		train_file.close();

		begin_nevos = 1;
		end_nevos = nevos_for_training;
		begin_melanoma = 1;
		end_melanoma = melanomas_for_training;
	} else {
		// Show the amount of images used for testing
		cout << (number_of_nevos - nevos_for_training)+(number_of_melanomas-melanomas_for_training)
				<< " images used to test the KNN classifier:" << endl;
		cout << number_of_nevos - nevos_for_training << " nevos and " << number_of_melanomas-melanomas_for_training
				<< " melanoma images." << endl;

		// Save this info in a txt file called test_info.txt in the following way: total_of_images nevos melanomas
		ofstream test_file;
		test_file.open("test_info.txt");
		test_file << (number_of_nevos - nevos_for_training)+(number_of_melanomas-melanomas_for_training) << " "<<number_of_nevos - nevos_for_training <<" "<< number_of_melanomas-melanomas_for_training;
		test_file.close();

		begin_nevos = nevos_for_training + 1;
		end_nevos = number_of_nevos;
		begin_melanoma = melanomas_for_training + 1;
		end_melanoma = number_of_melanomas;
	}

	// Time to capture the texture descriptors from the nevos
	for (int i = begin_nevos; i <= end_nevos; i++) {

		stringstream imgPath;
		imgPath << nevFolder << i << ".jpg";
		src_image = imread(imgPath.str());
		if (!src_image.data) {
			cout << "File not found" << endl;
		} else {

			segmented_image = get_segmented_image(src_image);
			char filename[120];
			sprintf(filename,"/home/gdaco001/melanoma/MelanomaPDI/cropped_nevo_database/%d.jpg", i);
			imwrite(filename,segmented_image);
			texture_descriptors = get_texture_descriptors(segmented_image);

			for (int j = 0; j < 5; j++) {
				trainData << texture_descriptors[j] << " ";
			}
			responses << "0" << "\n";
			trainData << "\n";
		}
	}

	// Time to capture the texture descriptors from the melanomas
	for (int i = begin_melanoma; i <= end_melanoma; i++) {
		stringstream imgPath;
		imgPath << melFolder << i << ".jpg";
		src_image = imread(imgPath.str());
		if (!src_image.data) {
			cout << "File not found" << endl;
		} else {
			segmented_image = get_segmented_image(src_image);
			char filename[120];
			sprintf(filename,"/home/gdaco001/melanoma/MelanomaPDI/cropped_melanoma_database/%d.jpg", i);
			imwrite(filename,segmented_image);
			texture_descriptors = get_texture_descriptors(segmented_image);

			for (int j = 0; j < 5; j++) {

				trainData << texture_descriptors[j] << " ";
			}
			responses << "1" << "\n";
			trainData << "\n";
		}
	}
	trainData.close();
	responses.close();
	if (training) {
		cout << "Database ready for training" << endl;
	} else {
		cout << "Database ready for testing" << endl;
	}
}

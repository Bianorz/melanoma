#include <sstream>
#include <ctime>
#include <stack>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "melanoma_libs.h"
#include "histogram.h"
using namespace cv;
using namespace std;

int main() {

	tic();
	// Folder inside my project where all my nevo images are
	String nevoFolder = "nevo_database/";
	// Folder inside my project where all my melanoma images are
	String melanomaFolder = "melanoma_database/";

	/*
	Files used to train the knn classifier.
	In the first trainData.csv row we will have the information of the contrast,
	dissimilarity, homogeneity, energy and entropy descritors of one image.
	The responses.csv file will indicate that these descritors are from a
	nevo (0) or melanoma (1) image.
	*/
	String trainDataFile = "trainData.csv"; // File
	String responsesFile = "responses.csv";


	// Files used to check the results of the knn classifier.
	String testDataFile = "testData.csv";
	String realValuesTestData = "realData.csv";

	// Database's percentage used for training the knn classification
	int samples_training_percentange = 70;

	// Get features to train the KNN classifier
	get_texture_data(nevoFolder, melanomaFolder, samples_training_percentange,trainDataFile,responsesFile,true);

	// Get features to test the KNN classifier
	get_texture_data(nevoFolder, melanomaFolder, samples_training_percentange,testDataFile,realValuesTestData,false);
	toc();

	system("python energy_KNN.py");


}

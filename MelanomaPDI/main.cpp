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

	String nevoFolder = "nevo_database/";
	String melanomaFolder = "melanoma_database/";

	String trainDataFile = "trainData.csv";
	String responsesFile = "responses.csv";

	String testDataFile = "testData.csv";
	String realValuesTestData = "realData.csv";

	// Database's percentage used for training the knn classification
	int samples_training_percentange = 70;

	get_training_data(nevoFolder, melanomaFolder, samples_training_percentange,trainDataFile,responsesFile);

	get_test_data(nevoFolder, melanomaFolder, samples_training_percentange,testDataFile,realValuesTestData);
	toc();
}

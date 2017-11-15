#include <sstream>
#include <ctime>
#include <stack>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "melanoma_libs.h"

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

	// Features are {contrast,dissimilarity,homogeneity,energy,entropy}
	// ORDER MATTER! Use '1' to select, 0 to deselect
	bool texture_features_selection[] = { 1, 1, 1, 1, 1 };

	get_training_data(nevoFolder, melanomaFolder, samples_training_percentange,
			texture_features_selection,trainDataFile,responsesFile);

	get_test_data(nevoFolder, melanomaFolder, samples_training_percentange,
			texture_features_selection,testDataFile,realValuesTestData);
	toc();
}

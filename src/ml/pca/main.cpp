#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "Eigen.h"

using namespace cv;
using namespace std;

vector<Mat> imageLoad2(string path) {
	vector<String> imageNames;
	vector<Mat> images;

	glob(path, imageNames, false);

	int numOfImage = imageNames.size();

	for (int i = 0; i < numOfImage; i++) {
		images.push_back(imread(imageNames[i]));
	}
	return images;
}

vector<Mat> imageLoad(string path, int numOfImage) {
	vector<String> imageNames;
	vector<Mat> images;

	glob(path, imageNames, false);

	//int numOfImage = imageNames.size();

	for (int i = 0; i < numOfImage; i++) {
		Mat ycrcb;
		cvtColor(imread(imageNames[i]), ycrcb, CV_BGR2YCrCb);

		Mat GrayIm;

		Mat channels[3];
		split(ycrcb, channels);

		GrayIm = channels[0];

		Mat GrayImD;
		GrayIm.convertTo(GrayImD, CV_64F, 1.f / 255);
		images.push_back(GrayImD);
	}
	return images;

}

Mat imagesResize(vector<Mat> images, int numOfImage) {
	int numOfRows = images[0].rows;
	int numOfCols = images[0].cols;
	Mat resizedImage = Mat1d(numOfRows*numOfCols, numOfImage, 0.0);
	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfCols; j++) {
			for (int n = 0; n < numOfImage; n++) {
				resizedImage.at <double>(i*numOfCols + j, n) = images[n].at <double>(i, j);
			}
		}
	}
	return resizedImage;
}

Mat getMeanFace(vector<Mat> images, int numOfImage) {
	int numOfRows = images[0].rows;
	int numOfCols = images[0].cols;
	Mat MeanFace = Mat1d(numOfRows*numOfCols, 1, 0.0);

	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfCols; j++) {
			for (int n = 0; n < numOfImage; n++) {
				MeanFace.at <double>(i*numOfCols + j) = MeanFace.at<double>(i*numOfCols + j) + images[n].at <double>(i, j);
			}
		}
	}
	MeanFace = MeanFace / numOfImage;
	return MeanFace;
}

Mat getDemeanFaces(Mat images, Mat meanFace, int numOfImage, int numOfRows, int numOfCols) {
	Mat demeanImage = Mat1d(numOfRows*numOfCols, numOfImage, 0.0);
	for (int i = 0; i < images.cols; i++) {
		demeanImage.col(i) = images.col(i) - meanFace;
	}
	return demeanImage;
}

Mat getHR(Mat imTest, int upscale, Mat MeanFaceLR, Mat MeanFaceHR, int numofImage, Mat EigenVecLR, Mat EigenValLRN, Mat demeanImageLR, Mat demeanImageHR, int numOfRowsHR, int numOfColsHR, double alpha) {


	Mat imTest_ycrcb;
	cvtColor(imTest, imTest_ycrcb, CV_BGR2YCrCb);
	Mat channels[3];
	split(imTest_ycrcb, channels);

	Mat imTest_y = channels[0];
	Mat imTest_cr = channels[1];
	Mat imTest_cb = channels[2];


	imTest_y.convertTo(imTest_y, CV_64F, 1.f / 255);


	int numOfRowsTestLR = imTest_y.rows;
	int numOfColsTestLR = imTest_y.cols;

	Mat ImTest = Mat1d(numOfRowsTestLR*numOfColsTestLR, 1);

	for (int i = 0; i < numOfRowsTestLR; i++) {
		for (int j = 0; j < numOfColsTestLR; j++) {
			ImTest.at <double>(i*numOfColsTestLR + j) = imTest_y.at <double>(i, j);
		}
	}



	for (int i = 0; i < EigenValLRN.rows; i++) {
		if (isnan(EigenValLRN.at<double>(i, i))) {
			EigenValLRN.at<double>(i, i) = 0;
		}
	}

	Mat EigenVal = EigenVecLR * EigenValLRN;

	Mat EigenFac = demeanImageLR * EigenVecLR * EigenValLRN;
	
	Mat temp = (EigenFac.t() * (ImTest - MeanFaceLR));


	
	for (int i = 0; i < numofImage; i++) {
		if ((abs(temp.at<double>(i))) > (alpha * 1.0 / EigenValLRN.at<double>(i, i))) {
			if (temp.at<double>(i) > 0.0) {
				temp.at<double>(i) = alpha * 1.0 / EigenValLRN.at<double>(i, i);
			}
			else {
				temp.at<double>(i) = -alpha * 1.0 / EigenValLRN.at<double>(i, i);
			}
		};
	}
	
	Mat coef = EigenVal * temp;

	Mat ImOut = Mat1d(numOfRowsHR * numOfColsHR, 1, 0.0);

	for (int i = 0; i < numofImage; i++) {
		ImOut = ImOut + demeanImageHR.col(i) * coef.at<double>(i);
	}
	ImOut = ImOut + MeanFaceHR;

	Mat ImResult = Mat1d(numOfRowsHR, numOfColsHR);
	for (int i = 0; i < numOfRowsHR; i++) {
		for (int j = 0; j < numOfColsHR; j++) {
			ImResult.at<double>(i, j) = ImOut.at<double>(i*numOfColsHR + j);
		}
	}
	ImResult.convertTo(imTest_y, CV_8U, 255);

	resize(imTest_cr, imTest_cr, Size(), upscale, upscale, INTER_CUBIC);
	resize(imTest_cb, imTest_cb, Size(), upscale, upscale, INTER_CUBIC);

	vector <Mat> new_channels;
	new_channels.push_back(imTest_y);
	new_channels.push_back(imTest_cr);
	new_channels.push_back(imTest_cb);
	Mat test_hr;
	merge(new_channels, test_hr);

	cvtColor(test_hr, test_hr, CV_YCrCb2BGR);

	return test_hr;
}

vector<String> getImageName(String path) {
	vector<String> imageNames;
	glob(path, imageNames, false);
	int size = imageNames.size();
	for (int i = 0; i < size; i++) {
		string name = imageNames[i];
		name = name.substr(name.find_last_of("\\") + 1);
		imageNames[i] = name;
	}
	return imageNames;
}

String path_sample_lr;
String path_sample_hr;
String path_test_lr;
String path_test_hr;
int upscale;

void readParameter(const char* filename)
{
	std::stringstream buffer;
	std::string line;
	std::string paramName;
	std::string paramValuestr;

	std::ifstream fin(filename);
	if (!fin.good())
	{
		std::string msg("parameters file not found");
		msg.append(filename);
		throw std::runtime_error(msg);
	}
	while (fin.good())
	{
		getline(fin, line);
		if (line[0] != '#')
		{
			buffer.clear();//before multiple convertion ,clear() is necessary !
			buffer << line;
			buffer >> paramName;
			if (paramName.compare("path_sample_lr") == 0)
			{
				buffer >> paramValuestr;
				path_sample_lr = paramValuestr;
			}
			else if (paramName.compare("path_sample_hr") == 0)
			{
				buffer >> paramValuestr;
				path_sample_hr = paramValuestr;
			}
			else if (paramName.compare("path_test_lr") == 0)
			{
				buffer >> paramValuestr;
				path_test_lr = paramValuestr;
			}
			else if (paramName.compare("path_test_hr") == 0)
			{
				buffer >> paramValuestr;
				path_test_hr = paramValuestr;
			}
			else if (paramName.compare("upscale") == 0)
			{
				buffer >> paramValuestr;
				upscale = stoi(paramValuestr);
			}
		}

	}

	fin.close();

}


int main() {

	path_sample_lr = "sample_lr/*";
	path_sample_hr = "sample_hr/*";
	path_test_lr = "test_lr/*";
	path_test_hr = "test_pca/";
	//upscale = 4;
	
	vector<Mat> imTestLR = imageLoad2(path_test_lr);
	vector<String> imNames = getImageName(path_test_lr);
	int numOfImage = getImageName(path_sample_hr).size();
	int test_size = imTestLR.size();
	double alpha = 0.1;

	vector<Mat> imSampleLR = imageLoad(path_sample_lr, numOfImage);
	vector<Mat> imSampleHR = imageLoad(path_sample_hr, numOfImage);
	upscale = imSampleHR[0].rows / imSampleLR[0].rows;

	int numOfRowsLR = imSampleLR[0].rows;
	int numOfColsLR = imSampleLR[0].cols;
	int numOfRowsHR = imSampleHR[0].rows;
	int numOfColsHR = imSampleHR[0].cols;

	Mat allImageLR = imagesResize(imSampleLR, numOfImage);
	Mat allImageHR = imagesResize(imSampleHR, numOfImage);

	Mat MeanFaceLR = getMeanFace(imSampleLR, numOfImage);
	Mat MeanFaceHR = getMeanFace(imSampleHR, numOfImage);

	Mat demeanImageLR = getDemeanFaces(allImageLR, MeanFaceLR, numOfImage, numOfRowsLR, numOfColsLR);
	Mat demeanImageHR = getDemeanFaces(allImageHR, MeanFaceHR, numOfImage, numOfRowsHR, numOfColsHR);

	Mat CorrLR = demeanImageLR.t() * demeanImageLR;
	Mat CorrHR = demeanImageHR.t() * demeanImageHR;

	Mat EigenVecLR = Mat1d(numOfImage, numOfImage, 0.0);
	Mat EigenValLR = Mat1d(numOfImage, 1, 0.0);


	Mat EigenVecHR = Mat1d(numOfImage, numOfImage, 0.0);
	Mat EigenValHR = Mat1d(numOfImage, 1, 0.0);

	double *CorrLR1 = (double*)CorrLR.data;
	double *CorrHR1 = (double*)CorrHR.data;

	double *EigenVecLR1 = (double*)EigenVecLR.data;
	double *EigenValLR1 = (double*)EigenValLR.data;

	double *EigenVecHR1 = (double*)EigenVecHR.data;
	double *EigenValHR1 = (double*)EigenValHR.data;

	CEigen eig1;

	eig1.cvJacobiEigens_64d(CorrLR1, EigenVecLR1, EigenValLR1, numOfImage, 0);
	eig1.cvJacobiEigens_64d(CorrHR1, EigenVecHR1, EigenValHR1, numOfImage, 0);

	pow(EigenValLR, -0.5, EigenValLR);
	pow(EigenValHR, -0.5, EigenValHR);

	EigenVecLR = EigenVecLR.t();
	EigenVecHR = EigenVecHR.t();


	Mat EigenValLRN = Mat1d(numOfImage, numOfImage, 0.0);

	for (int i = 0; i < numOfImage; i++) {
		EigenValLRN.at<double>(i, i) = EigenValLR.at<double>(i);
	}

	Mat EigenValHRN = Mat1d(numOfImage, numOfImage, 0.0);

	for (int i = 0; i < numOfImage; i++) {
		EigenValHRN.at<double>(i, i) = EigenValHR.at<double>(i);
	}

	for (int i = 0; i < test_size; i++) {
		Mat imTest = imTestLR[i];
		Mat ImResult = getHR(imTest, upscale, MeanFaceLR, MeanFaceHR, numOfImage, EigenVecLR, EigenValLRN, demeanImageLR, demeanImageHR, numOfRowsHR, numOfColsHR, alpha);

		string file = path_test_hr + imNames[i];
		cout << file << endl;
		imwrite(file, ImResult);
	
	}
	return 0;
}

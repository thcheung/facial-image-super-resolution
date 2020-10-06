#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "Eigen.h"
#include "myFusedLasso.h"

using namespace cv;
using namespace std;


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


	Mat coef = EigenVal * temp;


	for (int i = 0; i < numofImage; i++) {
		if ((abs(coef.at<double>(i))) > (alpha * 1.0 / EigenValLRN.at<double>(i, i))) {
			if (coef.at<double>(i) > 0.0) {
				coef.at<double>(i) = alpha * 1.0 / EigenValLRN.at<double>(i, i);
				//cout << "1";
			}
			else {
				coef.at<double>(i) = -alpha * 1.0 / EigenValLRN.at<double>(i, i);
				//cout << "2";
			}
		};
	}

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

double calDistance(Mat v1, Mat v2) {

	return norm(v1 - v2, NORM_L1);
}

vector<int> knn(Mat vec, vector<Mat> vecs, int k) {

	int numOfVec = vecs.size();
	vector<double> dists(numOfVec);
	vector<int> index(k);

	for (int i = 0; i < numOfVec; i++) {

		dists[i] = calDistance(vec, vecs[i]);

	}

	for (int i = 0; i < k; i++) {
		int current_index = 0;
		for (int j = 0; j < numOfVec; j++)
		{
			if (dists[j] < dists[current_index])
				current_index = j;
		}
		dists[current_index] = INFINITY;
		index[i] = current_index;
	}

	return index;
}

Mat patch2Image(vector<Mat> patches, int numOfPatch_i, int numOfPatch_j, int sizeOfPatch, int sizeOfOverlapping, int numOfRowsHR, int numOfColsHR) {

	int step_size = sizeOfPatch - sizeOfOverlapping;
	Mat Result = Mat1d(numOfRowsHR, numOfColsHR, 0.0);
	Mat Count = Mat1d(numOfRowsHR, numOfColsHR, 0.0);
	for (int i = 0; i < numOfPatch_i; i++) {
		for (int j = 0; j < numOfPatch_j; j++) {
			int index = j * numOfPatch_i + i;
			for (int m = 0; m < sizeOfPatch; m++) {
				for (int n = 0; n < sizeOfPatch; n++) {
					Result.at<double>(i*step_size + m, j*step_size + n) = Result.at<double>(i*step_size + m, j*step_size + n) + patches[index].at<double>(m, n);
					Count.at<double>(i*step_size + m, j*step_size + n) = Count.at<double>(i*step_size + m, j*step_size + n) + 1;

				}
			}

		}
	}
	divide(Result, Count, Result);
	return Result;
}

Mat mySobX(Mat imgIn) {
	int numOfRows = imgIn.rows;
	int numOfColumns = imgIn.cols;
	Mat imgOut = Mat1d(numOfRows, numOfColumns);

	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfColumns; j++) {
			imgOut.at<double>(i, j) = 0;
		}
	}

	for (int i = 1; i < numOfRows - 1; i++) {
		for (int j = 1; j < numOfColumns - 1; j++) {
			imgOut.at<double>(i, j) = -1 * imgIn.at<double>(i - 1, j - 1) - 2 * imgIn.at<double>(i, j - 1)
				- 1 * imgIn.at<double>(i + 1, j - 1) + 1 * imgIn.at<double>(i - 1, j + 1)
				+ 2 * imgIn.at<double>(i, j + 1) + 1 * imgIn.at<double>(i + 1, j + 1);
		}
	}

	return imgOut;
}

Mat mySobY(Mat imgIn) {
	int numOfRows = imgIn.rows;
	int numOfColumns = imgIn.cols;
	Mat imgOut = Mat1d(numOfRows, numOfColumns);

	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfColumns; j++) {
			imgOut.at<double>(i, j) = 0;
		}
	}

	for (int i = 1; i < numOfRows - 1; i++) {
		for (int j = 1; j < numOfColumns - 1; j++) {
			imgOut.at<double>(i, j) = 1 * imgIn.at<double>(i - 1, j - 1) + 2 * imgIn.at<double>(i - 1, j)
				+ 1 * imgIn.at<double>(i - 1, j + 1) - 1 * imgIn.at<double>(i + 1, j - 1)
				- 2 * imgIn.at<double>(i + 1, j) + -1 * imgIn.at<double>(i + 1, j + 1);
		}
	}
	return imgOut;
}

vector<Mat> getPatches(Mat imageIn, int sizeOfPatch, int sizeOfOverlapping) {
	int numOfRows = imageIn.rows;
	int numOfCols = imageIn.cols;
	int numOfpatch_j = floor((numOfRows - sizeOfOverlapping) / (sizeOfPatch - sizeOfOverlapping));
	int numOfpatch_i = floor((numOfCols - sizeOfOverlapping) / (sizeOfPatch - sizeOfOverlapping));
	vector<Mat> allPatch(numOfpatch_i*numOfpatch_j);
	for (int i = 0; i < numOfpatch_i; i++) {
		for (int j = 0; j < numOfpatch_j; j++) {
			int m = i * (sizeOfPatch - sizeOfOverlapping);
			int n = j * (sizeOfPatch - sizeOfOverlapping);
			Mat imagePatch = imageIn(Rect(m, n, sizeOfPatch, sizeOfPatch));
			int index = i * numOfpatch_j + j;
			allPatch[index] = imagePatch;
		}
	}
	return allPatch;

}

Mat getMeanFace_2(Mat image) {
	int numOfRows = image.rows;
	int numOfCols = image.cols;
	double sum = 0;
	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfCols; j++) {
			sum = sum + image.at <double>(i, j);
		}
	}
	sum = sum / (numOfRows*numOfCols);
	Mat output = Mat1d(numOfRows, numOfCols, sum);
	return output;
}

Mat getDemeanFace(Mat image) {
	Mat MeanFace = getMeanFace_2(image);
	return image - MeanFace;
}

Mat imageResize(Mat image) {
	int numOfRows = image.rows;
	int numOfCols = image.cols;
	Mat resizedImage = Mat1d(numOfRows*numOfCols, 1, 0.0);
	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfCols; j++) {
			resizedImage.at <double>(i*numOfCols + j) = image.at <double>(i, j);
		}
	}
	return resizedImage;
}

Mat getFeature(Mat feature1, Mat feature2, Mat feature3, Mat feature4) {
	vector<Mat> features;
	features.push_back(feature1);
	features.push_back(feature2);
	features.push_back(feature3);
	features.push_back(feature4);
	Mat featureM = imagesResize(features, 4);
	return imageResize(featureM);
}

Mat getHR_lle(int upscale, Mat imTest, int sizeOfPatch_lr, int sizeOfOverlapping_lr, vector<Mat> features, int k, vector<Mat> patches_lr, int sizeOfPatch_hr, vector<Mat> patches_hr, int numOfpatch_j, int numOfpatch_i, int sizeOfOverlapping_hr, int numOfRowsHR, int numOfColsHR, String test_sr) {
	Mat imTest_ycrcb;
	cvtColor(imTest, imTest_ycrcb, CV_BGR2YCrCb);
	Mat channels[3];
	split(imTest_ycrcb, channels);

	Mat imTest_y = channels[0];
	Mat imTest_cr = channels[1];
	Mat imTest_cb = channels[2];


	imTest_y.convertTo(imTest_y, CV_64F, 1.f / 255);
	vector <Mat> patch_test = getPatches(imTest_y, sizeOfPatch_lr, sizeOfOverlapping_lr);

	Mat graX_test = mySobX(imTest_y);
	Mat graY_test = mySobY(imTest_y);
	vector<Mat>feature1InImage_lr = getPatches(graX_test, sizeOfPatch_lr, sizeOfOverlapping_lr);
	vector<Mat>feature2InImage_lr = getPatches(graY_test, sizeOfPatch_lr, sizeOfOverlapping_lr);
	vector<Mat>feature3InImage_lr = getPatches(mySobX(graX_test), sizeOfPatch_lr, sizeOfOverlapping_lr);
	vector<Mat>feature4InImage_lr = getPatches(mySobY(graY_test), sizeOfPatch_lr, sizeOfOverlapping_lr);
	vector <Mat> test_hr;


	int numOfPatches = patch_test.size();

	for (int i = 0; i < numOfPatches; i++) {
		Mat feature = getFeature(feature1InImage_lr[i], feature2InImage_lr[i], feature3InImage_lr[i], feature4InImage_lr[i]);

		vector<int> index = knn(feature, features, k);

		Mat ones = Mat1d(1, k, 1);

		Mat X = Mat1d((int)sizeOfPatch_lr*sizeOfPatch_lr, k, (double)0);

		for (int j = 0; j < k; j++) {

			Mat currentPatch = imageResize(patches_lr[index[j]]);
			int size = currentPatch.rows;
			for (int m = 0; m < size; m++) {
				X.at <double>(m, j) = currentPatch.at<double>(m);
			}
		}

		Mat K = imageResize(getDemeanFace(patch_test[i])) * ones - X;
		Mat G = K.t()*K;
		ones = ones.t();
		Mat W = G.inv()*ones / (ones.t()*G.inv()*ones);
		int numOfVec = W.rows;

		double sum = 0;
		for (int j = 0; j < numOfVec; j++) {

			sum = sum + W.at<double>(j);
		}

		Mat WN = W / sum;
		Mat patch_hr = Mat1d(sizeOfPatch_hr, sizeOfPatch_hr, 0.0);

		for (int j = 0; j < k; j++) {
			patch_hr = WN.at<double>(j) * patches_hr[index[j]] + patch_hr;
		}
		Mat meanFaceHR;

		resize(getMeanFace_2(patch_test[i]), meanFaceHR, Size(), upscale, upscale, INTER_NEAREST);
		patch_hr = patch_hr + meanFaceHR;
		test_hr.push_back(patch_hr);

	}

	Mat Imresult = patch2Image(test_hr, numOfpatch_j, numOfpatch_i, sizeOfPatch_hr, sizeOfOverlapping_hr, numOfRowsHR, numOfColsHR);

	Imresult.convertTo(imTest_y, CV_8U, 255);

	resize(imTest_cr, imTest_cr, Size(), upscale, upscale, INTER_CUBIC);
	resize(imTest_cb, imTest_cb, Size(), upscale, upscale, INTER_CUBIC);

	vector <Mat> new_channels;
	new_channels.push_back(imTest_y);
	new_channels.push_back(imTest_cr);
	new_channels.push_back(imTest_cb);
	Mat test_hr_3;
	merge(new_channels, test_hr_3);

	cvtColor(test_hr_3, test_hr_3, CV_YCrCb2BGR);

	return test_hr_3;
}

vector<int> sortVec(Mat vec, vector<Mat> vecs) {
	int numOfVec = vecs.size();
	vector<int> index(numOfVec);

	vector<double> dists(numOfVec);
	for (int i = 0; i < numOfVec; i++) {
		dists[i] = calDistance(vec, vecs[i]);
	}


	vector<double> sortedDists = dists;
	sort(sortedDists.begin(), sortedDists.end());

	for (int i = 0; i < numOfVec; i++) {
		for (int j = 0; j < numOfVec; j++) {
			if (sortedDists[i] == dists[j]) {
				index[i] = j;
				dists[j] = INFINITY;
				break;
			}
		}
	}


	return index;
}

Mat findWeight(Mat matrix, Mat vec, int numOfVecs, double a, double b) {
	int numOfPixels = matrix.rows;
	int numOfSamples = matrix.cols;
	Mat lamda = Mat1d(1, 1, a);
	Mat lamda2 = Mat1d(1, 1, b);
	Mat W = Mat1d(numOfVecs, 1, 0.0);
	mxArray *matrix_mx;
	matrix_mx = mxCreateDoubleMatrix(numOfPixels, numOfSamples, mxREAL);

	double* buffer = (double*)mxGetPr(matrix_mx);

	for (int i = 0; i < numOfPixels; i++) {
		for (int j = 0; j < numOfSamples; j++) {
			buffer[j*numOfPixels + i] = matrix.at<double>(i, j);
		}
	}

	mxArray *vec_mx;
	vec_mx = mxCreateDoubleMatrix(numOfPixels, 1, mxREAL);
	double* buffer2 = (double*)mxGetPr(vec_mx);
	for (int i = 0; i < numOfPixels; i++) {
		buffer2[i] = vec.at<double>(i);
	}


	mxArray *a_mx;
	a_mx = mxCreateDoubleScalar(a);

	mxArray *b_mx;
	b_mx = mxCreateDoubleScalar(b);

	mxArray *output = mxCreateDoubleMatrix(numOfSamples, 1, mxREAL);
	mlfMyFusedLeastR(1, &output, matrix_mx, vec_mx, a_mx, b_mx);
	double* temp = (double*)mxGetPr(output);
	for (int i = 0; i < numOfSamples; i++) {
		W.at<double>(i) = temp[i];
	}

	mxDestroyArray(a_mx);
	mxDestroyArray(vec_mx);
	mxDestroyArray(matrix_mx);
	mxDestroyArray(output);
	return W;

}

Mat getHR_ssr(int upscale, Mat imTest, int sizeOfPatch_lr, int sizeOfOverlapping_lr,vector<Mat> imSampleLR, vector<Mat> imSampleHR, int numOfImage, int sizeOfPatch_hr, int numOfpatch_j, int numOfpatch_i, int sizeOfOverlapping_hr, int numOfRowsHR, int numOfColsHR, double a, double b) {
	Mat imTest_ycrcb;
	cvtColor(imTest, imTest_ycrcb, CV_BGR2YCrCb);
	Mat channels[3];
	split(imTest_ycrcb, channels);

	Mat imTest_y = channels[0];
	Mat imTest_cr = channels[1];
	Mat imTest_cb = channels[2];


	imTest_y.convertTo(imTest_y, CV_64F, 1.f / 255);



	vector<Mat> patch_test = getPatches(imTest_y, sizeOfPatch_lr, sizeOfOverlapping_lr);


	int numOfPatches = patch_test.size();

	vector<Mat> test_hr;

	for (int i = 0; i < numOfPatches; i++) {

		Mat testVec = imageResize(getDemeanFace(patch_test[i]));

		vector<Mat> patches_lr, patches_hr;

		for (int j = 0; j < numOfImage; j++) {
			vector<Mat>patchesInImage_lr = getPatches(imSampleLR[j], sizeOfPatch_lr, sizeOfOverlapping_lr);
			vector<Mat>patchesInImage_hr = getPatches(imSampleHR[j], sizeOfPatch_hr, sizeOfOverlapping_hr);
			patches_lr.push_back(patchesInImage_lr[i]);
			patches_hr.push_back(patchesInImage_hr[i]);
		}

		vector<Mat> vectors_lr;


		int numOfVecs = patches_lr.size();
		for (int j = 0; j < numOfVecs; j++) {
			vectors_lr.push_back(imageResize(patches_lr[j]));
		}

		vector<Mat> unSortedVecLr(numOfImage), unSortedVecHr(numOfImage);
		for (int j = 0; j < numOfImage; j++) {
			unSortedVecLr[j] = getDemeanFace(patches_lr[j]);
			unSortedVecHr[j] = getDemeanFace(patches_hr[j]);
		}

		vector<int> index = sortVec(testVec, vectors_lr);

		vector<Mat> SortedVecLr(numOfImage), SortedVecHr(numOfImage);
		for (int j = 0; j < numOfImage; j++) {

			SortedVecLr[j] = unSortedVecLr[index[j]];
			SortedVecHr[j] = unSortedVecHr[index[j]];

		}
		Mat matrix = imagesResize(unSortedVecLr, numOfImage);
		Mat patch_hr = Mat1d(sizeOfPatch_hr, sizeOfPatch_hr, 0.0);
		Mat W = findWeight(matrix, testVec, numOfImage, a, b);
		for (int j = 0; j < numOfImage; j++) {
			patch_hr = W.at<double>(j) * unSortedVecHr[j] + patch_hr;
		}
		Mat meanFaceHR;
		resize(getMeanFace_2(patch_test[i]), meanFaceHR, Size(), upscale, upscale, INTER_NEAREST);
		patch_hr = patch_hr + meanFaceHR;
		test_hr.push_back(patch_hr);

	}

	Mat Imresult = patch2Image(test_hr, numOfpatch_j, numOfpatch_i, sizeOfPatch_hr, sizeOfOverlapping_hr, numOfRowsHR, numOfColsHR);
	Imresult.convertTo(imTest_y, CV_8U, 255);

	resize(imTest_cr, imTest_cr, Size(), upscale, upscale, INTER_CUBIC);
	resize(imTest_cb, imTest_cb, Size(), upscale, upscale, INTER_CUBIC);

	vector <Mat> new_channels;
	new_channels.push_back(imTest_y);
	new_channels.push_back(imTest_cr);
	new_channels.push_back(imTest_cb);
	Mat test_hr_3;
	merge(new_channels, test_hr_3);

	cvtColor(test_hr_3, test_hr_3, CV_YCrCb2BGR);

	return test_hr_3;
}

int upscale;
int dataset;
String method;
String test_lr;
String test_sr;

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


			if (paramName.compare("method") == 0)
			{
				buffer >> paramValuestr;
				method = paramValuestr;
			}
			else if (paramName.compare("test_sr") == 0)
			{
				buffer >> paramValuestr;
				test_sr = paramValuestr;
			}
			else if (paramName.compare("test_lr") == 0)
			{
				buffer >> paramValuestr;
				test_lr = paramValuestr;
			}
			else if (paramName.compare("upscale") == 0)
			{
				buffer >> paramValuestr;
				upscale = stoi(paramValuestr);
			}
			else if (paramName.compare("dataset") == 0)
			{
				buffer >> paramValuestr;
				dataset = stoi(paramValuestr);
			}
		}

	}

	fin.close();

}


int main() {
	//cout << "Programe starts.\n";
	cout << 0 << endl;
	std::cout.flush();
	String path_lr_4x_1 = "datasets/dataset1/sample_lr_4x/*.jpg";
	String path_lr_8x_1 = "datasets/dataset1/sample_lr_8x/*.jpg";
	String path_hr_1 = "datasets/dataset1/sample_hr/*.jpg";
	int numOfImage_1 = 100;

	String path_lr_4x_2 = "datasets/dataset2/sample_lr_4x/*.jpg";
	String path_lr_8x_2 = "datasets/dataset2/sample_lr_8x/*.jpg";
	String path_hr_2 = "datasets/dataset2/sample_hr/*.jpg";
	int numOfImage_2 = 500;

	vector<Mat> imSampleLR_4x_1 = imageLoad(path_lr_4x_1, numOfImage_1);
	vector<Mat> imSampleLR_8x_1 = imageLoad(path_lr_8x_1, numOfImage_1);
	vector<Mat> imSampleHR_1 = imageLoad(path_hr_1, numOfImage_1);

	vector<Mat> imSampleLR_4x_2 = imageLoad(path_lr_4x_2, numOfImage_2);
	vector<Mat> imSampleLR_8x_2 = imageLoad(path_lr_8x_2, numOfImage_2);
	vector<Mat> imSampleHR_2 = imageLoad(path_hr_2, numOfImage_2);


	int numOfRowsLR_4x_1 = imSampleLR_4x_1[0].rows;
	int numOfColsLR_4x_1 = imSampleLR_4x_1[0].cols;
	int numOfRowsLR_8x_1 = imSampleLR_8x_1[0].rows;
	int numOfColsLR_8x_1 = imSampleLR_8x_1[0].cols;
	int numOfRowsHR_1 = imSampleHR_1[0].rows;
	int numOfColsHR_1 = imSampleHR_1[0].cols;

	//cout << "Training samples are loaded.\n";

	/*
	PCA is preloaded below.
	*/
	double alpha = 0.8;


	Mat allImageLR_4x_1 = imagesResize(imSampleLR_4x_1, numOfImage_1);
	Mat allImageLR_8x_1 = imagesResize(imSampleLR_8x_1, numOfImage_1);
	Mat allImageHR_1 = imagesResize(imSampleHR_1, numOfImage_1);

	Mat MeanFaceLR_4x_1 = getMeanFace(imSampleLR_4x_1, numOfImage_1);
	Mat MeanFaceLR_8x_1 = getMeanFace(imSampleLR_8x_1, numOfImage_1);
	Mat MeanFaceHR_1 = getMeanFace(imSampleHR_1, numOfImage_1);

	Mat demeanImageLR_4x_1 = getDemeanFaces(allImageLR_4x_1, MeanFaceLR_4x_1, numOfImage_1, numOfRowsLR_4x_1, numOfColsLR_4x_1);
	Mat demeanImageLR_8x_1 = getDemeanFaces(allImageLR_8x_1, MeanFaceLR_8x_1, numOfImage_1, numOfRowsLR_8x_1, numOfColsLR_8x_1);
	Mat demeanImageHR_1 = getDemeanFaces(allImageHR_1, MeanFaceHR_1, numOfImage_1, numOfRowsHR_1, numOfColsHR_1);

	Mat CorrLR_4x_1 = demeanImageLR_4x_1.t() * demeanImageLR_4x_1;
	Mat CorrLR_8x_1 = demeanImageLR_8x_1.t() * demeanImageLR_8x_1;
	Mat CorrHR_1 = demeanImageHR_1.t() * demeanImageHR_1;

	Mat EigenVecLR_4x_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);
	Mat EigenVecLR_8x_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);

	Mat EigenValLR_4x_1 = Mat1d(numOfImage_1, 1, 0.0);
	Mat EigenValLR_8x_1 = Mat1d(numOfImage_1, 1, 0.0);

	Mat EigenVecHR_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);
	Mat EigenValHR_1 = Mat1d(numOfImage_1, 1, 0.0);

	double *CorrLR1_4x_1 = (double*)CorrLR_4x_1.data;
	double *CorrLR1_8x_1 = (double*)CorrLR_8x_1.data;
	double *CorrHR1_1 = (double*)CorrHR_1.data;

	double *EigenVecLR1_4x_1 = (double*)EigenVecLR_4x_1.data;
	double *EigenValLR1_4x_1 = (double*)EigenValLR_4x_1.data;
	double *EigenVecLR1_8x_1 = (double*)EigenVecLR_8x_1.data;
	double *EigenValLR1_8x_1 = (double*)EigenValLR_8x_1.data;

	double *EigenVecHR1_1 = (double*)EigenVecHR_1.data;
	double *EigenValHR1_1 = (double*)EigenValHR_1.data;

	CEigen eig1_1;

	eig1_1.cvJacobiEigens_64d(CorrLR1_4x_1, EigenVecLR1_4x_1, EigenValLR1_4x_1, numOfImage_1, 0);
	eig1_1.cvJacobiEigens_64d(CorrLR1_8x_1, EigenVecLR1_8x_1, EigenValLR1_8x_1, numOfImage_1, 0);
	eig1_1.cvJacobiEigens_64d(CorrHR1_1, EigenVecHR1_1, EigenValHR1_1, numOfImage_1, 0);

	pow(EigenValLR_4x_1, -0.5, EigenValLR_4x_1);
	pow(EigenValLR_8x_1, -0.5, EigenValLR_8x_1);
	pow(EigenValHR_1, -0.5, EigenValHR_1);

	EigenVecLR_4x_1 = EigenVecLR_4x_1.t();
	EigenVecLR_8x_1 = EigenVecLR_8x_1.t();
	EigenVecHR_1 = EigenVecHR_1.t();


	Mat EigenValLRN_4x_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);
	Mat EigenValLRN_8x_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);

	for (int i = 0; i < numOfImage_1; i++) {
		EigenValLRN_4x_1.at<double>(i, i) = EigenValLR_4x_1.at<double>(i);
		EigenValLRN_8x_1.at<double>(i, i) = EigenValLR_8x_1.at<double>(i);
	}

	Mat EigenValHRN_1 = Mat1d(numOfImage_1, numOfImage_1, 0.0);

	for (int i = 0; i < numOfImage_1; i++) {
		EigenValHRN_1.at<double>(i, i) = EigenValHR_1.at<double>(i);
	}

	int numOfRowsLR_4x_2 = imSampleLR_4x_2[0].rows;
	int numOfColsLR_4x_2 = imSampleLR_4x_2[0].cols;
	int numOfRowsLR_8x_2 = imSampleLR_8x_2[0].rows;
	int numOfColsLR_8x_2 = imSampleLR_8x_2[0].cols;
	int numOfRowsHR_2 = imSampleHR_2[0].rows;
	int numOfColsHR_2 = imSampleHR_2[0].cols;

	Mat allImageLR_4x_2 = imagesResize(imSampleLR_4x_2, numOfImage_2);
	Mat allImageLR_8x_2 = imagesResize(imSampleLR_8x_2, numOfImage_2);
	Mat allImageHR_2 = imagesResize(imSampleHR_2, numOfImage_2);

	Mat MeanFaceLR_4x_2 = getMeanFace(imSampleLR_4x_2, numOfImage_2);
	Mat MeanFaceLR_8x_2 = getMeanFace(imSampleLR_8x_2, numOfImage_2);
	Mat MeanFaceHR_2 = getMeanFace(imSampleHR_2, numOfImage_2);

	Mat demeanImageLR_4x_2 = getDemeanFaces(allImageLR_4x_2, MeanFaceLR_4x_2, numOfImage_2, numOfRowsLR_4x_2, numOfColsLR_4x_2);
	Mat demeanImageLR_8x_2 = getDemeanFaces(allImageLR_8x_2, MeanFaceLR_8x_2, numOfImage_2, numOfRowsLR_8x_2, numOfColsLR_8x_2);
	Mat demeanImageHR_2 = getDemeanFaces(allImageHR_2, MeanFaceHR_2, numOfImage_2, numOfRowsHR_2, numOfColsHR_2);

	Mat CorrLR_4x_2 = demeanImageLR_4x_2.t() * demeanImageLR_4x_2;
	Mat CorrLR_8x_2 = demeanImageLR_8x_2.t() * demeanImageLR_8x_2;
	Mat CorrHR_2 = demeanImageHR_2.t() * demeanImageHR_2;

	Mat EigenVecLR_4x_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);
	Mat EigenVecLR_8x_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);

	Mat EigenValLR_4x_2 = Mat1d(numOfImage_2, 1, 0.0);
	Mat EigenValLR_8x_2 = Mat1d(numOfImage_2, 1, 0.0);

	Mat EigenVecHR_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);
	Mat EigenValHR_2 = Mat1d(numOfImage_2, 1, 0.0);

	double *CorrLR1_4x_2 = (double*)CorrLR_4x_2.data;
	double *CorrLR1_8x_2 = (double*)CorrLR_8x_2.data;
	double *CorrHR1_2 = (double*)CorrHR_2.data;

	double *EigenVecLR1_4x_2 = (double*)EigenVecLR_4x_2.data;
	double *EigenValLR1_4x_2 = (double*)EigenValLR_4x_2.data;
	double *EigenVecLR1_8x_2 = (double*)EigenVecLR_8x_2.data;
	double *EigenValLR1_8x_2 = (double*)EigenValLR_8x_2.data;

	double *EigenVecHR1_2 = (double*)EigenVecHR_2.data;
	double *EigenValHR1_2 = (double*)EigenValHR_2.data;

	CEigen eig1_2;

	eig1_2.cvJacobiEigens_64d(CorrLR1_4x_2, EigenVecLR1_4x_2, EigenValLR1_4x_2, numOfImage_2, 0);
	eig1_2.cvJacobiEigens_64d(CorrLR1_8x_2, EigenVecLR1_8x_2, EigenValLR1_8x_2, numOfImage_2, 0);
	eig1_2.cvJacobiEigens_64d(CorrHR1_2, EigenVecHR1_2, EigenValHR1_2, numOfImage_2, 0);

	pow(EigenValLR_4x_2, -0.5, EigenValLR_4x_2);
	pow(EigenValLR_8x_2, -0.5, EigenValLR_8x_2);
	pow(EigenValHR_2, -0.5, EigenValHR_2);

	EigenVecLR_4x_2 = EigenVecLR_4x_2.t();
	EigenVecLR_8x_2 = EigenVecLR_8x_2.t();
	EigenVecHR_2 = EigenVecHR_2.t();


	Mat EigenValLRN_4x_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);
	Mat EigenValLRN_8x_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);

	for (int i = 0; i < numOfImage_2; i++) {
		EigenValLRN_4x_2.at<double>(i, i) = EigenValLR_4x_2.at<double>(i);
		EigenValLRN_8x_2.at<double>(i, i) = EigenValLR_8x_2.at<double>(i);
	}

	Mat EigenValHRN_2 = Mat1d(numOfImage_2, numOfImage_2, 0.0);

	for (int i = 0; i < numOfImage_2; i++) {
		EigenValHRN_2.at<double>(i, i) = EigenValHR_2.at<double>(i);
	}

	//cout << "Training samples for PCA are ready.\n";
	/*
	LLE is preloaded below.
	*/


	int k = 3;

	int sizeOfPatch_lr_8x = 4;
	int sizeOfOverlapping_lr_8x = 2;
	int sizeOfPatch_lr_4x = 8;
	int sizeOfOverlapping_lr_4x = 4;

	int sizeOfPatch_hr_4x = sizeOfPatch_lr_4x * 4;
	int sizeOfOverlapping_hr_4x = sizeOfOverlapping_lr_4x * 4;
	int sizeOfPatch_hr_8x = sizeOfPatch_lr_8x * 8;
	int sizeOfOverlapping_hr_8x = sizeOfOverlapping_lr_8x * 8;


	int numOfpatch_j_4x_1 = floor((numOfRowsLR_4x_1 - sizeOfOverlapping_lr_4x) / (sizeOfPatch_lr_4x - sizeOfOverlapping_lr_4x));
	int numOfpatch_i_4x_1 = floor((numOfColsLR_4x_1 - sizeOfOverlapping_lr_4x) / (sizeOfPatch_lr_4x - sizeOfOverlapping_lr_4x));
	int numOfpatch_j_8x_1 = floor((numOfRowsLR_8x_1 - sizeOfOverlapping_lr_8x) / (sizeOfPatch_lr_8x - sizeOfOverlapping_lr_8x));
	int numOfpatch_i_8x_1 = floor((numOfColsLR_8x_1 - sizeOfOverlapping_lr_8x) / (sizeOfPatch_lr_8x - sizeOfOverlapping_lr_8x));

	vector<Mat> patches_lr_4x_1, patches_hr_4x_1, features_4x_1;

	for (int i = 0; i < numOfImage_1; i++) {
		vector<Mat>patchesInImage_lr = getPatches(imSampleLR_4x_1[i], sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>patchesInImage_hr = getPatches(imSampleHR_1[i], sizeOfPatch_hr_4x, sizeOfOverlapping_hr_4x);
		Mat graX = mySobX(imSampleLR_4x_1[i]);
		Mat graY = mySobY(imSampleLR_4x_1[i]);
		vector<Mat>feature1InImage_lr = getPatches(graX, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature2InImage_lr = getPatches(graY, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature3InImage_lr = getPatches(mySobX(graX), sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature4InImage_lr = getPatches(mySobY(graY), sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		int numOfPatch = patchesInImage_lr.size();
		for (int j = 0; j < numOfPatch; j++) {
			patches_lr_4x_1.push_back(getDemeanFace(patchesInImage_lr[j]));
			patches_hr_4x_1.push_back(getDemeanFace(patchesInImage_hr[j]));
			features_4x_1.push_back(getFeature(feature1InImage_lr[j], feature2InImage_lr[j], feature3InImage_lr[j], feature4InImage_lr[j]));
		}
	}

	vector<Mat> patches_lr_8x_1, patches_hr_8x_1, features_8x_1;

	for (int i = 0; i < numOfImage_1; i++) {
		vector<Mat>patchesInImage_lr = getPatches(imSampleLR_8x_1[i], sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>patchesInImage_hr = getPatches(imSampleHR_1[i], sizeOfPatch_hr_8x, sizeOfOverlapping_hr_8x);
		Mat graX = mySobX(imSampleLR_8x_1[i]);
		Mat graY = mySobY(imSampleLR_8x_1[i]);
		vector<Mat>feature1InImage_lr = getPatches(graX, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature2InImage_lr = getPatches(graY, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature3InImage_lr = getPatches(mySobX(graX), sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature4InImage_lr = getPatches(mySobY(graY), sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		int numOfPatch = patchesInImage_lr.size();
		for (int j = 0; j < numOfPatch; j++) {
			patches_lr_8x_1.push_back(getDemeanFace(patchesInImage_lr[j]));
			patches_hr_8x_1.push_back(getDemeanFace(patchesInImage_hr[j]));
			features_8x_1.push_back(getFeature(feature1InImage_lr[j], feature2InImage_lr[j], feature3InImage_lr[j], feature4InImage_lr[j]));
		}
	}

	int numOfpatch_j_4x_2 = floor((numOfRowsLR_4x_2 - sizeOfOverlapping_lr_4x) / (sizeOfPatch_lr_4x - sizeOfOverlapping_lr_4x));
	int numOfpatch_i_4x_2 = floor((numOfColsLR_4x_2 - sizeOfOverlapping_lr_4x) / (sizeOfPatch_lr_4x - sizeOfOverlapping_lr_4x));
	int numOfpatch_j_8x_2 = floor((numOfRowsLR_8x_2 - sizeOfOverlapping_lr_8x) / (sizeOfPatch_lr_8x - sizeOfOverlapping_lr_8x));
	int numOfpatch_i_8x_2 = floor((numOfColsLR_8x_2 - sizeOfOverlapping_lr_8x) / (sizeOfPatch_lr_8x - sizeOfOverlapping_lr_8x));

	vector<Mat> patches_lr_4x_2, patches_hr_4x_2, features_4x_2;

	for (int i = 0; i < numOfImage_2; i++) {
		vector<Mat>patchesInImage_lr = getPatches(imSampleLR_4x_2[i], sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>patchesInImage_hr = getPatches(imSampleHR_2[i], sizeOfPatch_hr_4x, sizeOfOverlapping_hr_4x);
		Mat graX = mySobX(imSampleLR_4x_2[i]);
		Mat graY = mySobY(imSampleLR_4x_2[i]);
		vector<Mat>feature1InImage_lr = getPatches(graX, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature2InImage_lr = getPatches(graY, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature3InImage_lr = getPatches(mySobX(graX), sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		vector<Mat>feature4InImage_lr = getPatches(mySobY(graY), sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x);
		int numOfPatch = patchesInImage_lr.size();
		for (int j = 0; j < numOfPatch; j++) {
			patches_lr_4x_2.push_back(getDemeanFace(patchesInImage_lr[j]));
			patches_hr_4x_2.push_back(getDemeanFace(patchesInImage_hr[j]));
			features_4x_2.push_back(getFeature(feature1InImage_lr[j], feature2InImage_lr[j], feature3InImage_lr[j], feature4InImage_lr[j]));
		}
	}

	vector<Mat> patches_lr_8x_2, patches_hr_8x_2, features_8x_2;

	for (int i = 0; i < numOfImage_2; i++) {
		vector<Mat>patchesInImage_lr = getPatches(imSampleLR_8x_2[i], sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>patchesInImage_hr = getPatches(imSampleHR_2[i], sizeOfPatch_hr_8x, sizeOfOverlapping_hr_8x);
		Mat graX = mySobX(imSampleLR_8x_2[i]);
		Mat graY = mySobY(imSampleLR_8x_2[i]);
		vector<Mat>feature1InImage_lr = getPatches(graX, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature2InImage_lr = getPatches(graY, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature3InImage_lr = getPatches(mySobX(graX), sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		vector<Mat>feature4InImage_lr = getPatches(mySobY(graY), sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x);
		int numOfPatch = patchesInImage_lr.size();
		for (int j = 0; j < numOfPatch; j++) {
			patches_lr_8x_2.push_back(getDemeanFace(patchesInImage_lr[j]));
			patches_hr_8x_2.push_back(getDemeanFace(patchesInImage_hr[j]));
			features_8x_2.push_back(getFeature(feature1InImage_lr[j], feature2InImage_lr[j], feature3InImage_lr[j], feature4InImage_lr[j]));
		}
	}


	/*
	SSR is preloaded below.
	*/

	mclmcrInitialize();
	myFusedLassoInitialize();

	double a = 0.0001;
	double b = 0.0001;

	//cout << "Training samples for SSR are ready.\n";

	/*
	Super-resolution is done below.
	*/
	cout << 1 << endl;
	std::cout.flush();

	while (1) {
		/*
		load parameters through cin
		*/

		/*
		cout << "Please select the dataset 1 or 2: ";
		string temp_dataset;
		cin >> temp_dataset;
		int dataset = stoi(temp_dataset);


		cout << "Please enter the upscaling factor: ";
		string temp_upscale;
		cin >> temp_upscale;
		int upscale = stoi(temp_upscale);

		cout << "Please enter the name of method: ";
		string temp_method;
		cin >> temp_method;
		String method = temp_method;

		cout << "Please enter the path of input LR image: ";
		string temp_test_lr;
		cin >> temp_test_lr;
		String test_lr = temp_test_lr;

		cout << "Please enter the path output HR image: ";
		string temp_test_sr;
		cin >> temp_test_sr;
		String test_sr = temp_test_sr;
		*/

		/*
		load parameters through txt
		*/

		string temp;
		cin >> temp;

		const char* filename = "parameters.txt";
		readParameter(filename);
		filename = "parameters.txt";
		readParameter(filename);
		//cout << "Parameters loaded.\n";
		//cout << "Dataset: " << dataset <<endl << "Upscaling Factor: " << upscale << endl << "Method: " << method << endl << "LR path: " << test_lr << endl << "SR path: "  << test_sr << endl;

		Mat imTest = imread(test_lr);
		Mat ImResult;
		if (method == "PCA") {
			if (dataset == 1) {
				if (upscale == 4) {
					ImResult = getHR(imTest, upscale, MeanFaceLR_4x_1, MeanFaceHR_1, numOfImage_1, EigenVecLR_4x_1, EigenValLRN_4x_1, demeanImageLR_4x_1, demeanImageHR_1, numOfRowsHR_1, numOfColsHR_1, alpha);
				}
				else if (upscale == 8) {
					ImResult = getHR(imTest, upscale, MeanFaceLR_8x_1, MeanFaceHR_1, numOfImage_1, EigenVecLR_8x_1, EigenValLRN_8x_1, demeanImageLR_8x_1, demeanImageHR_1, numOfRowsHR_1, numOfColsHR_1, alpha);
				}
			}
			else if (dataset == 2) {
				if (upscale == 4) {
					ImResult = getHR(imTest, upscale, MeanFaceLR_4x_2, MeanFaceHR_2, numOfImage_2, EigenVecLR_4x_2, EigenValLRN_4x_2, demeanImageLR_4x_2, demeanImageHR_2, numOfRowsHR_2, numOfColsHR_2, alpha);
				}
				else if (upscale == 8) {
					ImResult = getHR(imTest, upscale, MeanFaceLR_8x_2, MeanFaceHR_2, numOfImage_2, EigenVecLR_8x_2, EigenValLRN_8x_2, demeanImageLR_8x_2, demeanImageHR_2, numOfRowsHR_2, numOfColsHR_2, alpha);
				}
			}
		}
		if (method == "LLE") {
			if (dataset == 1) {
				if (upscale == 4) {
					ImResult = getHR_lle(upscale, imTest, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x, features_4x_1, k, patches_lr_4x_1, sizeOfPatch_hr_4x, patches_hr_4x_1, numOfpatch_j_4x_1, numOfpatch_i_4x_1, sizeOfOverlapping_hr_4x, numOfRowsHR_1, numOfColsHR_1, test_sr);
				}
				else if (upscale == 8) {
					ImResult = getHR_lle(upscale, imTest, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x, features_8x_1, k, patches_lr_8x_1, sizeOfPatch_hr_8x, patches_hr_8x_1, numOfpatch_j_8x_1, numOfpatch_i_8x_1, sizeOfOverlapping_hr_8x, numOfRowsHR_1, numOfColsHR_1, test_sr);
				}
			}
			else if (dataset == 2) {
				if (upscale == 4) {
					ImResult = getHR_lle(upscale, imTest, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x, features_4x_2, k, patches_lr_4x_2, sizeOfPatch_hr_4x, patches_hr_4x_2, numOfpatch_j_4x_2, numOfpatch_i_4x_2, sizeOfOverlapping_hr_4x, numOfRowsHR_2, numOfColsHR_2, test_sr);
				}
				else if (upscale == 8) {
					ImResult = getHR_lle(upscale, imTest, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x, features_8x_2, k, patches_lr_8x_2, sizeOfPatch_hr_8x, patches_hr_8x_2, numOfpatch_j_8x_2, numOfpatch_i_8x_2, sizeOfOverlapping_hr_8x, numOfRowsHR_2, numOfColsHR_2, test_sr);
				}
			}
		}
		if (method == "SSR") {
			if (dataset == 1) {
				if (upscale == 4) {
					ImResult = getHR_ssr(upscale, imTest, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x, imSampleLR_4x_1, imSampleHR_1, numOfImage_1, sizeOfPatch_hr_4x, numOfpatch_j_4x_1, numOfpatch_i_4x_1, sizeOfOverlapping_hr_4x, numOfRowsHR_1, numOfColsHR_1, a, b);
				}
				else if (upscale == 8) {
					ImResult = getHR_ssr(upscale, imTest, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x, imSampleLR_8x_1, imSampleHR_1, numOfImage_1, sizeOfPatch_hr_8x, numOfpatch_j_8x_1, numOfpatch_i_8x_1, sizeOfOverlapping_hr_8x, numOfRowsHR_1, numOfColsHR_1, a, b);
				}
			}
			else if (dataset == 2) {
				if (upscale == 4) {
					ImResult = getHR_ssr(upscale, imTest, sizeOfPatch_lr_4x, sizeOfOverlapping_lr_4x, imSampleLR_4x_2, imSampleHR_2, numOfImage_2, sizeOfPatch_hr_4x, numOfpatch_j_4x_2, numOfpatch_i_4x_2, sizeOfOverlapping_hr_4x, numOfRowsHR_2, numOfColsHR_2, a, b);
				}
				else if (upscale == 8) {
					ImResult = getHR_ssr(upscale, imTest, sizeOfPatch_lr_8x, sizeOfOverlapping_lr_8x, imSampleLR_8x_2, imSampleHR_2, numOfImage_2, sizeOfPatch_hr_8x, numOfpatch_j_8x_2, numOfpatch_i_8x_2, sizeOfOverlapping_hr_8x, numOfRowsHR_2, numOfColsHR_2, a, b);
				}
			}
		}
		//cout << "Image generated\n";
		imwrite(test_sr, ImResult);
		cout << 1 << endl;
		std::cout.flush();
	}

	
	system("pause");
	return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "myFusedLasso.h"


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


Mat getMeanFace(Mat image) {
	int numOfRows = image.rows;
	int numOfCols = image.cols;
	double sum = 0;
	for (int i = 0; i < numOfRows; i++) {
		for (int j = 0; j < numOfCols; j++) {
				sum  = sum + image.at <double>(i, j);
		}
	}
	sum = sum/(numOfRows*numOfCols) ;
	Mat output = Mat1d(numOfRows, numOfCols, sum);
	return output;
}

Mat getDemeanFace(Mat image) {
	Mat MeanFace = getMeanFace(image);
	return image - MeanFace;
}

vector<Mat> imageLoad(string path) {
	vector<String> imageNames;
	vector<Mat> images;

	glob(path, imageNames, false);

	int numOfImage = imageNames.size();

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

double calDistance(Mat v1, Mat v2) {
	Mat M3 = v1 - v2;
	double ans = M3.dot(M3);

	return ans;
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
	mclmcrInitialize();
	myFusedLassoInitialize();

	path_sample_lr = "sample_lr/*";
	path_sample_hr = "sample_hr/*";
	path_test_lr = "test_lr/*";
	path_test_hr = "test_ssr/";
	//upscale = 4;

	vector<Mat> imSampleLR = imageLoad(path_sample_lr);
	vector<Mat> imSampleHR = imageLoad(path_sample_hr);

	upscale = imSampleHR[0].rows / imSampleLR[0].rows;

	vector<Mat> imTestLR = imageLoad2(path_test_lr);
	vector<String> imNames = getImageName(path_test_lr);
	int test_size = imTestLR.size();

	int sizeOfPatch_lr = 4;
	int sizeOfOverlapping_lr = 2;

	double a = 0.00001;
	double b = 0.001;

	int sizeOfPatch_hr = sizeOfPatch_lr * upscale;
	int sizeOfOverlapping_hr = sizeOfOverlapping_lr * upscale;





	int numOfImage = imSampleHR.size();

	int numOfRowsLR = imSampleLR[0].rows;
	int numOfColsLR = imSampleLR[0].cols;
	int numOfRowsHR = imSampleHR[0].rows;
	int numOfColsHR = imSampleHR[0].cols;
	int numOfpatch_j = floor((numOfRowsLR - sizeOfOverlapping_lr) / (sizeOfPatch_lr - sizeOfOverlapping_lr));
	int numOfpatch_i = floor((numOfColsLR - sizeOfOverlapping_lr) / (sizeOfPatch_lr - sizeOfOverlapping_lr));


	for (int i = 0; i < test_size; i++) {
		Mat imTest = imTestLR[i];
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
			resize(getMeanFace(patch_test[i]), meanFaceHR, Size(), upscale, upscale, INTER_NEAREST);
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

		resize(imTest, imTest, Size(), upscale, upscale, INTER_NEAREST);


		string file = path_test_hr + imNames[i];
		cout << file << endl;
		imwrite(file, test_hr_3);

	}
	return 0;
}
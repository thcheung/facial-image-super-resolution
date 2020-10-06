#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


using namespace cv;
using namespace std;

Mat getMeanFace(Mat image) {
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


Mat getFeature(Mat feature1,Mat feature2, Mat feature3, Mat feature4) {
	vector<Mat> features;
	features.push_back(feature1);
	features.push_back(feature2);
	features.push_back(feature3);
	features.push_back(feature4);
	Mat featureM = imagesResize(features, 4);
	return imageResize(featureM);
}

double calDistance(Mat v1, Mat v2) {

	return norm(v1-v2,NORM_L1);
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

Mat patch2Image(vector<Mat> patches,int numOfPatch_i,int numOfPatch_j, int sizeOfPatch, int sizeOfOverlapping,int numOfRowsHR,int numOfColsHR) {

	int step_size = sizeOfPatch - sizeOfOverlapping;
	Mat Result = Mat1d(numOfRowsHR, numOfColsHR, 0.0);
	Mat Count = Mat1d(numOfRowsHR, numOfColsHR, 0.0);
	for (int i = 0; i < numOfPatch_i; i++) {
		for (int j = 0; j < numOfPatch_j; j++) {
			int index = j * numOfPatch_i + i;
			for (int m = 0; m < sizeOfPatch; m++) {
				for (int n = 0; n < sizeOfPatch; n++) {
					Result.at<double>(i*step_size + m, j*step_size + n) = Result.at<double>(i*step_size + m, j*step_size + n) + patches[index].at<double>(m,n);
					Count.at<double>(i*step_size + m, j*step_size + n) = Count.at<double>(i*step_size + m, j*step_size + n) + 1;

				}
			}

		}
	}
	divide(Result, Count, Result);
	return Result;
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
			imgOut.at<double>(i, j) = -1 * imgIn.at<double>(i - 1, j - 1) -2 * imgIn.at<double>(i, j - 1)
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

	for (int i = 0;  i < numOfRows; i++) {
		for (int j = 0; j < numOfColumns; j++) {
			imgOut.at<double>(i, j) = 0;
		}
	}

	for (int i = 1; i < numOfRows - 1; i++) {
		for (int j = 1; j < numOfColumns - 1; j++) {
			imgOut.at<double>(i, j) = 1 * imgIn.at<double>(i - 1, j - 1) + 2 * imgIn.at<double>(i - 1, j)
				+ 1 * imgIn.at<double>(i - 1, j + 1) -1 * imgIn.at<double>(i + 1, j - 1)
				- 2 * imgIn.at<double>(i + 1, j) + -1 * imgIn.at<double>(i + 1, j + 1);
		}
	}
	return imgOut;
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
	path_test_hr = "test_lle/";

	vector<Mat> imSampleLR = imageLoad(path_sample_lr);
	vector<Mat> imSampleHR = imageLoad(path_sample_hr);
	upscale = imSampleHR[0].rows / imSampleLR[0].rows;

	vector<Mat> imTestLR = imageLoad2(path_test_lr);
	vector<String> imNames = getImageName(path_test_lr);

	int k = 3;

	int sizeOfPatch_lr = 4;
	int sizeOfOverlapping_lr = 2;
	int sizeOfPatch_hr = sizeOfPatch_lr * upscale;
	int sizeOfOverlapping_hr = sizeOfOverlapping_lr * upscale;


	int numOfImage = imSampleHR.size();
	int test_size = imTestLR.size();

	int numOfRowsLR = imSampleLR[0].rows;
	int numOfColsLR = imSampleLR[0].cols;
	int numOfRowsHR = imSampleHR[0].rows;
	int numOfColsHR = imSampleHR[0].cols;
	int numOfpatch_j = floor((numOfRowsLR - sizeOfOverlapping_lr) / (sizeOfPatch_lr - sizeOfOverlapping_lr));
	int numOfpatch_i = floor((numOfColsLR - sizeOfOverlapping_lr) / (sizeOfPatch_lr - sizeOfOverlapping_lr));

	vector<Mat> patches_lr, patches_hr, features;

	for (int i = 0; i < numOfImage; i++) {
		vector<Mat>patchesInImage_lr = getPatches(imSampleLR[i], sizeOfPatch_lr, sizeOfOverlapping_lr);
		vector<Mat>patchesInImage_hr = getPatches(imSampleHR[i], sizeOfPatch_hr, sizeOfOverlapping_hr);
		Mat graX = mySobX(imSampleLR[i]);
		Mat graY = mySobY(imSampleLR[i]);
		vector<Mat>feature1InImage_lr = getPatches(graX, sizeOfPatch_lr, sizeOfOverlapping_lr);
		vector<Mat>feature2InImage_lr = getPatches(graY, sizeOfPatch_lr, sizeOfOverlapping_lr);
		vector<Mat>feature3InImage_lr = getPatches(mySobX(graX), sizeOfPatch_lr, sizeOfOverlapping_lr);
		vector<Mat>feature4InImage_lr = getPatches(mySobY(graY), sizeOfPatch_lr, sizeOfOverlapping_lr);
		int numOfPatch = patchesInImage_lr.size();
		for (int j = 0; j < numOfPatch; j++) {
			patches_lr.push_back(getDemeanFace(patchesInImage_lr[j]));
			patches_hr.push_back(getDemeanFace(patchesInImage_hr[j]));
			features.push_back(getFeature(feature1InImage_lr[j], feature2InImage_lr[j], feature3InImage_lr[j], feature4InImage_lr[j]));

		}
	}

	for (int n = 0; n < test_size; n++) {
		Mat imTest = imTestLR[n];
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

		string file = path_test_hr + imNames[n];
		cout << file << endl;
		imwrite(file, test_hr_3);
	}
	return 0;
}
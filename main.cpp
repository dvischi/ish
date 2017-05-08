
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

// Matlab
#include "mat.h"

// LibSVM
#include "svm.h"

// OpenCV
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>


typedef std::vector<std::string> CSVHeader;
typedef std::vector<std::string> CSVRow;
typedef std::vector<std::string> CSVCol;
typedef std::vector<CSVRow> CSVData;

bool readCsvFile(std::string, CSVHeader&, CSVData&);



// Utility Functions
#include <functional>
#include <cctype>
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}

static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}


struct  PointSignal {
	int x;
	int y;
	cv::Vec3f rgb;
};

std::vector<cv::Vec3f> colors = {
	cv::Vec3f(1, 0, 0),
	cv::Vec3f(0, 0, 0),
	cv::Vec3f(0.9, 0.9, 0.9),
	cv::Vec3f(0, 0, 1),
	cv::Vec3f(0.2510, 0.8784, 0.8157)
};

void classifyPoints(
	cv::Mat image, std::vector<cv::Vec3f> circles, int r, struct svm_model *svmModel,
	std::vector<PointSignal> cep, std::vector<PointSignal> gene
) {
	for (std::vector<cv::Vec3f>::iterator it = circles.begin(); it != circles.begin() + 1; ++it) {
		int x, y;
		x = cvFloor((*it)[0]);
		y = cvFloor((*it)[1]);
		cv::Mat xdata;

		try {
			xdata = image(cv::Range(x - r, y - r), cv::Range(x + r, y + r));
		}
		catch (const std::exception&) {
			// xdata was located outside the image
			continue;
		}
		
		// convert image data from byte [0, 255] to float [0.0, 1.0]
		image.convertTo(image, CV_32FC1);

		int I = svm_predict(svmModel, (svm_node*) &xdata);

		switch (I)
		{
		case 1:
			cep.push_back({ x, y, colors[I] });
			break;
		case 2:
			gene.push_back({ x, y, colors[I] });
			break;
		case 3:
			cep.push_back({ x, y, colors[I] });
			gene.push_back({ x, y, colors[I] });
			break;
		default:
			// do nothing
			break;
		}
	}
}

// Main
int main() {
	std::string imgPath = ".\\images\\";
	std::string csvPath = ".\\";
	std::string svmPath = ".\\";

	CSVHeader header;
	CSVData data;

	if (!readCsvFile(csvPath + "demo1.csv", header, data)) {
		std::cerr << "Could not load CSV file!" << std::endl;
		exit(1);
	}

	int label_idx = -1;
	int files_idx = -1;
	CSVCol labels;
	CSVCol files;

	for (size_t col_idx = 0; col_idx < header.size(); col_idx++) {
		if (header[col_idx] == "Label")
			label_idx = col_idx;
		if (header[col_idx] == "Loc")
			files_idx = col_idx;
	}

	if (label_idx == -1 || files_idx == -1) {
		std::cerr << "Invalid headers!" << std::endl;
		exit(1);
	}

	for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
		labels.push_back(data[row_idx][label_idx]);
		files.push_back(rtrim(data[row_idx][files_idx]));
	}

	/*
	for (size_t col_idx = 0; col_idx < labels.size(); col_idx++) {
		std::cout << labels[col_idx] << ", ";
	}
	*/

	struct svm_model *svmModel;

	svmModel = svm_load_model(std::string(svmPath + "model.svm").c_str());
	if (svmModel == nullptr) {
		std::cerr << "Could not read model from MAT file!" << std::endl;
		exit(1);
	}

	//std::cout << files[0] << ", ";
	//std::cout << svmModel->rho[0] << svmModel->rho[1] << ", ";

	//std::cin.ignore();

	int r = 3; // point signal radius
	int r_min = r - 2;
	int r_max = r + 4;

	//for (std::vector<std::string>::iterator it = files.begin(); it != files.end(); ++it) {
	for (std::vector<std::string>::iterator it = files.begin(); it != files.begin() + 1; ++it) {
		cv::Mat image = cv::imread(imgPath + *it + "_PTEN_Zeiss_4096.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
		cv::Mat gray;
		if (!image.data)                              // Check for invalid input
		{
			std::cerr << "Could not read or find image file!" << std::endl;
			exit(1);
		}

		cv::cvtColor(image, gray, CV_BGR2GRAY);
		// smooth it, otherwise a lot of false circles may be detected
		GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

		std::vector<cv::Vec3f> circles;
		cv::HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1.0, r, 25.0, 10.0, r_min, r_max);

		std::cout << "# circles: " << circles.size() << std::endl;

		for (size_t i = 0; i < circles.size(); i++)
		{
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center
			cv::circle(image, center, 30, cv::Scalar(0, 255, 0), -1, 8, 0);
			// draw the circle outline
			cv::circle(image, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
		}

		std::vector<PointSignal> cep;
		std::vector<PointSignal> gene;

		classifyPoints(image, circles, r, svmModel, cep, gene);

		for (std::vector<PointSignal>::iterator it = cep.begin(); it != cep.begin() + 1; ++it)
			std::cout << "[" << it->x << "," << it->y << "]: " << it->rgb << std::endl;
		std::cin.ignore();

		cv::resize(image, image, cv::Size(1000, 1000));
		cv::imshow("Display window", image);
		cv::waitKey(0);
	}


	//svm_free_model_content(svmModel);
	svm_free_and_destroy_model(&svmModel);
		
	//dim = length(imread(strcat(imagepath, files{ 1 }, '.jpg'))); % image size

	std::cin.ignore();
}

bool readCsvFile(std::string csvPath, CSVHeader& header, CSVData& data)
{
	std::ifstream infile(csvPath.c_str());

	if (!infile)
	{
		return false;
	}

	header.resize(0);
	CSVRow row(0);
	data.resize(0);

	std::string line;
	while (std::getline(infile, line))
	{
		if (row.empty()) {
			size_t n = std::count(line.begin(), line.end(), ';');
			row.resize(n+1);
		}

		int i = 0;
		std::string cell;
		std::istringstream iss(line);
		while (std::getline(iss, cell, ';'))
		{
			row[i++] = cell;
		};

		if (header.empty()) {
			header.insert(header.begin(), row.begin(), row.end());
		}
		else {
			data.push_back(row);
		}
	}

	infile.close();

	return true;
}
/*
* ISHProfiler -- an image-based computational workflow for accurate detection of molecular signals
* <https://github.com/dvischi/ish>
*
* Copyright (c) 2017 Dario Vischi
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*/


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>

// Matlab
#include "mat.h"

// OpenCV
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>


typedef std::vector<std::string> CSVHeader;
typedef std::vector<std::string> CSVRow;
typedef std::vector<std::string> CSVCol;
typedef std::vector<CSVRow> CSVData;

bool readCsvFile(std::string, CSVHeader&, CSVData&);



// Utility Functions
//#include <algorithm>
#include <functional>
#include <cctype>
//#include <locale>
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
	cv::Scalar rgb;
};


struct  PointSignalPositions {
	std::vector<PointSignal> cep;
	std::vector<PointSignal> gene;
};


std::vector<cv::Scalar> colors = {
	cv::Scalar(255, 0, 0), // (1, 0, 0)
	cv::Scalar(0, 0, 0), // (0, 0, 0)
	cv::Scalar(230, 230, 230), // (0.9f, 0.9f, 0.9f)
	cv::Scalar(0, 0, 255), // (0, 0, 1)
	cv::Scalar(64, 224, 208) // (0.2510f, 0.8784f, 0.8157f)
};

void classifyPoints(
	cv::Mat& image, std::vector<cv::Vec3f> circles, int r, cv::SVM& svmModel,
	std::vector<PointSignal>& cep, std::vector<PointSignal>& gene
) {
	for (std::vector<cv::Vec3f>::iterator it = circles.begin(); it != circles.end(); ++it) {
		int x, y;
		x = cvFloor((*it)[0]);
		y = cvFloor((*it)[1]);
		cv::Mat roi;
		
		cv::Rect rect;
		try {
			rect = cv::Rect(x - r, y - r, 2 * r + 1, 2 * r + 1);
			roi = image(rect).clone();
		}
		catch (const std::exception&) {
			std::cout << "Warning: Index out of bounds: " << rect << std::endl;
			continue;
		}
		
		roi = roi.reshape(1, 1);

		// convert xdata from byte [0, 255] to float [0.0, 1.0]
		roi.convertTo(roi, CV_32FC1);
		roi /= 255;

		int I = static_cast<int>( svmModel.predict(roi, true) );

		switch (I)
		{
		case 1:
			cep.push_back({ x, y, colors[I-1] });
			break;
		case 2:
			gene.push_back({ x, y, colors[I-1] });
			break;
		case 5:
			cep.push_back({ x, y, colors[I-1] });
			gene.push_back({ x, y, colors[I-1] });
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

	cv::SVM svmModel;
	svmModel.load(std::string(svmPath + "svm.xml").c_str(), "ish_svm");

	int r = 3; // point signal radius
	int r_min = r - 2;
	int r_max = r + 4;
	cv::Mat gRatio(files.size(), 1, CV_32FC1);
	std::vector<PointSignalPositions> pos(files.size());

	//for (std::vector<std::string>::iterator it = files.begin(); it != files.end(); ++it) {
	for (std::vector<std::string>::iterator it = files.begin(); it != files.begin() + 1; ++it) {
		std::cout << "load image: " << *it + "_PTEN_Zeiss_4096.jpg" << std::endl;
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

		int n = std::distance(files.begin(), it);

		std::vector<PointSignal>& cep = pos[n].cep;
		std::vector<PointSignal>& gene = pos[n].gene;

		classifyPoints(image, circles, r, svmModel, cep, gene);

		std::cout << "# cep: " << pos[n].cep.size() << std::endl;
		std::cout << "# gene: " << pos[n].gene.size() << std::endl;

		if (cep.size() == 0)
			gRatio.at<float>(n, 0) = 0.0f;
		else {
			gRatio.at<float>(n, 0) = static_cast<float>( gene.size() / cep.size() );
		}
	}

	// ploting a sample image
	// int n = 66;
	int n = 0;
	cv::Mat image = cv::imread(imgPath + files[n] + "_PTEN_Zeiss_4096.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	std::cout << "# cep: " << pos[n].cep.size() << std::endl;
	for (size_t i = 0; i < pos[n].cep.size(); i++)
	{
		cv::Point center(cvRound(pos[n].cep[i].x), cvRound(pos[n].cep[i].y));
		int radius = cvRound(10);
		// draw the circle center
		//cv::circle(image, center, 30, cv::Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		cv::circle(image, center, radius, pos[n].cep[i].rgb, 3, 8, 0);
	}
	std::cout << "# gene: " << pos[n].gene.size() << std::endl;
	for (size_t i = 0; i < pos[n].gene.size(); i++)
	{
		cv::Point center(cvRound(pos[n].gene[i].x), cvRound(pos[n].gene[i].y));
		int radius = cvRound(10);
		// draw the circle center
		//cv::circle(image, center, 30, cv::Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		cv::circle(image, center, radius, pos[n].gene[i].rgb, 3, 8, 0);
	}

	cv::resize(image, image, cv::Size(1000, 1000));
	cv::imshow("Display window", image);
	cv::waitKey(0);
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
#include <opencv2/highgui/highgui.hpp>
#include "SlitherOCR.h"

#include <string>
#include <vector>
#include <iostream>

int main(int argc, const char* argv[])
{
	SlitherOCR ocr;
	//ocr.Train("train.txt", "slitherocr.dat");
	//ocr.TrainOrientation("train.txt", "slitherocr_ori.dat");
	ocr.LoadTrainedData("slitherocr.dat");
	ocr.LoadTrainedOrientationData("slitherocr_ori.dat");
	ocr.Load(argv[1]);
	
	std::vector<std::vector<int> > problem = ocr.OCR();

	if (problem.size() <= 2 || problem[0].size() <= 2) {
		std::cerr << "Error: " << argv[1] << ": Too small problem was detected";
		return 1;
	}

	bool improper_cell = false;
	for (auto &line : problem) {
		for (int v : line) if (v == -2) improper_cell = true;
	}

	if (improper_cell) {
		std::cerr << "Warning: " << argv[1] << ": Improper cell was detected";
	}

	std::cout << problem.size() << " " << problem[0].size() << "\n";
	for (std::vector<int> &line : problem) {
		for (int v : line) std::cout << (char)(v == -2 ? '*' : v == -1 ? '-' : (v + '0'));
		std::cout << "\n";
	}
	std::cout << "\n";

	ocr.Show();

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

#include <opencv2/highgui/highgui.hpp>
#include "SlitherOCR.h"

#include <string>

int main(int argc, const char* argv[])
{
	SlitherOCR ocr;
	//ocr.Train("train.txt", "slitherocr.dat");
	ocr.LoadTrainedData("slitherocr.dat");
	ocr.Load(argv[1]);
	ocr.ExtractBinary();
	ocr.DetectDots();
	ocr.ExcludeFalseDots();

	while (1) {
		ocr.ComputeGridLine();
		if (!ocr.RetriveUncaughtDots()) break;
	}

	ocr.ComputeGridCell();
	ocr.Show();

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

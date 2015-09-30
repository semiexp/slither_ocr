#include <opencv2/highgui/highgui.hpp>
#include "SlitherOCR.h"

#include <string>

int main(int argc, const char* argv[])
{
	SlitherOCR ocr;
	//ocr.Train("train.txt", "slitherocr.dat");
	//ocr.TrainOrientation("train.txt", "slitherocr_ori.dat");
	ocr.LoadTrainedData("slitherocr.dat");
	ocr.LoadTrainedOrientationData("slitherocr_ori.dat");
	ocr.Load(argv[1]);
	ocr.ExtractBinary();
	ocr.DetectDots();
	ocr.ExcludeFalseDots();

	while (1) {
		ocr.ComputeGridLine();
		if (!ocr.RetriveUncaughtDots()) break;
	}
	
	ocr.RemoveImproperEdges();
	ocr.ComputeGridCell();
	ocr.RecognizeProblem();
	ocr.Show();

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

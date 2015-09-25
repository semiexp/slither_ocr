#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

struct unionfind
{
	std::vector<int> val;

	unionfind(int n) {
		for (int i = 0; i < n; ++i) val.push_back(-1);
	}

	int root(int p) {
		return val[p] < 0 ? p : (val[p] = root(val[p]));
	}

	bool join(int p, int q) {
		p = root(p);
		q = root(q);
		if (p == q) return false;
		val[p] += val[q];
		val[q] = p;
		return true;
	}
};

class SlitherOCR
{
public:
	SlitherOCR() : data(nullptr), problem_height(-1), problem_width(-1) {}
	~SlitherOCR() { if (data) delete[] data; }
	void Load(const char* fn);
	void Load(cv::Mat &img);
	void ExtractBinary();
	void DetectDots();
	int ComputeThresholdForDetectDots();
	void ExcludeFalseDots();
	void ComputeGridLine();
	void RemoveImproperEdges();
	void ComputeGridCell();

	void Show();

	void Train(const char* input_file, const char* output_file);
	void LoadTrainedData(const char *filed);

private:
	struct rect {
		int ul, ur, bl, br; // (Upper / Bottom) (Left / Right)
	};

	int& at(int y, int x) { return data[y * img_width + x]; }
	int id(int y, int x) { return y * img_width + x; }
	int next_point(int from, int now) {
		for (int i = 0; i < grid[now].size(); ++i) if (grid[now][i] == from) return grid[now][(i + 1) % grid[now].size()];
		return -1;
	}
	rect rect_right(rect &r);
	rect rect_bottom(rect &r);
	void remove_edge(int p, int q);

	bool IsPossibleRect(rect &r);
	bool IsPossibleNeighborhood(int center, std::vector<int> &nb);
	
	cv::Mat ClipCell(rect &r, int size);
	void ReduceNoiseFromClip(cv::Mat &pic);

	int Recognize(cv::Mat &pic);
	void MarkDot(int y, int x);

	static const int CLIP_SIZE = 24;

	cv::Mat image;
	int img_height, img_width;
	int problem_height, problem_width;

	int *data;
	std::vector<int> dot_y, dot_x, dot_rep_y, dot_rep_x;
	std::vector<std::vector<int> > grid;

	cv::Ptr<cv::ml::SVM> svm;
};

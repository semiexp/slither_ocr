#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

struct unionfind
{
	std::vector<int> parent, val;

	void init(int n) {
		parent.clear(); val.clear();
		for (int i = 0; i < n; ++i) parent.push_back(-1);
		for (int i = 0; i < n; ++i) val.push_back(-1);
	}

	int root(int p) {
		return parent[p] < 0 ? p : (parent[p] = root(parent[p]));
	}

	bool join(int p, int q) {
		p = root(p);
		q = root(q);
		if (p == q) return false;
		parent[p] += parent[q];
		parent[q] = p;
		return true;
	}

	int union_size(int p) {
		return -parent[root(p)];
	}

	// NOTE: [x] can be incorrect value after join(x, y)
	int &operator[](const int idx) {
		return val[root(idx)];
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
	int RetriveUncaughtDots();
	std::vector<std::vector<int> > RecognizeProblem();

	void Show();

	void Train(const char* input_file, const char* output_file);
	void LoadTrainedData(const char *filed);

private:
	union rect {
		struct {
			int ul, ur, bl, br; // (Upper / Bottom) (Left / Right)
		};
		struct {
			int top, bottom, left, right;
		};
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

	static const int CLIP_SIZE = 24;

	cv::Mat image;
	int img_height, img_width;
	int problem_height, problem_width;

	int *data;
	std::vector<int> dot_y, dot_x, dot_rep_y, dot_rep_x;
	std::vector<std::vector<int> > grid;
	std::vector<std::vector<rect> > cells;

	unionfind units;
	std::vector<rect> unit_boundary;
	std::vector<bool> is_dot;

	cv::Ptr<cv::ml::SVM> svm;
};

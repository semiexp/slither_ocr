#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <cmath>

#include "SlitherOCR.h"

SlitherOCR::rect SlitherOCR::rect_right(rect &r)
{
	rect ret;
	ret.ul = r.ur;
	ret.bl = r.br;
	ret.br = next_point(ret.ul, ret.bl);
	ret.ur = next_point(ret.bl, ret.br);
	return ret;
}

SlitherOCR::rect SlitherOCR::rect_bottom(rect &r)
{
	rect ret;
	ret.ul = r.bl;
	ret.ur = r.br;
	ret.bl = next_point(ret.ur, ret.ul);
	ret.br = next_point(ret.ul, ret.bl);
	return ret;
}

void SlitherOCR::remove_edge(int p, int q)
{
	for (int i = 0; i < grid[p].size(); ++i) {
		if (grid[p][i] == q) {
			grid[p].erase(grid[p].begin() + i);
			break;
		}
	}
	for (int i = 0; i < grid[q].size(); ++i) {
		if (grid[q][i] == p) {
			grid[q].erase(grid[q].begin() + i);
			break;
		}
	}
}

bool SlitherOCR::IsPossibleRect(rect &r)
{
	if (dot_y[r.ul] + dot_x[r.ul] > dot_y[r.br] + dot_x[r.br]) return false;
	if (dot_y[r.ur] - dot_x[r.ur] > dot_y[r.bl] - dot_x[r.bl]) return false;
	if (next_point(r.ul, r.bl) != r.br) return false;
	if (next_point(r.bl, r.br) != r.ur) return false;
	if (next_point(r.br, r.ur) != r.ul) return false;
	if (next_point(r.ur, r.ul) != r.bl) return false;
	return true;
}

bool SlitherOCR::IsPossibleNeighborhood(int center, std::vector<int> &nb)
{
	static const double PI = 3.14159265358979323846;

	for (int _j = 0; _j < nb.size(); ++_j) {
		int j = nb[_j];
		for (int _k = _j + 1; _k < nb.size(); ++_k) {
			int k = nb[_k];
			int angle = (atan2(dot_y[k] - dot_y[center], dot_x[k] - dot_x[center]) - atan2(dot_y[j] - dot_y[center], dot_x[j] - dot_x[center])) / PI * 180;
			angle = (angle + 360) % 360;
			angle %= 90;
			if (25 <= angle && angle <= 65) return false;
		}
	}
	return true;
}

void SlitherOCR::Load(const char* fn)
{
	image = cv::imread(fn, cv::IMREAD_GRAYSCALE);

	cv::Mat image_tmp;
	resize(image, image_tmp, cv::Size(), 1, 1);
	image = image_tmp;
	img_height = image.rows;
	img_width = image.cols;
}

void SlitherOCR::Load(cv::Mat &img)
{
	cvtColor(img, image, CV_RGB2GRAY);
	img_height = image.rows;
	img_width = image.cols;
}

std::vector<std::vector<int> > SlitherOCR::OCR()
{
	ExtractBinary();
	DetectDots();
	ExcludeFalseDots();

	while (1) {
		ComputeGridLine();
		if (!RetriveUncaughtDots()) break;
	}

	RemoveImproperEdges();
	ComputeGridCell();
	return RecognizeProblem();
}

void SlitherOCR::Show()
{
	cv::Mat image_tmp;
	resize(image, image_tmp, cv::Size(), 0.3, 0.3);

	for (int p = 0; p < grid.size(); ++p) {
		for (int q : grid[p]) {
			line(image_tmp, cv::Point(dot_x[p] * 0.3, dot_y[p] * 0.3), cv::Point(dot_x[q] * 0.3, dot_y[q] * 0.3), cv::Scalar(0), 1);
		}
	}

	cv::namedWindow("ocr");
	cv::imshow("ocr", image_tmp);
}

void SlitherOCR::ExtractBinary()
{
	cv::Mat image_tmp;
	cv::adaptiveThreshold(image, image_tmp, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 15);
	image = image_tmp;

	if (data) delete[] data;

	data = new int[img_height * img_width];
	for (int y = 0; y < img_height; ++y) {
		for (int x = 0; x < img_width; ++x) {
			at(y, x) = (image.at<uchar>(y, x) == 0 ? 1 : 0);
		}
	}
}

int SlitherOCR::ComputeThresholdForDetectDots()
{
	bool *visited = new bool[img_height * img_width];
	std::fill(visited, visited + img_height * img_width, false);

	std::vector<int> cand_sizes;

	for (int y = 0; y < img_height; ++y) {
		for (int x = 0; x < img_width; ++x) {
			if (at(y, x) && !visited[id(y, x)]) {
				std::queue<std::pair<int, int> > Q;
				Q.push(std::make_pair(y, x));

				int top = y, bottom = y, left = x, right = x;
				int dots = 0;

				while (!Q.empty()) {
					int ty = Q.front().first, tx = Q.front().second;
					Q.pop();

					if (ty < 0 || tx < 0 || img_height <= ty || img_width <= tx || !at(ty, tx) || visited[id(ty, tx)]) continue;
					visited[id(ty, tx)] = true;
					top = std::min(top, ty);
					bottom = std::max(bottom, ty);
					left = std::min(left, tx);
					right = std::max(right, tx);
					++dots;

					Q.push(std::make_pair(ty + 1, tx));
					Q.push(std::make_pair(ty - 1, tx));
					Q.push(std::make_pair(ty, tx + 1));
					Q.push(std::make_pair(ty, tx - 1));
				}

				int size = std::max(bottom - top + 1, right - left + 1);
				if (size >= 3 && size * size * 0.7 < dots) {
					cand_sizes.push_back(size);
				}
			}
		}
	}

	std::sort(cand_sizes.begin(), cand_sizes.end());

	delete[] visited;
	return cand_sizes[cand_sizes.size() / 2];
}

void SlitherOCR::DetectDots()
{
	int threshold = ComputeThresholdForDetectDots();
	double threshold_low = threshold / 1.8, threshold_high = threshold * 2.2;

	bool *visited = new bool[img_height * img_width];
	std::fill(visited, visited + img_height * img_width, false);

	std::vector<int> dot_size;
	
	units.init(img_height * img_width);

	for (int y = 0; y < img_height; ++y) {
		for (int x = 0; x < img_width; ++x) {
			if (at(y, x) && !visited[id(y, x)]) {
				std::queue<std::pair<int, int> > Q;
				Q.push(std::make_pair(y, x));

				int top = y, bottom = y, left = x, right = x;
				int dots = 0;

				while (!Q.empty()) {
					int ty = Q.front().first, tx = Q.front().second;
					Q.pop();

					if (ty < 0 || tx < 0 || img_height <= ty || img_width <= tx || !at(ty, tx) || visited[id(ty, tx)]) continue;
					visited[id(ty, tx)] = true;
					top = std::min(top, ty);
					bottom = std::max(bottom, ty);
					left = std::min(left, tx);
					right = std::max(right, tx);
					++dots;

					units.join(id(y, x), id(ty, tx));
					Q.push(std::make_pair(ty + 1, tx));
					Q.push(std::make_pair(ty - 1, tx));
					Q.push(std::make_pair(ty, tx + 1));
					Q.push(std::make_pair(ty, tx - 1));
				}

				rect boundary;
				boundary.left = left;
				boundary.right = right;
				boundary.top = top;
				boundary.bottom = bottom;

				int unit_id = unit_boundary.size();
				units[id(y, x)] = unit_id;
				unit_boundary.push_back(boundary);
				is_dot.push_back(false);

				int height = bottom - top + 1, width = right - left + 1;

				if (threshold_low <= height && height <= threshold_high && threshold_low <= width && width <= threshold_high) {
					dot_y.push_back((top + bottom) / 2);
					dot_x.push_back((left + right) / 2);
					dot_rep_y.push_back(y);
					dot_rep_x.push_back(x);
					dot_size.push_back(dots);
					// is_dot is modified in ExcludeFalseDots
					// is_dot[unit_id] = true;
				}
			}
		}
	}

	delete[] visited;

	std::vector<int> dot_size_tmp = dot_size;
	sort(dot_size_tmp.begin(), dot_size_tmp.end());

	int mean = dot_size_tmp[dot_size_tmp.size() / 2];

	std::vector<int> dot_y2, dot_x2, dot_rep_y2, dot_rep_x2;
	for (int i = 0; i < dot_y.size(); ++i) {
		if (dot_size[i] > mean / 3) {
			dot_y2.push_back(dot_y[i]);
			dot_x2.push_back(dot_x[i]);
			dot_rep_y2.push_back(dot_rep_y[i]);
			dot_rep_x2.push_back(dot_rep_x[i]);
		}
	}
	dot_y2.swap(dot_y);
	dot_x2.swap(dot_x);
	dot_rep_y2.swap(dot_rep_y);
	dot_rep_x2.swap(dot_rep_x);
}

void SlitherOCR::ExcludeFalseDots()
{
	std::vector<bool> visited;
	std::vector<std::pair<int, int> > distance; // (distance, from)
	std::vector<std::pair<int, std::pair<int, int> > > edges;

	for (int i = 0; i < dot_y.size(); ++i) {
		visited.push_back(false);
		distance.push_back(std::make_pair(1 << 30, -1));
	}

	distance[0] = std::make_pair(0, -1);

	for (int i = 0; i < dot_y.size(); ++i) {
		int bp = -1;
		for (int j = 0; j < distance.size(); ++j) if (!visited[j]) {
			if (bp == -1 || distance[bp] > distance[j]) bp = j;
		}

		if (distance[bp].second != -1) {
			edges.push_back(std::make_pair(distance[bp].first, std::make_pair(distance[bp].second, bp)));
		}

		visited[bp] = true;
		for (int j = 0; j < distance.size(); ++j) if (!visited[j]) {
			distance[j] = std::min(distance[j], std::make_pair(
				(dot_y[j] - dot_y[bp]) * (dot_y[j] - dot_y[bp]) + (dot_x[j] - dot_x[bp]) * (dot_x[j] - dot_x[bp]),
				bp));
		}
	}

	int med = edges[edges.size() / 2].first;
	unionfind uf;
	uf.init(dot_y.size());

	sort(edges.begin(), edges.end());
	for (int i = 0; i < edges.size(); ++i) {
		if (edges[i].first > med * 1.4) break;

		uf.join(edges[i].second.first, edges[i].second.second);
	}

	std::pair<int, int> largest_group(-1, -1);
	for (int i = 0; i < dot_y.size(); ++i) {
		if (i == uf.root(i)) largest_group = std::max(largest_group, std::make_pair(uf.union_size(i), i));
	}

	std::vector<int> dot_y2, dot_x2, dot_rep_y2, dot_rep_x2;
	for (int i = 0; i < dot_y.size(); ++i) {
		if (uf.root(i) == largest_group.second) {
			dot_y2.push_back(dot_y[i]);
			dot_x2.push_back(dot_x[i]);
			dot_rep_y2.push_back(dot_rep_y[i]);
			dot_rep_x2.push_back(dot_rep_x[i]);
			is_dot[units[id(dot_rep_y[i], dot_rep_x[i])]] = true;
		}
	}
	dot_y2.swap(dot_y);
	dot_x2.swap(dot_x);
	dot_rep_y2.swap(dot_rep_y);
	dot_rep_x2.swap(dot_rep_x);
}

void SlitherOCR::ComputeGridLine()
{
	static const double PI = 3.14159265358979323846;

	grid.clear();
	for (int i = 0; i < dot_y.size(); ++i) grid.push_back(std::vector<int>());

	for (int i = 0; i < dot_y.size(); ++i) {
		std::vector<std::pair<int, int> > nearest(4, std::make_pair(1 << 30, -1));

		for (int j = 0; j < dot_y.size(); ++j) if (i != j) {
			std::pair<int, int> dist = std::make_pair((dot_y[i] - dot_y[j]) * (dot_y[i] - dot_y[j]) + (dot_x[i] - dot_x[j]) * (dot_x[i] - dot_x[j]), j);
			
			for (int k = 0; k < nearest.size(); ++k) {
				if (nearest[k] > dist) {
					for (int l = nearest.size() - 1; l > k; --l) nearest[l] = nearest[l - 1];
					nearest[k] = dist;
					break;
				}
			}
		}

		for (int _j = nearest.size() - 1; _j >= 0; --_j) {
			int j = nearest[_j].second;
			for (int _k = 0; _k < _j; ++_k) {
				int k = nearest[_k].second;
				int angle = (atan2(dot_y[k] - dot_y[i], dot_x[k] - dot_x[i]) - atan2(dot_y[j] - dot_y[i], dot_x[j] - dot_x[i])) / PI * 180;
				angle = (angle + 360) % 360;

				if (angle <= 15 || 345 <= angle) {
					nearest.erase(nearest.begin() + _j);
					break;
				}
			}
		}

		std::pair<int, int> largest_set(0, 0);
		for (int mask = 1; mask < (1 << nearest.size()); mask += 2) {
			int cnt = 0;

			std::vector<int> nb;
			for (int _j = 0; _j < nearest.size(); ++_j) if (mask & (1 << _j)) {
				nb.push_back(nearest[_j].second);
				++cnt;
			}

			if (IsPossibleNeighborhood(i, nb)) largest_set = std::max(largest_set, std::make_pair(cnt, mask));
		}

		for (int j = 0; j < nearest.size(); ++j) {
			// if (j > 1 && nearest[j].first > nearest[j - 1].first * 1.5) break;
			if (!(largest_set.second & (1 << j))) continue;

			int pt = nearest[j].second;

			grid[i].push_back(pt);
			grid[pt].push_back(i);
		}
	}

	for (int i = 0; i < grid.size(); ++i) {
		std::vector<std::pair<double, int> > graph_aux;

		for (int pt : grid[i]) {
			graph_aux.push_back(std::make_pair(atan2(dot_y[pt] - dot_y[i], dot_x[pt] - dot_x[i]), pt));
		}
		std::sort(graph_aux.begin(), graph_aux.end());

		grid[i].clear();
		for (int j = 0; j < graph_aux.size(); ++j) {
			if (j != 0 && graph_aux[j].second == graph_aux[j - 1].second) continue;
			grid[i].push_back(graph_aux[j].second);
		}
	}
}

void SlitherOCR::RemoveImproperEdges()
{
	for (int i = 0; i < grid.size(); ++i) {
		int max_cnt = -1, cand = 0;

		for (int mask = 0; mask < (1 << grid[i].size()); ++mask) {
			int cnt = 0;
			std::vector<int> nb;

			for (int j = 0; j < grid[i].size(); ++j) if (mask & (1 << j)) {
				++cnt;
				nb.push_back(grid[i][j]);
			}

			if (IsPossibleNeighborhood(i, nb)) {
				if (max_cnt < cnt) {
					max_cnt = cnt;
					cand = mask;
				} else if (max_cnt == cnt) {
					cand |= mask;
				}
			}
		}

		for (int j = (int)grid[i].size() - 1; j >= 0; --j) {
			if (!(cand & (1 << j))) {
				remove_edge(i, grid[i][j]);
			}
		}
	}
}

int SlitherOCR::RetriveUncaughtDots()
{
	int new_dots = 0;

	for (int p = 0; p < grid.size(); ++p) {
		for (int q : grid[p]) {
			int cy = dot_y[p] * 2 - dot_y[q], cx = dot_x[p] * 2 - dot_x[q];

			if (cy < 0 || cx < 0 || img_height <= cy || img_width <= cx || !at(cy, cx)) continue;

			int uid = units[id(cy, cx)];
			if (is_dot[uid]) continue;

			int p_id = units[id(dot_rep_y[p], dot_rep_x[p])];
			int p_height = unit_boundary[p_id].bottom - unit_boundary[p_id].top + 1;
			int p_width = unit_boundary[p_id].right - unit_boundary[p_id].left + 1;
			int u_height = unit_boundary[uid].bottom - unit_boundary[uid].top + 1;
			int u_width = unit_boundary[uid].right - unit_boundary[uid].left + 1;

			double height_ratio = (double)p_height / u_height;
			double width_ratio = (double)p_width / u_width;
			if (height_ratio < 1) height_ratio = 1 / height_ratio;
			if (width_ratio < 1) width_ratio = 1 / width_ratio;

			if (std::max(height_ratio, width_ratio) > 1.2) continue;

			is_dot[uid] = true;
			dot_y.push_back((unit_boundary[uid].top + unit_boundary[uid].bottom) / 2);
			dot_x.push_back((unit_boundary[uid].left + unit_boundary[uid].right) / 2);
			dot_rep_y.push_back(cy);
			dot_rep_x.push_back(cx);
			++new_dots;
		}
	}

	return new_dots;
}

void SlitherOCR::ComputeGridCell()
{
	std::pair<int, int> ul_pt_tmp(1 << 30, -1);
	for (int i = 0; i < dot_y.size(); ++i) ul_pt_tmp = std::min(ul_pt_tmp, std::make_pair(dot_y[i] + dot_x[i], i));

	int ul_pt1 = ul_pt_tmp.second;
	int cand1 = grid[ul_pt1][0], cand2 = grid[ul_pt1][1];
	int ul_pt2 = (dot_x[cand1] - dot_y[cand1] <= dot_x[cand2] - dot_y[cand2]) ? cand1 : cand2;

	rect base, scan_y, scan_x;
	base.ul = ul_pt1;
	base.bl = ul_pt2;
	base.br = next_point(base.ul, base.bl);
	base.ur = next_point(base.bl, base.br);

	scan_y = base;

	std::vector<std::vector<int> > problem;
	cells.clear();

	for (rect scan_y = base; IsPossibleRect(scan_y); scan_y = rect_bottom(scan_y)) {
		std::vector<rect> cell_line;
		for (rect scan_x = scan_y; IsPossibleRect(scan_x); scan_x = rect_right(scan_x)) {
			cell_line.push_back(scan_x);
		}
		cells.push_back(cell_line);
	}
}

std::vector<std::vector<int> > SlitherOCR::RecognizeProblem()
{
	std::vector<std::vector<int> > ret;
	int problem_height, problem_width;

	problem_height = cells.size();
	problem_width = 0;

	int ori_likelihood[4] = { 0, 0, 0, 0 };

	for (auto &line : cells) {
		for (rect &c : line) {
			cv::Mat pic = ClipCell(c, CLIP_SIZE);
			ReduceNoiseFromClip(pic);

			int ori = RecognizeOrientation(pic);
			if (0 <= ori && ori <= 3) ++ori_likelihood[ori];
		}
	}

	int ori = 0;
	for (int i = 1; i < 4; ++i) if (ori_likelihood[ori] < ori_likelihood[i]) ori = i;

	for (auto &line : cells) {
		std::vector<int> ret_line;
		problem_width = std::max(problem_width, (int)line.size());

		for (rect &c : line) {
			cv::Mat pic = ClipCell(c, CLIP_SIZE);
			ReduceNoiseFromClip(pic);

			for (int t = 0; t < ori; ++t) pic = RotateCounterClockwise(pic);
			int v = Recognize(pic);
			ret_line.push_back(v);
		}
		ret.push_back(ret_line);
	}

	for (std::vector<int> &line : ret) {
		while (line.size() < problem_width) line.push_back(-2);
	}

	for (int t = 0; t < ori; ++t) {
		ret = RotateProblemCounterClockwise(ret);
		problem_height = ret.size();
		problem_width = ret[0].size();
	}

	return ret;
}

cv::Mat SlitherOCR::ClipCell(rect &r, int size)
{
	cv::Mat ret(cv::Size(size, size), CV_8UC1);

	for (int y = 0; y < size; ++y) {
		for (int x = 0; x < size; ++x) {
			double y_ratio = y / (double)(size - 1), x_ratio = x / (double)(size - 1);

			double py = (dot_y[r.ul] * (1 - x_ratio) + dot_y[r.ur] * x_ratio) * (1 - y_ratio) + (dot_y[r.bl] * (1 - x_ratio) + dot_y[r.br] * x_ratio) * y_ratio;
			double px = (dot_x[r.ul] * (1 - x_ratio) + dot_x[r.ur] * x_ratio) * (1 - y_ratio) + (dot_x[r.bl] * (1 - x_ratio) + dot_x[r.br] * x_ratio) * y_ratio;

			double cnt = 0;
			if (at((int)py + 0, (int)px + 0) && !is_dot[units[id((int)py + 0, (int)px + 0)]]) cnt += (1 - (py - (int)py)) * (1 - (px - (int)px));
			if (at((int)py + 0, (int)px + 1) && !is_dot[units[id((int)py + 0, (int)px + 1)]]) cnt += (1 - (py - (int)py)) * (px - (int)px);
			if (at((int)py + 1, (int)px + 0) && !is_dot[units[id((int)py + 1, (int)px + 0)]]) cnt += (py - (int)py) * (1 - (px - (int)px));
			if (at((int)py + 1, (int)px + 1) && !is_dot[units[id((int)py + 1, (int)px + 1)]]) cnt += (py - (int)py) * (px - (int)px);

			ret.at<uchar>(y, x) = (cnt >= 0.5 ? 0 : 255);
		}
	}

	return ret;
}

cv::Mat SlitherOCR::RotateCounterClockwise(cv::Mat &pic)
{
	cv::Mat ret(cv::Size(pic.cols, pic.rows), CV_8UC1);

	for (int y = 0; y < pic.rows; ++y) {
		for (int x = 0; x < pic.cols; ++x) {
			int y2 = x;
			int x2 = pic.rows - 1 - y;

			ret.at<uchar>(y, x) = pic.at<uchar>(y2, x2);
		}
	}

	return ret;
}

std::vector<std::vector<int> > SlitherOCR::RotateProblemCounterClockwise(std::vector<std::vector<int> > &problem)
{
	int height = problem.size(), width = 0;
	for (auto &line : problem) width = std::max(width, (int)line.size());

	std::vector<std::vector<int> > ret;
	for (int y = 0; y < width; ++y) {
		std::vector<int> line;

		for (int x = 0; x < height; ++x) {
			int y2 = x;
			int x2 = width - 1 - y;
			
			if (problem.size() <= y2 || problem[y2].size() <= x2) line.push_back(-2);
			else line.push_back(problem[y2][x2]);
		}

		ret.push_back(line);
	}

	return ret;
}

void SlitherOCR::ReduceNoiseFromClip(cv::Mat &pic)
{
	bool *visited = new bool[pic.rows * pic.cols];
	std::fill(visited, visited + pic.rows * pic.cols, false);

	for (int y = 0; y < pic.rows; ++y) {
		for (int x = 0; x < pic.cols; ++x) {
			if (pic.at<uchar>(y, x) == 0 && !visited[y * pic.cols + x]) {
				std::vector<std::pair<int, int> > points;
				std::queue<std::pair<int, int> > Q;
				Q.push(std::make_pair(y, x));

				int top = y, bottom = y, left = x, right = x;
				int dots = 0;

				while (!Q.empty()) {
					int ty = Q.front().first, tx = Q.front().second;
					Q.pop();

					if (ty < 0 || tx < 0 || pic.rows <= ty || pic.cols <= tx || pic.at<uchar>(ty, tx) != 0 || visited[ty * pic.cols + tx]) continue;
					visited[ty * pic.cols + tx] = true;
					points.push_back(std::make_pair(ty, tx));
					top = std::min(top, ty);
					bottom = std::max(bottom, ty);
					left = std::min(left, tx);
					right = std::max(right, tx);
					++dots;

					Q.push(std::make_pair(ty + 1, tx));
					Q.push(std::make_pair(ty - 1, tx));
					Q.push(std::make_pair(ty, tx + 1));
					Q.push(std::make_pair(ty, tx - 1));
				}

				int height = bottom - top + 1, width = right - left + 1;

				if (height < pic.rows / 3 && width < pic.cols / 3) {
					// too small: presumably noise
					for (auto &p : points) {
						pic.at<uchar>(p.first, p.second) = 255;
					}
				}
			}
		}
	}

	delete[] visited;
}

void SlitherOCR::Train(const char *input_file, const char *output_file)
{
	std::ifstream ifs(input_file);

	std::vector<std::vector<int> > data;
	std::vector<int> expected;

	srand(1234);

	for (;;) {
		std::string line;
		int ans;

		ifs >> ans;
		if (ans == -1) break;

		expected.push_back(ans);

		std::vector<int> pic;
		for (int i = 0; i < CLIP_SIZE; ++i) {
			ifs >> line;
			for (int j = 0; j < CLIP_SIZE; ++j) pic.push_back(line[j] == '#' ? 1 : 0);
		}

		data.push_back(pic);

		for (int t = 0; t < 10; ++t) {
			std::vector<int> pic_noisy;
			pic_noisy = pic;

			for (int n = 0; n < 4; ++n) pic_noisy[rand() % (CLIP_SIZE * CLIP_SIZE)] ^= 1;

			expected.push_back(ans);
			data.push_back(pic_noisy);
		}
	}

	cv::Mat data_mat(data.size(), CLIP_SIZE * CLIP_SIZE, CV_32F);
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < CLIP_SIZE * CLIP_SIZE; ++j) {
			data_mat.at<float>(i, j) = data[i][j];
		}
	}
	cv::Mat expected_mat(expected.size(), 1, CV_32S);
	for (int i = 0; i < data.size(); ++i) {
		expected_mat.at<int>(i, 0) = expected[i];
	}

	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::RBF);
	svm->setGamma(0.05);

	svm->train(data_mat, cv::ml::ROW_SAMPLE, expected_mat);
	svm->save(output_file);
	std::cerr << "train end" << std::endl;
}

void SlitherOCR::TrainOrientation(const char *input_file, const char *output_file)
{
	std::ifstream ifs(input_file);

	std::vector<std::vector<int> > data;
	std::vector<int> expected;

	for (;;) {
		std::string line;
		int ans;

		ifs >> ans;
		if (ans == -1) break;

		std::vector<int> pic;
		for (int i = 0; i < CLIP_SIZE; ++i) {
			ifs >> line;
			for (int j = 0; j < CLIP_SIZE; ++j) pic.push_back(line[j] == '#' ? 1 : 0);
		}

		for (int t = 0; t < 4; ++t) {
			data.push_back(pic);

			// rotate clockwise by 90 degree
			std::vector<int> pic2;
			for (int i = 0; i < CLIP_SIZE; ++i) {
				for (int j = 0; j < CLIP_SIZE; ++j) {
					int y = CLIP_SIZE - j;
					int x = i;
					pic2.push_back(pic[y * CLIP_SIZE + x]);
				}
			}

			pic.swap(pic2);

			if (ans != 0) {
				expected.push_back(t);
			} else {
				expected.push_back(4); // 0 can't be used for orientation detection
			}
		}
	}

	cv::Mat data_mat(data.size(), CLIP_SIZE * CLIP_SIZE, CV_32F);
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < CLIP_SIZE * CLIP_SIZE; ++j) {
			data_mat.at<float>(i, j) = data[i][j];
		}
	}
	cv::Mat expected_mat(expected.size(), 1, CV_32S);
	for (int i = 0; i < data.size(); ++i) {
		expected_mat.at<int>(i, 0) = expected[i];
	}

	svm_ori = cv::ml::SVM::create();
	svm_ori->setType(cv::ml::SVM::C_SVC);
	svm_ori->setKernel(cv::ml::SVM::RBF);
	svm_ori->setGamma(0.05);

	svm_ori->train(data_mat, cv::ml::ROW_SAMPLE, expected_mat);
	svm_ori->save(output_file);
	std::cerr << "ori train end" << std::endl;
}

void SlitherOCR::LoadTrainedData(const char *file)
{
	svm = cv::Algorithm::load<cv::ml::SVM>(file);
}

void SlitherOCR::LoadTrainedOrientationData(const char *file)
{
	svm_ori = cv::Algorithm::load<cv::ml::SVM>(file);
}

int SlitherOCR::Recognize(cv::Mat &pic)
{
	int n_dots = 0;
	for (int i = 0; i < CLIP_SIZE; ++i) {
		for (int j = 0; j < CLIP_SIZE; ++j) {
			if (pic.at<uchar>(i, j) == 0) ++n_dots;
		}
	}

	if (n_dots < 10) return -1;

	cv::Mat input(1, CLIP_SIZE * CLIP_SIZE, CV_32F);
	for (int i = 0; i < CLIP_SIZE; ++i) {
		for (int j = 0; j < CLIP_SIZE; ++j) {
			input.at<float>(0, i * CLIP_SIZE + j) = (pic.at<uchar>(i, j) == 0 ? 1 : 0);
		}
	}
	return svm->predict(input);
}

int SlitherOCR::RecognizeOrientation(cv::Mat &pic)
{
	int n_dots = 0;
	for (int i = 0; i < CLIP_SIZE; ++i) {
		for (int j = 0; j < CLIP_SIZE; ++j) {
			if (pic.at<uchar>(i, j) == 0) ++n_dots;
		}
	}

	if (n_dots < 10) return 4;

	cv::Mat input(1, CLIP_SIZE * CLIP_SIZE, CV_32F);
	for (int i = 0; i < CLIP_SIZE; ++i) {
		for (int j = 0; j < CLIP_SIZE; ++j) {
			input.at<float>(0, i * CLIP_SIZE + j) = (pic.at<uchar>(i, j) == 0 ? 1 : 0);
		}
	}
	return svm_ori->predict(input);
}

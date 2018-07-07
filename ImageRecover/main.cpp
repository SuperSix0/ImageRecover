#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <io.h>
#include <time.h>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>

#define s_max 7

using namespace std;
using namespace cv;
double ans[500][500];
class node
{
public:
	int  x, y, z;
	node(int xx, int yy, int zz) :x(xx), y(yy), z(zz) {}
};
double w(double s)
{
	if (s <= 0.5)
		return 2.0/3 - 4 * s * s + 4 * s * s * s;
	else if (s <= 1)
		return 4.0/3 - 4 * s + 4 * s * s - 4.0/3 * s * s * s;
	else
		return 0;
}
double MLS(vector<node>nodes)
{
	int n = nodes.size();
	Mat_<double>p(6, 1), A(6, 6), B(6, n), z(n, 1);
	Mat_<double>ans;
	double s;
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 6; j++)
			A[i][j] = 0;
	for (int i = 0; i < n; i++)
		z[i][0] = nodes[i].z;
	//cout << "z" << endl << z << endl;
	for (int k = 0; k < n; k++)
	{
		p[0][0] = 1;
		p[1][0] = nodes[k].x;
		p[2][0] = nodes[k].y;
		p[3][0] = p[1][0] * p[1][0];
		p[4][0] = p[1][0] * p[2][0];
		p[5][0] = p[2][0] * p[2][0];
		s = sqrt(nodes[k].x * nodes[k].x + nodes[k].y * nodes[k].y);
		A += w(s / s_max) * p * p.t();
		for (int i = 0; i < 6; i++)
			B[i][k] = w(s / s_max) * p[i][0];
	}
	ans = A.inv(DECOMP_SVD) * B * z;
	return ans[0][0];
}
uchar check(double d)
{
	if (d < 0)
		return 0;
	else if (d > 255)
		return 255;
	else
		return (uchar)d;
}
void A()
{
	Mat a = imread("A.png");
	vector<Mat> channels;
	split(a, channels);
	Mat B = channels.at(0);
	Mat G = channels.at(1);
	Mat R = channels.at(2);
	Mat A_MLS;
	A_MLS = B.clone();
	int cols = B.cols, rows = B.rows;
	int x, y;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (B.at<uchar>(i, j) == 0)
			{
				vector <node> nodes;
				for (int a = -s_max; a <= s_max; a++)
				{
					for (int b = -s_max; b <= s_max; b++)
					{
						x = a + i;
						y = b + j;
						if (x < 0 || x >= rows || y < 0 || y >= cols)
							continue;
						if (B.at<uchar>(x, y) != 0)
							nodes.push_back(node(a, b, B.at<uchar>(x, y)));
					}
				}
				double tmp = MLS(nodes);
				//cout << i << " " << j << " " << tmp << endl;
				A_MLS.at<uchar>(i, j) = check(tmp);
			}
		}
	}
	imwrite("3150104476_A.png",A_MLS);
}
void B(String filename)
{
	Mat a = imread(filename);
	vector<Mat> channels;
	split(a, channels);
	Mat B = channels.at(0);
	Mat G = channels.at(1);
	Mat R = channels.at(2);
	Mat B_R, B_G, B_B, B_MLS;
	int cols = B.cols, rows = B.rows;
	int x, y;
	double tmp;
	B_B = B.clone();
	B_R = R.clone();
	B_G = G.clone();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (B.at<uchar>(i, j) == 0)
			{
				vector <node> nodes;
				for (int a = -s_max; a <= s_max; a++)
				{
					for (int b = -s_max; b <= s_max; b++)
					{
						x = a + i;
						y = b + j;
						if (x < 0 || x >= rows || y < 0 || y >= cols)
							continue;
						if (B.at<uchar>(x, y) != 0)
							nodes.push_back(node(a, b, B.at<uchar>(x, y)));
					}
				}
				tmp = MLS(nodes);
				B_B.at<uchar>(i, j) = check(tmp);
			}
			if (R.at<uchar>(i, j) == 0)
			{
				vector <node> nodes;
				for (int a = -s_max; a <= s_max; a++)
				{
					for (int b = -s_max; b <= s_max; b++)
					{
						x = a + i;
						y = b + j;
						if (x < 0 || x >= rows || y < 0 || y >= cols)
							continue;
						if (R.at<uchar>(x, y) != 0)
							nodes.push_back(node(a, b, R.at<uchar>(x, y)));
					}
				}
				tmp = MLS(nodes);
				//cout << i << " " << j << " " << tmp << endl;
				B_R.at<uchar>(i, j) = check(tmp);
			}
			if (G.at<uchar>(i, j) == 0)
			{
				vector <node> nodes;
				for (int a = -s_max; a <= s_max; a++)
				{
					for (int b = -s_max; b <= s_max; b++)
					{
						x = a + i;
						y = b + j;
						if (x < 0 || x >= rows || y < 0 || y >= cols)
							continue;
						if (G.at<uchar>(x, y) != 0)
							nodes.push_back(node(a, b, G.at<uchar>(x, y)));
					}
				}
				tmp = MLS(nodes);
				//cout << i << " " << j << " " << tmp << endl;
				B_G.at<uchar>(i, j) = check(tmp);
			}
		}
	}
	vector<Mat> src;
	src.push_back(B_B);
	src.push_back(B_G);
	src.push_back(B_R);
	merge(src, B_MLS);
	imwrite("3150104476_" + filename, B_MLS);
}

int main()
{
	A();
	cout << "A finish" << endl;
	B("B.png");
	cout << "B finish" << endl;
	B("C.png");
	cout << "C finish" << endl;
	system("pause");
}
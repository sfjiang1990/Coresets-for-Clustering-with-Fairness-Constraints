// BICO.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <map>

#include "src/point/l2metric.h"
#include "src/point/squaredl2metric.h"
#include "src/point/point.h"
#include "src/point/pointweightmodifier.h"
#include "src/clustering/bico.h"
#include "src/misc/randomness.h"
#include "src/misc/randomgenerator.h"
#include "src/datastructure/proxysolution.h"
#include "src/point/pointcentroid.h"
#include "src/point/pointweightmodifier.h"
#include "src/point/realspaceprovider.h"

struct WeightedPoint
{
	std::vector<double> coor;
	int dim;
	double weight;

	WeightedPoint(std::vector<double> c, double w)
	{
		dim = c.size();
		weight = w;
		for (int i = 0; i < dim; i++)
		{
			coor.push_back(c[i]);
		}
	}
};

int N;
int dim;
int k;
int suggestedSize;
std::vector<WeightedPoint> X;

void input()
{
	using namespace std;
	X.clear();
	cin >> N >> dim >> suggestedSize >> k;
	for (int i = 0; i < N; i++)
	{
		vector<double> tmp;
		for (int j = 0; j < dim; j++)
		{
			double coor;
			cin >> coor;
			tmp.push_back(coor);
		}
		//int color;
		//cin >> color;
		X.push_back(WeightedPoint(tmp, 1.0));
	}
}

void output(std::vector<WeightedPoint> coreset)
{
	using namespace std;

	cout << coreset.size() << endl;
	for (int i = 0; i < coreset.size(); i++)
	{
		for (int j = 0; j < coreset[i].dim; j++)
		{
			cout << coreset[i].coor[j];
			if (j != coreset[i].dim)
				cout << " ";
			else
				cout << endl;
		}
		cout << coreset[i].weight << endl;
	}
}

void work()
{
	using namespace CluE;

	std::vector<WeightedPoint> coreset;

	input();
	int n = X.size();
	Bico<Point> bico(dim, n, k, 10, suggestedSize, new SquaredL2Metric(), new PointWeightModifier());
	for (int j = 0; j < n; j++)
	{
			Point p(X[j].coor);
			bico << p;
		}
		ProxySolution<Point>* sol = bico.compute();
		for (std::size_t i = 0; i < sol->proxysets[0].size(); i++)
		{
			double weight = sol->proxysets[0][i].getWeight();
			// Output center of gravity
			std::vector<double> coresetPoint;
			for (std::size_t j = 0; j < sol->proxysets[0][i].dimension(); ++j)
			{
				coresetPoint.push_back(sol->proxysets[0][i][j]);
			}
			coreset.push_back(WeightedPoint(coresetPoint, weight));
		}
	output(coreset);
}

int main(int argc, char **argv)
{
	using namespace CluE;

	work();

	return 0;
}

package coreset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import base.ColorGroup;
import base.Line;
import base.Point;
import base.Utility;
import base.WeightedDouble;
import base.WeightedPoint;
import clust.Clustering;
import clust.Objective;
import solver.PCA;

// implementing abstract methods for coreset construction of k-means objective
// here we use the Lloyd's heuristic on line clustering
public class CoresetKMeans extends Coreset
{
	public CoresetKMeans(List<WeightedPoint> instance) {
		super(instance);
		// Collections.shuffle(instance, Utility.rand);
	}
	
	
	@Override
	protected List<WeightedPoint> getCoresetPoint(Segment s) {
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();
		if (s.T.isEmpty())
			return res;
		if (s.T.size() <= 2)
		{
			res.addAll(s.toPoints());
			return res;
		}
		double mean = WeightedDouble.mean(s.T);
		Point meanPoint = s.getPoint(mean);
		Point pl = s.getPointByIndex(0).data;
		ColorGroup plCol = s.getPointByIndex(0).cg;
		Point pr = s.getPointByIndex(s.T.size() - 1).data;
		ColorGroup prCol = s.getPointByIndex(s.T.size() - 1).cg;
		double wl = 0, wr = 0;
		for (WeightedDouble wd : s.T)
		{
			if (wd.data <= mean)
			{
				double dist = Point.dist(meanPoint, pl);
				if (dist != 0)
				{
					wl += Point.dist(meanPoint, s.getPoint(wd.data))
						/ dist * wd.weight;
				}
			}
			else
			{
				double dist = Point.dist(meanPoint, pr);
				if (dist != 0)
				{
					wr += Point.dist(meanPoint, s.getPoint(wd.data))
							/ dist * wd.weight;
				}
			}
		}
		
		int tot = (int)Math.round(WeightedDouble.totalWeight(s.T));
		double factor = 0;
		if (wl + wr != 0)
		{
			factor = (tot + 0.0) / (wl + wr);
		}
		wl *= factor;
		wr *= factor;
		double start = 0, end = 1;
		ArrayList<WeightedPoint> sol = new ArrayList<WeightedPoint>();
		sol.add(new WeightedPoint(pl, wl, plCol));
		sol.add(new WeightedPoint(pr, wr, prCol));
		while (end - start > 1e-1)
		{
			double mid = (start + end) / 2.0;
			ArrayList<WeightedPoint> Ct = new ArrayList<WeightedPoint>();
			Ct.add(new WeightedPoint(pl.multiply(mid).plus(meanPoint.multiply(1-mid)), wl, plCol));
			Ct.add(new WeightedPoint(pr.multiply(mid).plus(meanPoint.multiply(1-mid)), wr, prCol));
			if (WeightedPoint.cumulError(Ct, this.getObjective()) >= WeightedPoint.cumulError(s.toPoints(), this.getObjective()))
			{
				sol = Ct;
				end = mid;
			}
			else
			{
				start = mid;
			}
		}
		double prob = sol.get(0).weight - (int)sol.get(0).weight;
		prob = 1.0 - prob;
		assert(prob >= 0);
		if (Utility.rand.nextDouble() <= prob)
		{
			sol.get(0).weight = Math.floor(sol.get(0).weight);
		}
		else
		{
			sol.get(0).weight = Math.ceil(sol.get(0).weight);
		}
		sol.get(1).weight = tot - sol.get(0).weight;
		return sol;
	}

	@Override
	protected Objective getObjective() {
		return Objective.getObjective(2.0);
	}

	@Override
	protected Coreset1D construct1D(List<WeightedPoint> l, Line line, List<Point> optCenters) {
		return new KMeans1D(l, line, optCenters);
	}
	
	private double lineMeansObjective(Map<Integer, List<WeightedPoint>> X, Line[] lines)
	{
		double sum = 0;
		int n = X.size();
		for (int i = 0; i < n; i++)
		{
			List<WeightedPoint> list = X.get(i);
			for (WeightedPoint wp : list)
			{
				sum += lines[i] == null ? 0 : Math.pow(Line.dist(wp.data, lines[i]), 2) * wp.weight;
			}
		}
		return sum;
	}
	
	private Line[] lineMeansCenters(Map<Integer, List<WeightedPoint>> clustering)
	{
		Line[] res = new Line[clustering.size()];
		for (int i = 0; i < clustering.size(); i++)
		{
			List<WeightedPoint> X = clustering.get(i);
			if (X.size() == 0)
			{
				res[i] = null;
				continue;
			}
			WeightedPoint mean = WeightedPoint.mean(X);
			ArrayList<Point> dir = new ArrayList<Point>();
			for (int j = 0; j < X.size(); j++)
			{
				dir.add(X.get(j).data.minus(mean.data));
			}
			res[i] = new Line(mean.data, PCA.pca(dir, 0));
		}
		return res;
	}
	
	private static Map<Integer, List<WeightedPoint>> initialClustering(List<WeightedPoint> X, int m)
	{
		int n = X.size();
		Map<Integer, List<WeightedPoint>> clustering = new HashMap<Integer, List<WeightedPoint>>();
		for (int i = 0, j = 0; i < m; i++)
		{
			ArrayList<WeightedPoint> cluster = new ArrayList<WeightedPoint>();
			if (i != m - 1)
			{
				for (int cnt = 0; cnt < n / m; cnt++, j++)
				{
					cluster.add(X.get(j));
				}
			}
			else
			{
				for (; j < n; j++)
				{
					cluster.add(X.get(j));
				}
			}
			clustering.put(i, cluster);
		}
		return clustering;
	}
	
	private Object[] localSearchProj(List<WeightedPoint> X, int m, double tol)
	{
		Map<Integer, List<WeightedPoint>> clustering = initialClustering(X, m);
		Line[] lineCenters = lineMeansCenters(clustering);
		double obj = lineMeansObjective(clustering, lineCenters);
		
		while (true)
		{
			Map<Integer, List<WeightedPoint>> nClustering = Clustering.getLineClustering(X, Arrays.asList(lineCenters));
			Line[] nLineCenters = lineMeansCenters(nClustering);
			double nObj = lineMeansObjective(nClustering, nLineCenters);
			
			System.out.printf("%.6f, %.6f\n", obj - nObj, Math.abs(obj - nObj) / obj);
			if (nObj > obj || Math.abs(obj - nObj) / obj < tol)
				return new Object[] {obj, lineCenters};
			obj = nObj;
			lineCenters = nLineCenters;
			clustering = nClustering;
		}
	}

	@Override
	protected List<Line> projectToLines(int k, double eps, List<Point> optCenters)
	{
		double opt = Clustering.evaluate(instance, optCenters, getObjective());
		double fac = 1.2;
		int m = 3;
		while (true)
		{
			Object[] tmp = localSearchProj(instance, m, 1e-1);
			double obj = (double)tmp[0];
			System.out.println(obj / opt);
			// t = obj / opt; t + sqrt(t) <= eps; x^2 + x - eps <= 0; x <= (-1 + sqrt(1 + 4 eps)) / 2
			// we are optimistic about the threshold; if strict error is required, then this needs to be changed
			double threshold = Math.pow((Math.sqrt(1 + 10 * eps) - 1) / 2, 2);
			if (obj / opt <= threshold)
			{
				return Arrays.asList((Line[])(tmp[1]));
			}
			m = (m == (int)(m * fac)) ? m + 1 : (int)(m * fac);
		}
	}
}

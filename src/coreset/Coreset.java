package coreset;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import base.Line;
import base.Point;
import base.WeightedPoint;
import clust.Clustering;
import clust.Objective;
import solver.ClusteringSolver;

// abstract class for the coreset construction
public abstract class Coreset
{
	List<WeightedPoint> instance;
	protected Coreset(List<WeightedPoint> instance)
	{
		this.instance = instance;
	}

	protected abstract List<WeightedPoint> getCoresetPoint(Segment s);
	protected abstract Objective getObjective();

	private List<WeightedPoint> getCoresetPoints(List<Segment> list)
	{
		List<WeightedPoint> res = new ArrayList<WeightedPoint>();
		for (Segment s : list)
		{
			List<Segment> segCol = s.breakByColor();
			for (Segment seg : segCol)
			{
				res.addAll(getCoresetPoint(seg));
			}
		}
		return res;
	}

	protected abstract Coreset1D construct1D(List<WeightedPoint> l, Line line, List<Point> optCenters);
	protected abstract List<Line> projectToLines(int k, double eps, List<Point> optCenters);
	private List<Point> getOptCenters(int k)
	{
		return ClusteringSolver.localSearch(instance, k, this.getObjective());
	}

	public List<WeightedPoint> getCoreset(double eps, int k)
	{
		List<WeightedPoint> res = new ArrayList<WeightedPoint>();
		long t0 = System.currentTimeMillis();
		List<Point> optCenters = this.getOptCenters(k);
		System.out.println("find opt " + (System.currentTimeMillis() - t0));

		t0 = System.currentTimeMillis();
		List<Line> lines = this.projectToLines(k, eps, optCenters);
		System.out.println("# lines: " + lines.size());
		System.out.println("lines " + (System.currentTimeMillis() - t0));
		
		t0 = System.currentTimeMillis();
		Map<Integer, List<WeightedPoint>> lineClustering = Clustering.getLineClustering(this.instance, lines);
		System.out.println("proj " + (System.currentTimeMillis() - t0));
		t0 = System.currentTimeMillis();
		List<Coreset1D> onedList = new ArrayList<Coreset1D>();
		for (int i = 0; i < lines.size(); i++)
		{
			List<WeightedPoint> clust = lineClustering.get(i);
			onedList.add(this.construct1D(clust, lines.get(i), optCenters));
		}
		System.out.println("1ds1 " + (System.currentTimeMillis() - t0));

		t0 = System.currentTimeMillis();
		for (Coreset1D oned : onedList)
		{
			if (oned.isEmpty())
				continue;
			long t1 = System.currentTimeMillis();
			List<Segment> o = oned.getSegments(eps, k);
			System.out.println("segment " + (System.currentTimeMillis() - t1));
			t1 = System.currentTimeMillis();
			res.addAll(this.getCoresetPoints(o));
			System.out.println("coreset point " + (System.currentTimeMillis() - t1));
		}
		System.out.println("1ds2 " + (System.currentTimeMillis() - t0));
		
		return res;
	}
}

package coreset;

import java.util.ArrayList;
import java.util.List;

import base.Line;
import base.Point;
import base.WeightedDouble;
import base.WeightedPoint;
import clust.Objective;

// abstract class for constructing coreset in the 1D case
public abstract class Coreset1D
{
	protected Segment data;
	protected List<Point> optCenters;
	protected Coreset1D(List<WeightedPoint> pointSet, Line line, List<Point> optCenters)
	{
		data = new Segment(pointSet, line);
		this.optCenters = optCenters;
	}
	
	public boolean isEmpty()
	{
		return data.T.isEmpty();
	}

	protected abstract double evaluateError(DynamicCumErr derr);
	private Segment generateSegment(double start, double end)
	{
		List<WeightedDouble> list = WeightedDouble.fracSubList(data.T, start, end);
		Segment res = new Segment(data.line);
		res.T.addAll(list);
		res.sort();
		return res;
	}
	protected abstract double getThreshold(double eps, int k, double opt);
	protected abstract Objective getObjective();
	
	private List<Segment> segmentation(int s, int t, double eps, int k, double opt)
	{
		double threshold = getThreshold(eps, k, opt);
		List<Segment> res = new ArrayList<Segment>();
		for (int i = s; i < t;)
		{
			int start = i, end = i;
			DynamicCumErr derr = new DynamicCumErr();
			for (;i < t && this.evaluateError(derr) <= threshold; i++)
			{
				derr.add(data.T.get(i));
			}
			end = i;
			res.add(generateSegment(start, end));
		}
		return res;
	}
	
	public Segment getSegment()
	{
		return this.data;
	}

	public List<Segment> getSegments(double eps, int k)
	{
		double hopt = 0;
		ArrayList<WeightedPoint> pnt = data.toPoints();
		for (WeightedPoint P : pnt)
		{
			hopt += P.weight * getObjective().value(Point.dist(P.data, optCenters));
		}
		return segmentation(0, data.T.size(), eps, k, hopt);
	}
}

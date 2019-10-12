package base;

import java.util.List;

import clust.Objective;

public class WeightedPoint extends ColoredWeighted<Point>
{
	public WeightedPoint(Point d, double w, ColorGroup cg)
	{
		super(d, w, cg);
	}
	
	public static WeightedPoint mean(List<WeightedPoint> X)
	{
		WeightedPoint res = new WeightedPoint(Point.uniform(X.get(0).data.dim, 0),
				totalWeight(X), X.get(0).cg);
		double sum = 0;
		for (WeightedPoint wp : X)
		{
			sum += wp.weight;
			res.data = res.data.plus(wp.data.multiply(wp.weight));
		}
		if (sum != 0)
			res.data = res.data.multiply( 1.0 / sum);
		return res;
	}
	
	public static double cumulError(List<WeightedPoint> X, Objective O)
	{
		Point mean = mean(X).data;
		double sum = 0;
		for (WeightedPoint wp : X)
		{
			sum += O.value(Point.dist(mean,  wp.data)) * wp.weight;
		}
		return sum;
	}
	
	@Override
	public int hashCode()
	{
		return this.data.hashCode() * cg.hashCode() * Double.hashCode(weight);
	}
	
	@Override
	public boolean equals(Object o)
	{
		WeightedPoint p = (WeightedPoint)o;
		return cg.equals(p.cg) && weight == p.weight && 
				p.data.equals(this.data);
	}

	@Override
	public Object copy() {
		return new WeightedPoint(this.data, this.weight, this.cg);
	}
}

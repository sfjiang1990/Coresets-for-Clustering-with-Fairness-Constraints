package base;

import java.util.ArrayList;

public class WeightedDouble extends ColoredWeighted<Double> implements Comparable<WeightedDouble>
{
	public WeightedDouble(double d, double w, ColorGroup cg)
	{
		super(d, w, cg);
	}

	@Override
	public int compareTo(WeightedDouble o)
	{
		return Double.compare(this.data, o.data);
	}
	
	public static double mean(ArrayList<WeightedDouble> T)
	{
		double sum = 0;
		double sumw = 0;
		for (WeightedDouble wd : T)
		{
			sum += wd.weight * wd.data;
			sumw += wd.weight;
		}
		return sum / sumw;
	}
	
	@Override
	protected Object copy() {
		return new WeightedDouble(this.data, this.weight, this.cg);
	}
}

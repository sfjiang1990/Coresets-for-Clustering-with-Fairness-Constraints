package coreset;

import java.util.ArrayList;
import java.util.List;

import base.WeightedDouble;

// this class is used to accelerate the repeated evaluation of cumulative error
public class DynamicCumErr
{
	private ArrayList<WeightedDouble> list = new ArrayList<WeightedDouble>();
	private int leMid;
	private List<WeightedDouble> weightedSum = new ArrayList<WeightedDouble>();
	private List<WeightedDouble> weightedSumSq = new ArrayList<WeightedDouble>();
	
	private WeightedDouble getMean()
	{
		WeightedDouble wd = weightedSum.get(weightedSum.size() - 1);
		return new WeightedDouble(wd.data * (
				wd.weight == 0 ? 0 : 1.0 / wd.weight),
				wd.weight, null);
	}
	
	public void clear()
	{
		list.clear();
		leMid = 0;
		weightedSum.clear();
		weightedSumSq.clear();
	}

	public void add(WeightedDouble p)
	{
		if (list.size() == 0)
		{
			weightedSum.add(new WeightedDouble(p.data * p.weight, p.weight, p.cg));
			weightedSumSq.add(new WeightedDouble(p.data * p.data * p.weight, p.weight, p.cg));
		}
		else
		{
			WeightedDouble pre = weightedSum.get(weightedSum.size() - 1);
			WeightedDouble preSq = weightedSumSq.get(weightedSumSq.size() - 1);
			weightedSum.add(new WeightedDouble(pre.data + p.data * p.weight, pre.weight + p.weight, null));
			weightedSumSq.add(new WeightedDouble(preSq.data + p.data * p.data * p.weight, preSq.weight + p.weight, null));
		}
		list.add(p);
		WeightedDouble mean = getMean();
		while (true)
		{
			if (leMid == list.size() - 1)
				break;
			if (list.get(leMid).data >= mean.data)
			{
				break;
			}
			else
			{
				leMid++;
			}
		}
		 
	}
	
	private WeightedDouble getSum(List<WeightedDouble> l, int start, int end)
	{
		if (start > end)
			return new WeightedDouble(0, 0, null);
		return new WeightedDouble(l.get(end).data - (start == 0 ? 0 : l.get(start - 1).data),
				l.get(end).weight - (start == 0 ? 0 : l.get(start-1).weight),
						null);
	}
	
	public double getError1()
	{
		if (list.size() == 0)
			return 0;
		WeightedDouble left = getSum(weightedSum, 0, leMid - 1);
		WeightedDouble right = getSum(weightedSum, leMid, weightedSum.size() - 1);
		WeightedDouble mean = getMean();
		return (mean.data * left.weight - left.data) + (right.data - mean.data * right.weight);
	}
	public double getError2()
	{
		if (list.size() == 0)
			return 0;
		WeightedDouble sumsq = getSum(weightedSumSq, 0, weightedSumSq.size() - 1);
		WeightedDouble sum = getSum(weightedSum, 0, weightedSum.size() - 1);
		WeightedDouble mean = getMean();
		return sumsq.data + mean.data * mean.data * mean.weight - 2 * mean.data * sum.data;
	}
}

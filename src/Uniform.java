import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import base.ColorGroup;
import base.Utility;
import base.WeightedPoint;

// the Uniform sampling baseline
public class Uniform
{
	private List<WeightedPoint> instance;
	
	public Uniform(List<WeightedPoint> instance)
	{
		this.instance = instance;
	}
	
	public List<WeightedPoint> getCoreset(int m)
	{
		List<WeightedPoint> res = new ArrayList<WeightedPoint>();
		double rho = (m + 0.0) / instance.size();
		Map<ColorGroup, List<WeightedPoint>> M = WeightedPoint.colorMap(instance);
		for (ColorGroup cg : M.keySet())
		{
			List<WeightedPoint> l = M.get(cg);
			int nSample = (int)((l.size() * rho));
			if (nSample == 0)
				nSample = 1;
			int weight = (int)((l.size() / (nSample + 0.0)));
			int slack = l.size() - weight * nSample;
			for (int i = 0; i < nSample; i++)
			{
				WeightedPoint p = l.get(Utility.rand.nextInt(l.size()));
				if (i == nSample - 1)
				{
					weight += slack;
				}
				res.add(new WeightedPoint(p.data, weight, cg));
			}
		}
		return res;
	}
}

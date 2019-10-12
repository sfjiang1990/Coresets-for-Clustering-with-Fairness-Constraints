package coreset;

import java.util.List;

import base.Line;
import base.Point;
import base.WeightedPoint;
import clust.Objective;

public class KMedian1D extends Coreset1D
{
    public KMedian1D(List<WeightedPoint> pointSet, Line line, List<Point> optCenters)
    {
        super(pointSet, line, optCenters);
    }

    @Override
    protected double getThreshold(double eps, int k, double opt)
    {
        return eps * opt / k;
    }

	@Override
	protected Objective getObjective()
	{
		return Objective.getObjective(1.0);
	}

	@Override
	protected double evaluateError(DynamicCumErr derr) {
		return derr.getError1();
	}
}

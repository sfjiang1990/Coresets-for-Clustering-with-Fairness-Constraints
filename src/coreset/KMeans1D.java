package coreset;

import java.util.List;

import base.Line;
import base.Point;
import base.WeightedPoint;
import clust.Objective;

public class KMeans1D extends Coreset1D
{

	protected KMeans1D(List<WeightedPoint> pointSet, Line line, List<Point> optCenters) {
		super(pointSet, line, optCenters);
	}

	@Override
	protected double getThreshold(double eps, int k, double opt)
	{
        return eps * eps * opt / k / k;
	}

	@Override
	protected Objective getObjective() {
		return Objective.getObjective(2.0);
	}

	@Override
	protected double evaluateError(DynamicCumErr derr) {
		return derr.getError2();
	}
}

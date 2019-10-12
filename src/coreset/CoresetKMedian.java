package coreset;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import base.Line;
import base.Point;
import base.Utility;
import base.WeightedPoint;
import clust.Clustering;
import clust.Objective;

// implementing abstract methods for constructing coreset of k-median
// boundedProj is the heuristic that we use to construct the set of projection lines;
// the idea is: searching for an exponentially growing threshold, and incrementally add random projection lines until the threshold or error bound is reached.
public final class CoresetKMedian extends Coreset {
    public CoresetKMedian(List<WeightedPoint> instance) {
        super(instance);
    }

    @Override
    protected ArrayList<WeightedPoint> getCoresetPoint(Segment s) {
        ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();
        if (s.T.isEmpty())
        	return res;
        WeightedPoint mean = WeightedPoint.mean(s.toPoints());
        mean.weight = Math.round(mean.weight);
        res.add(mean);
        return res;
    }

    @Override
    protected Coreset1D construct1D(List<WeightedPoint> l, Line line , List<Point> optCenters) {
        return new KMedian1D(l, line, optCenters);
    }

    @Override
    protected Objective getObjective() {
    	return Objective.getObjective(1.0);
    }
    
    private ArrayList<Point> boundedProj(List<Line> dir, List<Point> X, double eps, double obj)
    {
    	HashSet<Integer> dirRemain = new HashSet<Integer>();
    	for (int i = 0; i < dir.size(); i++)
    	{
    		dirRemain.add(i);
    	}
    	ArrayList<Point> res = new ArrayList<Point>();
    	double[] proj = new double[X.size()];
    	Arrays.fill(proj, 1e20);
    	int m = 1;
    	double fac = 1.2;
    	double threshold = 0.9;
    	while (dirRemain.size() > 0)
    	{
    		long t0 = System.currentTimeMillis();
    		int[] dirRemainArr = new int[dirRemain.size()];
    		int cnt = 0;
    		for (Integer t : dirRemain)
    		{
    			dirRemainArr[cnt++] = t;
    		}

    		List<Line> samples = new ArrayList<Line>();
    		
    		double sum = 0;
    		double old = 0;
   			for (int i = 0; i < X.size(); i++)
   			{
   				old += proj[i];
   			}
   			for (int i = 0; i < m; i++)
   			{
    				samples.add(dir.get(
    						dirRemainArr[Utility.rand.nextInt(dirRemain.size())]
    						));
   			}
   			for (int k = 0; k < X.size(); k++)
   			{
   				proj[k] = Math.min(proj[k], Line.dist(X.get(k), samples));
   			}
    		while (true)
    		{
    			List<Line> tmp = new ArrayList<Line>();
    			while (samples.size() < m)
    			{
    				Line l = dir.get(
    						dirRemainArr[Utility.rand.nextInt(dirRemain.size())]
    						);
    				tmp.add(l);
    				samples.add(l);
    			}
    			sum = 0;
    			for (int k = 0; k < X.size(); k++)
    			{
    				proj[k] = Math.min(proj[k], Line.dist(X.get(k), tmp));
    				sum += proj[k];
    			}
    			if (sum / obj <= eps)
    			{
    				break;
    			}
    			if (sum / old > threshold)
    			{
    				int nv = (int)(Math.round(m * fac));
    				m = nv == m ? m + 1 : nv;
    				//m++;
    			}
    			else
    			{
    				m = 1;
    				break;
    			}
    		}
    		
    		
    		for (Integer i : dirRemainArr)
    		{
    			// System.out.println(Point.dist(dir.get(i), minDir));
    			for (Line l : samples)
    			{
    				if (Point.dist(dir.get(i).direction, l.direction) <= eps)
    				{
    					dirRemain.remove(i);
    					break;
    				}
    			}
    		}
    		
    		for (Line l : samples)
    		{
    			res.add(l.direction);
    		}
    		System.out.println(sum / obj + " " + (System.currentTimeMillis() - t0));
    		if (sum / obj <= eps)
    			break;
    	}
    	return res;
    }

	private ArrayList<Point> projectForCluster(Point center, List<Point> X, double eps) {
		ArrayList<Line> dir = new ArrayList<Line>();
		for (int i = 0; i < X.size(); i++)
		{
			dir.add(new Line(center, X.get(i).minus(center).normalize()));
		}
    	double obj = 0;
    	for (int i = 0; i < X.size(); i++)
    	{
    		obj += Point.dist(X.get(i), center);
    	}
    	return this.boundedProj(dir, X, eps, obj);
	}
	
	@Override
	protected ArrayList<Line> projectToLines(int k, double eps, List<Point> optCenters)
	{
		ArrayList<Line> res = new ArrayList<Line>();

		List<Point> C = optCenters;
		List<WeightedPoint>[] clusters = Clustering.getClusters(this.instance, C);
		
		for (int i = 0; i < C.size(); i++)
		{
			ArrayList<Point> pnt = this.projectForCluster(C.get(i), WeightedPoint.flatten(clusters[i]), eps);
			for (Point p : pnt)
			{
				res.add(new Line(C.get(i), p));
			}
		}
		return res;
	}
}

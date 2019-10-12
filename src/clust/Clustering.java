package clust;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import base.Line;
import base.Point;
import base.WeightedPoint;
import ilog.concert.IloIntExpr;
import ilog.concert.IloIntVar;
import ilog.cplex.IloCplex;

public class Clustering
{
    private Clustering() {}
    public static double evaluate(List<WeightedPoint> X, List<Point> C, Objective O)
    {
        double sum = 0;
        for (WeightedPoint p : X)
        {
            sum += p.weight * O.value(Point.dist(p.data, C));
        }
        return sum;
    }
    
    // evaluate the (optimal) fair clustering value given F and C, via ILP (solved by CPLEX)
    // F[color][cluster]
    public static double evaluate(List<WeightedPoint> X, int[][] F, List<Point> C, Objective O)
    {
    	int numCol = F.length;
    	int k = C.size();
    	int n = X.size();
    	
    	double res = 0;
		try
		{
			IloCplex cplex = new IloCplex();
			// cplex.setOut(null);
			// cplex.setWarning(null);
			
			int[][] id = new int[n][k];
			int nVar = 0;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < k; j++)
				{
					id[i][j] = nVar++;
				}
			}
			int[] lb = new int[nVar];
			int[] ub = new int[nVar];
			
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < k; j++)
				{
					lb[id[i][j]] = 0;
					ub[id[i][j]] = (int)Math.round(X.get(i).weight);
				}
			}

			IloIntVar[] x = cplex.intVarArray(nVar, lb, ub);

			// build objective
			double[] objCoe = new double[nVar];
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < k; j++)
				{
					objCoe[id[i][j]] = O.value(Point.dist(X.get(i).data, C.get(j)));
				}
			}
			cplex.addMinimize(cplex.scalProd(x, objCoe));
			
			for (int i = 0; i < n; i++)
			{
				ArrayList<IloIntExpr> term = new ArrayList<IloIntExpr>();
				for (int j = 0; j < k; j++)
				{
					term.add(x[id[i][j]]);
				}
				IloIntExpr[] tmp = new IloIntExpr[term.size()];
				tmp = term.toArray(tmp);
				cplex.addEq(cplex.sum(tmp), (int)Math.round(X.get(i).weight));
			}
			
			for (int i = 0; i < numCol; i++)
			{
				for (int j = 0; j < k; j++)
				{
					ArrayList<IloIntExpr> term = new ArrayList<IloIntExpr>();
					for (int t = 0; t < n; t++)
					{
						for (Integer cc : X.get(t).cg.c)
						{
							if (cc == i)
							{
								term.add(x[id[t][j]]);
							}
						}
					}
					IloIntExpr[] tmp = new IloIntExpr[term.size()];
					tmp = term.toArray(tmp);
					cplex.addEq(cplex.sum(tmp), F[i][j]);
				}
			}
			// solve		
			cplex.solve();
			res = cplex.getObjValue();
			cplex.end();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		return res;
    }
    
	public static List<WeightedPoint>[] getClusters(List<WeightedPoint> X, List<Point> C)
	{
		int m = C.size();
		ArrayList<WeightedPoint>[] res = new ArrayList[m];
		for (int i = 0; i < m; i++)
		{
			res[i] = new ArrayList<WeightedPoint>();
		}
		for (WeightedPoint wp : X)
		{
			double min = Double.MAX_VALUE;
			int id = -1;
			for (int i = 0; i < m; i++)
			{
				double d = Point.dist(wp.data, C.get(i));
				if (d < min)
				{
					min = d;
					id = i;
				}
			}
			res[id].add(wp);
		}
		return res;
	}
	
	public static Map<Integer, List<WeightedPoint>> getLineClustering(List<WeightedPoint> X, List<Line> lineCenters)
	{
		int m = lineCenters.size();
		Map<Integer, List<WeightedPoint>> res = new HashMap<Integer, List<WeightedPoint>>();
		for (int i = 0; i < m; i++)
		{
			res.put(i, new ArrayList<WeightedPoint>());
		}
		for (WeightedPoint wp : X)
		{
			double min = Double.MAX_VALUE;
			int id = -1;
			for (int i = 0; i < m; i++)
			{
				if (lineCenters.get(i) == null)
					continue;
				double dist = Line.dist(wp.data, lineCenters.get(i));
				if (dist < min)
				{
					min = dist;
					id = i;
				}
			}
			res.get(id).add(wp);
		}
		return res;
	}
}

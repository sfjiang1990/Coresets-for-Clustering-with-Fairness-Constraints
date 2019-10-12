package solver;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import base.Point;
import base.WeightedPoint;
import clust.Clustering;
import clust.Objective;

public class ClusteringSolver
{
	// a native local search algorithm for finding approximate solution for k-median/means
    public static List<Point> localSearch(List<WeightedPoint> X, int k, Objective O)
    {
        HashSet<Integer> remain = new HashSet<Integer>();
        HashSet<Integer> sol = new HashSet<Integer>();

        for (int i = 0; i < X.size(); i++)
        {
        	remain.add(i);
        }
        for (int i = 0; i < k; i++)
        {
            sol.add(i);
            remain.remove(i);
        }

        ArrayList<Point> C = new ArrayList<Point>();
        for (Integer i : sol)
        {
        	C.add(X.get(i).data);
        }
        double current = Clustering.evaluate(X, C, O);
        int counter = 0;
        while (true)
        {
        	counter++;
        	// for performance, we terminate the search early
        	if (counter == 5)
        		return C;
            boolean flag = false;
            int del = 0, ins = 0;
            for (Integer i : remain)
            {
                if (flag) break;
                for (Integer j : sol)
                {
                    // swap: + i - j
                    C.clear();
                    C.add(X.get(i).data);
                    for (Integer t : sol)
                    {
                        if (!t.equals(j))
                        {
                            C.add(X.get(t).data);
                        }
                    }
                    double tmp = Clustering.evaluate(X, C, O);
                    if (tmp < current)
                    {
                        current = tmp;
                        flag = true;
                        ins = i;
                        del = j;
                        break;
                    }
                }
            }
            if (!flag) return C;
            sol.remove(del);
            sol.add(ins);
            remain.remove(ins);
            remain.add(del);
        }
    }
}

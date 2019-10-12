package base;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Point implements Serializable
{
    /**
	 * 
	 */
	private static final long serialVersionUID = -6900717486566469326L;
	public double[] coor;
    public int dim;

    private static Random rand = new Random(0);

    public Point(double[] coo)
    {
        this.dim = coo.length;
        coor = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            coor[i] = coo[i];
        }
    }

    public static Point uniform(int dim, double t)
    {
        double[] res = new double[dim];
        Arrays.fill(res, t);
        return new Point(res);
    }

    public static Point gaussian(int dim)
    {
        double[] res = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            res[i] = rand.nextGaussian();
        }
        return new Point(res).normalize();
    }

    private Point ope(Point other, double t, double s)
    {
        double[] res = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            res[i] = t * coor[i] + s * other.coor[i];
        }
        return new Point(res);
    }

    public Point plus(Point other)
    {
        return this.ope(other, 1, 1);
    }

    public Point minus(Point other)
    {
        return this.ope(other, 1, -1);
    }

    public Point multiply(double t)
    {
        return this.ope(uniform(dim, 0), t, 0);
    }

    public double dot(Point other)
    {
        double sum = 0;
        for (int i = 0; i < dim; i++)
        {
            sum += coor[i] * other.coor[i];
        }
        return sum;
    }

    public double norm()
    {
        return Math.sqrt(this.dot(this));
    }

    public Point normalize()
    {
        double n = this.norm();
        return n == 0 ? this : this.multiply(1.0 / n);
    }

    public static double dist(Point p, Point q)
    {
        return p.minus(q).norm();
    }

    public static double dist(Point p, List<Point> S)
    {
        double m = Double.MAX_VALUE;
        for (Point s : S)
        {
            m = Math.min(m, Point.dist(p, s));
        }
        return m;
    }

    public static Point mean(List<Point> X)
    {
    	if (X.size() == 0)
    		return null;
    	Point res = uniform(X.get(0).dim, 0);
    	for (Point p : X)
    	{
    		res = res.plus(p);
    	}
    	return res.multiply(1.0 / X.size());
    }
    
    @Override
    public int hashCode()
    {
    	int res = 0;
    	for (int i = 0; i < dim; i++)
    	{
    		res *= Double.hashCode(coor[i]);
    	}
    	return res;
    }
    
    @Override
    public boolean equals(Object o)
    {
    	Point p = (Point)o;
    	if (dim != p.dim)
    		return false;
    	for (int i = 0; i < dim; i++)
    	{
    		if (coor[i] != p.coor[i])
    			return false;
    	}
    	return true;
    }
}

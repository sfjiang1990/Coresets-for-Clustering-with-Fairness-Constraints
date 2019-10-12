package base;

import java.util.List;

public class Line
{
	public Point center;
	public Point direction;
	
	public Line(Point c, Point d)
	{
		this.center = c;
		this.direction = d.normalize();
	}
	
	private static double fixSqrt(double t)
    {
        if (t < 0) return 0;
        return Math.sqrt(t);
    }
	
	public static double dist(Point p, Line r)
	{
		Point Vcp = p.minus(r.center);
        double proj = Vcp.dot(r.direction);
        return fixSqrt(Math.pow(Vcp.norm(), 2) - proj * proj);
	}
	
	public static double dist(Point p, List<Line> lines)
	{
		double min = Double.MAX_VALUE;
		for (Line l : lines)
		{
			if (l == null)
				continue;
			min = Math.min(min, dist(p, l));
		}
		return min;
	}
}

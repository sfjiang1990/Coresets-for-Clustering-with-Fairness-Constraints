package coreset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import base.ColorGroup;
import base.Line;
import base.Point;
import base.WeightedDouble;
import base.WeightedPoint;

// a segment represents a bucket in the line case
public class Segment
{
	public Line line;
	
	public ArrayList<WeightedDouble> T = new ArrayList<WeightedDouble>();
	
	public Segment(List<WeightedPoint> X, Line line)
	{
		this.line = line;
		
		for (WeightedPoint p : X)
		{
			double t = p.data.minus(line.center).dot(line.direction);
			// double t = p.data.minus(center).norm();
			T.add(new WeightedDouble(t, p.weight, p.cg));
		}
		Collections.sort(T);
	}
	
	public List<Segment> breakByColor()
	{
		List<Segment> res = new ArrayList<Segment>();
		HashMap<ColorGroup, List<WeightedDouble>> colMap = WeightedDouble.colorMap(T);
		for (ColorGroup cg : colMap.keySet())
		{
			Segment ns = new Segment(this.line);
			ns.T.addAll(colMap.get(cg));
			sort();
			res.add(ns);
		}
		return res;
	}
	
	public Segment(Line line)
	{
		this.line = line;
	}
	
	public void sort()
	{
		Collections.sort(T);
	}
	
	public Point getPoint(double t)
	{
		return line.center.plus(line.direction.multiply(t));
	}
	
	public WeightedPoint getPointByIndex(int i)
	{
		return new WeightedPoint(getPoint(T.get(i).data), T.get(i).weight, T.get(i).cg);
	}
	
	public ArrayList<WeightedPoint> toPoints()
	{
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();
		for (WeightedDouble t : T)
		{
			res.add(new WeightedPoint(getPoint(t.data), t.weight, t.cg));
		}
		return res;
	}
	
	public double totalWeight()
	{
		return WeightedDouble.totalWeight(T);
	}
}

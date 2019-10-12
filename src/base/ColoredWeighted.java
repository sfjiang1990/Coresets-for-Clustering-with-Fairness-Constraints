package base;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class ColoredWeighted<T>
{
	public double weight;
	public T data;
	public ColorGroup cg;
	
	protected abstract Object copy();

	protected ColoredWeighted(T data, double w, ColorGroup cg)
	{
		this.data = data;
		this.weight = w;
		this.cg = cg;
	}
	
	public static <T, V extends ColoredWeighted<T>> List<T> flatten(List<V> X)
	{
		ArrayList<T> res = new ArrayList<T>();
		for (V x : X)
		{
			res.add(x.data);
		}
		return res;
	}
	
	public static double totalWeight(List<? extends ColoredWeighted> X)
	{
		double sum = 0;
		for (ColoredWeighted p : X)
		{
			sum += p.weight;
		}
		return sum;
	}
	
	public static <T extends ColoredWeighted> HashMap<ColorGroup, List<T>> colorMap(List<T> X)
	{
		HashMap<ColorGroup, List<T>> res = new HashMap<ColorGroup, List<T>>();

		for (T cwp : X) 
		{
			if (!res.containsKey(cwp.cg))
			{
				res.put(cwp.cg, new ArrayList<T>());
			}
			res.get(cwp.cg).add(cwp);
		}
		
		return res;
	}
	
	public static <T extends ColoredWeighted> List<T> fracSubList(List<T> X, double start, double end)
	{
		ArrayList<T> res = new ArrayList<T>();
		int startCeil = (int)Math.ceil(start);
		int startFloor = (int)Math.floor(start);
		int endFloor = (int)Math.floor(end);
		
		for (int i = startCeil; i < endFloor; i++)
		{
			res.add((T)X.get(i).copy());
		}
		
		if (startCeil != start)
		{
			T tmp = (T)X.get(startFloor).copy();
			tmp.weight *= startCeil - start;
			res.add(tmp);
		}
		if (end != endFloor)
		{
			T tmp = (T)X.get(endFloor).copy();
			tmp.weight *= end - endFloor;
			res.add(tmp);
		}
		return res;
	}
}

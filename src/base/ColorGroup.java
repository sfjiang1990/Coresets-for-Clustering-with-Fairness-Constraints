package base;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ColorGroup
{
	public Set<Integer> c = new HashSet<Integer>();
	public ColorGroup(int[] col)
	{
		for (int i : col)
		{
			c.add(i);
		}
	}
	
	public List<Integer> toList()
	{
		List<Integer> res = new ArrayList<Integer>();
		for (Integer i : c)
		{
			res.add(i);
		}
		return res;
	}
	
	@Override
	public int hashCode()
	{
		int res = 1;
		for (Integer cc : c)
		{
			res *= cc.hashCode();
		}
		return res;
	}

	@Override
	public boolean equals(Object o)
	{
		ColorGroup cg = (ColorGroup)o;
		if (cg.c.size() != this.c.size())
			return false;
		for (Integer cc : cg.c)
		{
			if (!this.c.contains(cc))
				return false;
		}
		return true;
	}
}

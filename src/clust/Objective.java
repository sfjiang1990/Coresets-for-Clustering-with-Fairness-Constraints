package clust;

import java.util.HashMap;

public class Objective
{
	private double z;
	private Objective(double z) {this.z = z;}
	public double value(double v) {
		return Math.pow(Math.abs(v), z);
	}
	private static HashMap<Double, Objective> M = new HashMap<Double, Objective>();
	public static Objective getObjective(double z)
	{
		if (!M.containsKey(z))
		{
			M.put(z, new Objective(z));
		}
		return M.get(z);
	}
}

import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import base.ColorGroup;
import base.Point;
import base.WeightedPoint;

public class BICO
{
	private ArrayList<WeightedPoint> instance;
	private int dim;
	public static final String bico = "bico/bico.exe";
	public BICO(ArrayList<WeightedPoint> instance)
	{
		this.instance = instance;
		this.dim = instance.get(0).data.dim;
	}
	
	private static void feedInput(List<WeightedPoint> instance, OutputStream os, int suggestedSize, int k)
	{
		int dim = instance.get(0).data.dim;
		PrintWriter out = new PrintWriter(new BufferedOutputStream(os));
		out.printf("%d %d %d %d\n", instance.size(), dim, suggestedSize, k);
		for (int i = 0; i < instance.size(); i++)
		{
			WeightedPoint wp = instance.get(i);
			for (int j = 0; j < wp.data.dim; j++)
			{
				out.printf("%.8f ", wp.data.coor[j]);
			}
			// out.println(wp.color);
		}
		out.flush();
	}
	
	private static ArrayList<WeightedPoint> getOutput(InputStream is, int dim, ColorGroup cg)
	{
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();

		Scanner scan = new Scanner(is);
		/*while (scan.hasNext())
		{
			System.out.println(scan.nextLine());
		}*/
		int coresetSize = scan.nextInt();
		for (int i = 0; i < coresetSize; i++)
		{
			double[] coor = new double[dim];
			for (int j = 0; j < dim; j++)
			{
				coor[j] = scan.nextDouble();
			}
			double weight = scan.nextDouble();
			res.add(new WeightedPoint(new Point(coor), weight, cg));
		}
		scan.close();
		return res;
	}
	
	public ArrayList<WeightedPoint> getCoreset(int k, double eps, int suggestedSize)
	{
		Process process = null;
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();

		Map<ColorGroup, List<WeightedPoint>> colMap = WeightedPoint.colorMap(instance);
		for (ColorGroup cg : colMap.keySet())
		{
			try
			{
				int dim = colMap.get(cg).get(0).data.dim;
				process = new ProcessBuilder(bico).start();
				feedInput(colMap.get(cg), process.getOutputStream(), suggestedSize, k);
				res.addAll(getOutput(process.getInputStream(), dim, cg));
				process.waitFor();
			}
			catch (Exception e)
			{
				e.printStackTrace();
			}
		}
		return res;
	}
}

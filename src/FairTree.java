import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.List;
import java.util.Scanner;

import base.WeightedPoint;

// wrapper for FairTree python implementation
public class FairTree
{
	private List<WeightedPoint> instance;
	public static final String fair = "fairtree/fairtree.py";
	public static final String fairwt = "fairtree/fairtree_wt.py";
	
	public FairTree(List<WeightedPoint> instance)
	{
		this.instance = instance;
	}
	
	private static void feedInput(List<WeightedPoint> instance, OutputStream os, boolean weighted)
	{
		PrintWriter out = new PrintWriter(new BufferedOutputStream(os));
		for (int i = 0; i < instance.size(); i++)
		{
			WeightedPoint wp = instance.get(i);
			if (weighted)
			{
				{
					out.printf("%d,%d,", wp.cg.toList().get(0) == 0 ? 0 : 1, (int)Math.round(wp.weight));
					for (int j = 0; j < wp.data.dim; j++)
					{
						out.printf("%.8f", wp.data.coor[j]);
						if (j != wp.data.dim - 1)
							out.print(',');
						else
							out.println();
					}
				}
			}
			else
			{
				for (int cnt = 0; cnt < (int)Math.round(wp.weight); cnt++)
				{
					out.printf("%d,", wp.cg.toList().get(0) == 0 ? 0 : 1);
					for (int j = 0; j < wp.data.dim; j++)
					{
						out.printf("%.8f", wp.data.coor[j]);
						if (j != wp.data.dim - 1)
							out.print(',');
						else
							out.println();
					}
				}
			}
		}
		out.flush();
	}
	
	public Object[] getResult(int p, int q, int k, boolean weighted, String program)
	{
		Process process = null;
		double obj = 0;
		double fairletTime = 0;
		double clustTime = 0;
		int node = 0;
		try
		{
			File tmp = File.createTempFile("aaa", "bbb");
			tmp.deleteOnExit();
			feedInput(instance, new FileOutputStream(tmp), weighted);
			System.out.println(tmp.getAbsolutePath());
			process = new ProcessBuilder("python", program, "" + p, "" + q, "" + k, tmp.getAbsolutePath()).start();
			Scanner scan = new Scanner(process.getInputStream());
			while (scan.hasNext())
			{
				String line = scan.nextLine();
				if (line.startsWith("Number of nodes in"))
				{
					node = Integer.parseInt(line.split(":")[1].trim());
				}
				if (line.startsWith("k-Median"))
				{
					Scanner lineScanner = new Scanner(line);
					lineScanner.next();
					lineScanner.next();
					obj = lineScanner.nextDouble();
					line = scan.nextLine();
					lineScanner = new Scanner(line);
					lineScanner.next();
					lineScanner.next();
					fairletTime = lineScanner.nextDouble();
					line = scan.nextLine();
					lineScanner = new Scanner(line);
					lineScanner.next();
					lineScanner.next();
					clustTime = lineScanner.nextDouble();
					break;
				}
			}
			scan.close();
			process.waitFor();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		return new Object[] {obj, fairletTime, clustTime, node};
	}
}

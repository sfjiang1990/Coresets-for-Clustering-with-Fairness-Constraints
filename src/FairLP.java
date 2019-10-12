import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.List;
import java.util.Scanner;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import base.WeightedPoint;

// wrapper for FairLP python implementation
public class FairLP
{
	private List<WeightedPoint> instance;
	public static final String fair = "fairlp/unweighted/example.py";
	public static final String fair_wt = "fairlp/weighted/example.py";
	private DataDescriptor dd;
	private static final String data = "fairlp/data/data.csv";
	private static final String config = "fairlp/config/config.ini";
	private static final String dataConfig = "fairlp/config/data_config.ini";
	
	public FairLP(List<WeightedPoint> instance, DataDescriptor dd)
	{
		this.instance = instance;
		this.dd = dd;
	}
	
	private void createData(boolean weighted) throws Exception
	{
		FileOutputStream os = new FileOutputStream(new File(data));
		PrintWriter out = new PrintWriter(new BufferedOutputStream(os));
		for (int i = 0; i < dd.dataHeader.length; i++)
		{
			out.print(dd.dataHeader[i]+",");
		}
		for (int i = 0; i < dd.sensHeader.length; i++)
		{
			out.print(dd.sensHeader[i]+",");
		}
		if (weighted)
			out.println("multiplicity");
		else
			out.println();
		for (int i = 0; i < instance.size(); i++)
		{
			WeightedPoint wp = instance.get(i);
			{
				for (int j = 0; j < wp.data.dim; j++)
				{
					out.printf("%.8f,", wp.data.coor[j]);
				}
				for (Integer color : wp.cg.c)
				{
					out.printf("%d,", color);
					//int id = colorIdMap.get(color);
					//output[wp.data.dim+id] = Integer.toString(color);
				}
				if (weighted)
					out.println((int)Math.round(wp.weight));
				else
					out.println();
			}
		}
		out.close();
	}
	
	private void createConfig() throws Exception
	{
		PrintWriter out = null;
		out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File(config))));
		out.println("[DEFAULT]");
		out.println("config_file = fairlp/config/data_config.ini");
		out.println("violating = False");
		out.println();
		out.println("[main]");
		out.println("data_dir = fairlp/output/");
		out.println("dataset = main");
		out.println("num_clusters = 3");
		out.println("deltas = 0.2");
		out.printf("max_points = %d", (int)(Math.round(WeightedPoint.totalWeight(instance))) + 1000);
		out.println();
		out.close();

		out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File(dataConfig))));
		out.println("[DEFAULT]");
		out.println("scaling = false");
		out.println("clustering_method = kmeans");
		out.println();
		out.println("[main]");
		out.println("csv_file = fairlp/data/data.csv");
		out.println("separator = ,");
		out.print("columns = ");
		for (int i = 0; i < dd.dataHeader.length; i++)
		{
			out.print(dd.dataHeader[i]);
			if (i == dd.dataHeader.length - 1)
				out.println();
			else
				out.print(",");
		}
		out.print("text_columns = ");
		for (int i = 0; i < dd.sensHeader.length; i++)
		{
			out.print(dd.sensHeader[i]);
			if (i == dd.sensHeader.length - 1)
				out.println();
			else
				out.print(",");
		}
		out.print("variable_of_interest = ");
		for (int i = 0; i < dd.sensHeader.length; i++)
		{
			out.print(dd.sensHeader[i]);
			if (i == dd.sensHeader.length - 1)
				out.println();
			else
				out.print(",");
		}
		out.print("fairness_variable = ");
		for (int i = 0; i < dd.sensHeader.length; i++)
		{
			out.print(dd.sensHeader[i]);
			if (i == dd.sensHeader.length - 1)
				out.println();
			else
				out.print(",");
		}
		for (int i = 0; i < dd.sensHeader.length; i++)
		{
			out.print(dd.sensHeader[i] + "_conditions = ");
			int m = dd.sensMap[i].size();
			for (int j = 0; j < m; j++)
			{
				out.printf("lambda x : x == %d", j);
				if (j == m - 1)
					out.println();
				else
					out.print(", ");
			}
		}
		out.close();
	}
	
	public Object[] getResult(boolean weighted) throws Exception
	{
		String program = weighted ? fair_wt : fair;
		Process process = null;
		double obj = 0;
		double time = 0;
		try
		{
			this.createData(weighted);
			this.createConfig();
			ProcessBuilder pb = new ProcessBuilder("py", program, "main");
			pb.redirectErrorStream(true);
			process = pb.start();
			Scanner scan = new Scanner(process.getInputStream());
			while (scan.hasNext())
			{
				System.out.println(scan.nextLine());
			}
			scan.close();
			process.waitFor();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		JSONParser parser = new JSONParser();
		Reader reader = new FileReader(new File("fairlp/output/main"));
		JSONObject jobj = (JSONObject)parser.parse(reader);
		obj = (Double)jobj.get("fair_score");
		time = (Double)jobj.get("total_time");

		return new Object[] {obj, time};
	}
}

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import base.ColorGroup;
import base.Point;
import base.WeightedPoint;

public class DataUtil
{
	private DataUtil() {}
	public static ArrayList<WeightedPoint> getData(DataDescriptor dd)
			throws Exception
	{
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();
		GZIPInputStream instream = new GZIPInputStream(new FileInputStream(new File(dd.path)));
		CSVParser csv = dd.format.withFirstRecordAsHeader().parse(new InputStreamReader(instream));
		Map<String, Integer> MM = csv.getHeaderMap();
		for (String s : MM.keySet())
		{
			System.out.print(s + ", " + MM.get(s));
		}
		System.out.println();
		for (CSVRecord r : csv)
		{
			boolean flag = false;
			double[] p = new double[dd.dataHeader.length];
			int cnt = 0;
			for (String h : dd.dataHeader)
			{
				try
				{
					p[cnt] = Double.parseDouble(r.get(h));
				}
				catch(NumberFormatException e)
				{
					flag = true;
				}
				cnt++;
			}
			int[] cg = new int[dd.sensHeader.length];
			cnt = 0;
			for (String s : dd.sensHeader)
			{
				if (!dd.sensMap[cnt].containsKey(r.get(s).trim()))
				{
					flag = true;
					break;
				}
				cg[cnt] = dd.sensMap[cnt].get(r.get(s).trim());
				cnt++;
			}
			if (!flag)
				res.add(new WeightedPoint(new Point(p), 1, new ColorGroup(cg)));
		}
		csv.close();
		return res;
	}
}

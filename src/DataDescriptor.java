import java.util.HashMap;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;

public class DataDescriptor
{
	private DataDescriptor(String path, String[] dataHeader, String[] sensHeader, Map<String, Integer>[] sensMap, CSVFormat format)
	{
		this.path = path;
		this.dataHeader = dataHeader;
		this.sensHeader = sensHeader;
		this.sensMap = sensMap;
		this.format = format;
	}
	public String path;
	public String[] dataHeader;
	public String[] sensHeader;
	public Map<String, Integer>[] sensMap;
	public CSVFormat format;
	
	private static Map<String, Integer>[] toMap(String[][] sensGroups, int[][] label)
	{
		Map<String, Integer>[] sensMap = new HashMap[sensGroups.length];
		if (label == null)
		{
			label = new int[sensGroups.length][];
			int cnt = 0;
			for (int i = 0; i < sensGroups.length; i++)
			{
				label[i] = new int[sensGroups[i].length];
				for (int j = 0; j < sensGroups[i].length; j++)
				{
					label[i][j] = cnt++;
				}
			}
		}
		for (int i = 0; i < sensGroups.length; i++)
		{
			sensMap[i] = new HashMap<String, Integer>();
			for (int j = 0; j < sensGroups[i].length; j++)
			{
				sensMap[i].put(sensGroups[i][j], label[i][j]);
			}
		}
		return sensMap;
	}
	
	public static DataDescriptor adult()
	{
		return new DataDescriptor("data/adult/data.csv.gz", new String[] {
				"age", "fnlwgt", "education-num",
				"capital-gain", "capital-loss", "hours-per-week"
		}, new String[] {
				"sex", "marital-status"
		}, toMap(new String[][] {
			{"Male", "Female"},
			{"Married-civ-spouse", "Divorced",
				"Never-married", "Separated",
				"Widowed", "Married-spouse-absent", "Married-AF-spouse"}}, null),
				CSVFormat.RFC4180);
	}
	public static DataDescriptor adultBinary()
	{
		return new DataDescriptor("data/adult/data.csv.gz", new String[] {
				"age", "fnlwgt", "education-num",
				"capital-gain", "capital-loss", "hours-per-week"
		}, new String[] {
				"sex"
		}, toMap(new String[][] {
			{"Male", "Female"}
		}, null), CSVFormat.RFC4180);
	}
	public static DataDescriptor bank()
	{
		return new DataDescriptor("data/bank/data.csv.gz", new String[] {
				"age", "duration", "campaign",
				"pdays", "previous", "emp.var.rate",
				"cons.price.idx", "cons.conf.idx", "euribor3m",
				"nr.employed"
		}, new String[] {
				"marital", "default"
		}, toMap(new String[][] {
			{"divorced", "married", "single", "unknown"},
			{"no", "yes", "unknown"}}, new int[][] {
				{0, 1, 2, 3},
				{4, 5, 6}
			}), CSVFormat.newFormat(';').withQuote('\"'));
	}
	public static DataDescriptor bankBinary()
	{
		return new DataDescriptor("data/bank/data.csv.gz", new String[] {
				"age", "duration", "campaign",
				"pdays", "previous", "emp.var.rate",
				"cons.price.idx", "cons.conf.idx", "euribor3m",
				"nr.employed"
		}, new String[] {
				"marital"
		}, toMap(new String[][] {
			{"divorced", "married", "single", "unknown"},
		}, new int[][] {
			{0, 1, 0, 0}
		}), CSVFormat.newFormat(';').withQuote('\"'));
	}
	public static DataDescriptor diabetes()
	{
		return new DataDescriptor("data/diabetes/data.csv.gz", new String[] {
			"time_in_hospital", "num_lab_procedures", "num_procedures",
			"num_medications", "number_outpatient", "number_emergency",
			"number_inpatient", "number_diagnoses"
		}, new String[] {
				"gender", "age"
		}, toMap(new String[][] {
			{"Male", "Female", "Unknown/Invalid"},
			{"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"}
		}, new int[][] {
				{0, 1, 1},
				{2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
		}), CSVFormat.RFC4180);
	}
	public static DataDescriptor diabetesBinary()
	{
		return new DataDescriptor("data/diabetes/data.csv.gz", new String[] {
			"time_in_hospital", "num_lab_procedures", "num_procedures",
			"num_medications", "number_outpatient", "number_emergency",
			"number_inpatient", "number_diagnoses"
		}, new String[] {
				"gender",
		}, toMap(new String[][] {
			{"Male", "Female", "Unknown/Invalid"},
		}, new int[][] {
				{0, 1, 1},
		}), CSVFormat.RFC4180);
	}
	public static DataDescriptor athlete()
	{
		return new DataDescriptor("data/athlete/data.csv.gz", new String[] {
				"Weight", "Age", "Height"
		}, new String[] {
				"Sex"
		}, toMap(new String[][] {
			{"M", "F"}
		}, new int[][] {
			{0, 1}
		}), CSVFormat.RFC4180);
	}
	public static DataDescriptor census1990()
	{
		return new DataDescriptor("data/census1990/data.csv.gz", new String[] {
				"dAncstry1", "dAncstry2", "iAvail",
				"iCitizen", "iClass", "dDepart", "iFertil",
				"iDisabl1", "iDisabl2", "iEnglish",
				"iFeb55", "dHispanic", "dHour89"
		}, new String[] {
				"iSex", "iMarital"
		}, toMap(new String[][] {
			{"0", "1"},
			{"0", "1", "2", "3", "4"}
		}, null), CSVFormat.RFC4180);
	}
	public static DataDescriptor census1990Binary()
	{
		return new DataDescriptor("data/census1990/data.csv.gz", new String[] {
				"dAncstry1", "dAncstry2", "iAvail",
				"iCitizen", "iClass", "dDepart", "iFertil",
				"iDisabl1", "iDisabl2", "iEnglish",
				"iFeb55", "dHispanic", "dHour89"
		}, new String[] {
				"iSex"
		}, toMap(new String[][] {
			{"0", "1"},
		}, null), CSVFormat.RFC4180);
	}
}

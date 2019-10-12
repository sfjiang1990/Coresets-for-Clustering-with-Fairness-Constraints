import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import base.ColorGroup;
import base.Point;
import base.Utility;
import base.WeightedPoint;
import clust.Clustering;
import clust.Objective;
import coreset.CoresetKMeans;
import coreset.CoresetKMedian;

public class Main
{
	// this debug switch could take three values: gen, true, false.
	// the purpose of this is to cache the objective values on the original data set
	String debug = "false";
	private ArrayList<WeightedPoint> downSample(ArrayList<WeightedPoint> l, double prob)
	{
		ArrayList<WeightedPoint> res = new ArrayList<WeightedPoint>();
		Random rand = new Random(0);
		for (WeightedPoint p : l)
		{
			if (rand.nextDouble() <= prob)
			{
				res.add(p);
			}
		}
		return res;
	}
	
	// normalize the data coordinates to [0, 1] by dividing the l_\infty norm
	public static void normalize(ArrayList<WeightedPoint> instance)
	{
		int dim = instance.get(0).data.dim;
		for (int i = 0; i < dim; i++)
		{
			double max = 0;
			for (WeightedPoint wp : instance)
			{
				max = Math.max(max, Math.abs(wp.data.coor[i]));
			}
			if (max != 0)
			{
				for (WeightedPoint wp : instance)
				{
					wp.data.coor[i] /= max;
				}
			}
		}
	}
	
	private static Map<String, Integer> nameToInt(String[] col)
	{
		HashMap<String, Integer> res = new HashMap<String, Integer>();
		int cnt = 0;
		for (String s : col)
		{
			res.put(s.toLowerCase(), cnt++);
		}
		return res;
	}

	// the following methods is used to fetch the dataset, according to DataDescriptor.
	public ArrayList<WeightedPoint> adultBinary() throws Exception
	{
		return DataUtil.getData(DataDescriptor.adultBinary());
	}
	
	public ArrayList<WeightedPoint> bankBinary() throws Exception
	{
		return DataUtil.getData(DataDescriptor.bankBinary());
	}
	public ArrayList<WeightedPoint> adult() throws Exception
	{
		return DataUtil.getData(DataDescriptor.adult());
	}
	
	public ArrayList<WeightedPoint> bank() throws Exception
	{
		return DataUtil.getData(DataDescriptor.bank());
	}
	
	
	public ArrayList<WeightedPoint> diabetesBinary() throws Exception
	{
		return DataUtil.getData(DataDescriptor.diabetesBinary());
	}
	
	public ArrayList<WeightedPoint> diabetes() throws Exception
	{
		return DataUtil.getData(DataDescriptor.diabetes());
	}
	
	public ArrayList<WeightedPoint> census1990() throws Exception
	{
		return DataUtil.getData(DataDescriptor.census1990());
	}
	

	public ArrayList<WeightedPoint> census1990Binary() throws Exception
	{
		return DataUtil.getData(DataDescriptor.census1990Binary());
	}
	
	public ArrayList<WeightedPoint> athlete() throws Exception
	{
		return DataUtil.getData(DataDescriptor.athlete());
	}

	// evaluate the maximum error of the coreset
	double evaluateError(ArrayList<WeightedPoint> instance, ArrayList<WeightedPoint> coreset, int k, int cases, Objective O)
	{
		double err = 0;
		for (int t = 0; t < cases; t++)
		{
			ArrayList<Point> C = this.getCenter(instance, k);
			double obj = Clustering.evaluate(instance, C, O);
			double cor = Clustering.evaluate(coreset, C, O);
			err = Math.max(err, Math.abs(cor - obj) / obj);
		}
		return err;
	}
	
	private int[] randPart(int n, int k)
	{
		int[] res = new int[k];
		if (k == 1)
		{
			res[0] = n;
		}
		else
		{
			int remain = n;
			for (int j = 0; j < k - 1; j++)
			{
				res[j] = Utility.rand.nextInt(remain - (k - j - 1));
				remain -= res[j];
			}
			res[k - 1] = remain;
			assert(res[k - 1] >= 0);
		}
		return res;
	}
	
	int[][] getF(HashMap<ColorGroup, List<WeightedPoint>> colorMap, int k)
	{
		HashSet<Integer> col = new HashSet<Integer>();
		for (ColorGroup c : colorMap.keySet())
		{
			col.addAll(c.c);
		}
		int numCol = col.size();
		int[][] F = new int[numCol][k];
		for (ColorGroup c : colorMap.keySet())
		{
			int size = colorMap.get(c).size();
			int[] tmp = randPart(size, k);
			for (Integer cc : c.c)
			{
				for (int i = 0; i < k; i++)
				{
					F[cc][i] += tmp[i];
				}
			}
		}
		return F;
	}
	
	ArrayList<Point> getCenter(ArrayList<WeightedPoint> instance, int k)
	{
		ArrayList<Point> C = new ArrayList<Point>();
		for (int j = 0; j < k; j++)
		{
			C.add(instance.get(Utility.rand.nextInt(instance.size())).data);
		}
		return C;
	}
	
	ArrayList<Object[]> getTests(ArrayList<WeightedPoint> instance, int cases, int k)
	{
		HashMap<ColorGroup, List<WeightedPoint>> colorMap = WeightedPoint.colorMap(instance);
		ArrayList<Object[]> res = new ArrayList<Object[]>();
		for (int t = 0; t < cases; t++)
		{
			int[][] F = getF(colorMap, k);
			ArrayList<Point> C = getCenter(instance, k);
			res.add(new Object[] {F, C});
		}
		return res;
	}
	
	Object[] evaluateCoreset(List<WeightedPoint> X, int k, Objective O, ArrayList<Object[]> tests, int cases, long[] objRunTime, double[] objValue)
	{
		double[] perTime = new double[cases];
		double[] perErr = new double[cases];
		class Run implements Runnable
		{
			final int i;
			public Run(int i)
			{
				this.i = i;
			}
			@Override
			public void run() {
				int[][] F = (int[][])tests.get(i)[0];
				ArrayList<Point> C = (ArrayList<Point>)(tests.get(i)[1]);
				
				long t0 = System.currentTimeMillis();
				double cor = Clustering.evaluate(X, F, C, O);
				perTime[i] = System.currentTimeMillis() - t0;
				
				perErr[i] = Math.abs(cor - objValue[i]) / objValue[i];
				//err = Math.max(err, Math.abs(cor - objValue[i]) / objValue[i]);
				
				System.out.printf("evaluate coreset progress: %d/%d\n", i, cases);
			}
		}

		ExecutorService ser = Executors.newFixedThreadPool(1);
		for (int i = 0; i < cases; i++)
		{
			ser.execute(new Run(i));
		}
		ser.shutdown();
		while (!ser.isTerminated()) {
			try {
				Thread.sleep(1000);
				System.out.println("waiting");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		double avgTime = 0;
		double err = 0;
		for (int i = 0; i < cases; i++)
		{
			avgTime += perTime[i];
			err = Math.max(err, perErr[i]);
		}
		return new Object[] {avgTime / cases, err};
	}
	
	Object[] evaluateOriginal(ArrayList<WeightedPoint> X, int k, Objective O, ArrayList<Object[]> tests, String name)
	{
		int cases = tests.size();
		
		long[] runTime = new long[cases];
		double[] objValue = new double[cases];
		
		class tmpRun implements Runnable
		{
			
			final int i;
			public tmpRun(int i)
			{
				this.i = i;
			}

			@Override
			public void run() {
				int[][] F = (int[][])tests.get(i)[0];
				ArrayList<Point> C = (ArrayList<Point>)(tests.get(i)[1]);
				
				long t0 = System.currentTimeMillis();
				objValue[i] = Clustering.evaluate(X, F, C, O);
				runTime[i] = System.currentTimeMillis() - t0;
				System.out.printf("evaluate original progress: %d/%d\n", i, cases);	
			}
			
		}
		ExecutorService ser = Executors.newFixedThreadPool(1);
		for (int i = 0; i < cases; i++)
		{
			ser.execute(new tmpRun(i));
			/*int[][] F = (int[][])tests.get(i)[0];
			ArrayList<Point> C = (ArrayList<Point>)(tests.get(i)[1]);
			
			long t0 = System.currentTimeMillis();
			objValue.add(Clustering.evaluate(X, F, C, O));
			runTime.add(System.currentTimeMillis() - t0);
			System.out.printf("evaluate original progress: %d/%d\n", i, cases);*/
		}
		ser.shutdown();
		while (!ser.isTerminated()) {
			try {
				Thread.sleep(1000);
				System.out.println("waiting");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return new Object[] {runTime, objValue};
	}
	
	Object[] prepareTests(ArrayList<WeightedPoint> instance, int k, int cases, Objective O, String name)
	{
		ArrayList<Object[]> tests = null;
		long[] objRunTime = null;
		double[] objValue = null;
		if (debug.equals("true"))
		{
			try {
				ObjectInputStream in = new ObjectInputStream(new FileInputStream(new File("data/debug_" + name +".data")));
				tests = (ArrayList<Object[]>)in.readObject();
				objRunTime = (long[])in.readObject();
				objValue = (double[])in.readObject();
				in.close();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else
		{
			tests = getTests(instance, cases, k);
			Object[] tmp = evaluateOriginal(instance, k , O, tests, name);
			objRunTime = (long[])(tmp[0]);
			objValue = (double[])(tmp[1]);
			if (debug.equals("gen"))
			{
				try {
					ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(new File("data/debug_" + name + ".data")));
					out.writeObject(tests);
					out.writeObject(objRunTime);
					out.writeObject(objValue);
					out.close();
				} catch (Exception e)
				{
					e.printStackTrace();
				}
			}
		}
		return new Object[] {tests, objRunTime, objValue};
	}
	
	void evaluateFairMeans(ArrayList<WeightedPoint> instance, int k, int cases, double[] eps, String name) throws Exception
	{
		System.out.println("evaluate fair means");
		Objective O = Objective.getObjective(2.0);
		
		PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File("output/" + name +"_means.csv"))));
		out.println("eps,emperr_our,emperr_bico,emperr_uniform,our_size,bico_size,uniform_size,obj_time,our_time,bico_time,uniform_time,our_cons_time,bico_cons_time,uniform_cons_time");
		
		Object[] tmp = this.prepareTests(instance, k, cases, O, name + "_means");
		ArrayList<Object[]> tests = (ArrayList<Object[]>)(tmp[0]);
		long[] objRunTime = (long[])(tmp[1]);
		double[] objValue = (double[])(tmp[2]);
		
		double objTime = 0;
		for (Long t : objRunTime)
		{
			objTime += t;
		}
		objTime /= objRunTime.length;
		
		for (double e : eps)
		{
			System.out.printf("progress: %f\n", e);
			System.out.println("constructing coreset");
			long ourConsTime = System.currentTimeMillis();
			List<WeightedPoint> ourCoreset = new CoresetKMeans(instance).getCoreset(e, k);
			ourConsTime = System.currentTimeMillis() - ourConsTime;
			System.out.println(instance.size() + " " + WeightedPoint.totalWeight(ourCoreset));
			System.out.println("constructing bico");
			long bicoConsTime = 0;
			ArrayList<WeightedPoint> bicoCoreset = null;
			int start = 1, end = ourCoreset.size() * 2;
			while (start != end)
			{
				int mid = (start + end) / 2;
				bicoConsTime = System.currentTimeMillis();
				bicoCoreset = new BICO(instance).getCoreset(k, e, mid);
				bicoConsTime = System.currentTimeMillis() - bicoConsTime;
				if (bicoCoreset.size() < ourCoreset.size() * 0.95)
				{
					start = mid + 1;
				}
				else if (bicoCoreset.size() > ourCoreset.size() * 1.05)
				{
					end = mid - 1;
				}
				else
				{
					break;
				}
			}
			System.out.println("bico size: " + bicoCoreset.size());
			System.out.println("bico finished constructing");
			System.out.println("constructing uniform");
			long uniformConsTime = System.currentTimeMillis();
			List<WeightedPoint> uniformCoreset = new Uniform(instance).getCoreset(ourCoreset.size());
			uniformConsTime = System.currentTimeMillis() - uniformConsTime;
			Object[] our = evaluateCoreset(ourCoreset, k, O, tests, cases, objRunTime, objValue);
			Object[] bico = evaluateCoreset(bicoCoreset, k , O, tests, cases, objRunTime, objValue);
			Object[] uniform = evaluateCoreset(uniformCoreset, k, O, tests, cases, objRunTime, objValue);
			
			double ourTime = (double)our[0];
			double ourErr = (double)our[1];
			double bicoTime = (double)bico[0];
			double bicoErr = (double)bico[1];
			double uniformTime = (double)uniform[0];
			double uniformErr = (double)uniform[1];
			out.printf("%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d", e, ourErr, bicoErr, uniformErr,
					ourCoreset.size(), bicoCoreset.size(), uniformCoreset.size(),
					objTime, ourTime, bicoTime, uniformTime,
					ourConsTime, bicoConsTime, uniformConsTime);
			out.println();
			// System.out.printf("%.6f, %.6f\n", ourErr, bicoErr);
		}
		
		out.close();
	}
	
	void evaluateFairMedian(ArrayList<WeightedPoint> instance, int k, int cases, double[] eps, String name) throws Exception
	{
		System.out.println("evaluate fair median");
		Objective O = Objective.getObjective(1.0);

		PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File("output/" + name + "_median.csv"))));
		out.println("eps,emperr_our,emperr_uniform,our_size,uniform_size,obj_time,our_time,uniform_time,our_cons_time,uniform_cons_time");
		
		Object[] tmp = this.prepareTests(instance, k, cases, O, name + "_median");
		ArrayList<Object[]> tests = (ArrayList<Object[]>)(tmp[0]);
		long[] objRunTime = (long[])(tmp[1]);
		double[] objValue = (double[])(tmp[2]);
		
		double objTime = 0;
		for (Long t : objRunTime)
		{
			objTime += t;
		}
		objTime /= objRunTime.length;

		for (double e : eps)
		{
			System.out.printf("progress: %f\n", e);
			long ourConsTime = System.currentTimeMillis();
			List<WeightedPoint> ourCoreset = new CoresetKMedian(instance).getCoreset(e, k);
			ourConsTime = System.currentTimeMillis() - ourConsTime;
			// System.out.println("coreset size: " + ourCoreset.size());
			Object[] our = evaluateCoreset(ourCoreset, k,O, tests, cases, objRunTime, objValue);
			long uniformConsTime = System.currentTimeMillis();
			List<WeightedPoint> uniformCoreset = new Uniform(instance).getCoreset(ourCoreset.size());
			uniformConsTime = System.currentTimeMillis() - uniformConsTime;
			System.out.println(instance.size() + " " + WeightedPoint.totalWeight(uniformCoreset) + " "
					+ WeightedPoint.totalWeight(ourCoreset));
			Object[] uniform = evaluateCoreset(uniformCoreset, k, O, tests, cases, objRunTime, objValue);
			
			double ourTime = (double)our[0];
			double ourErr = (double)our[1];
			double uniformTime = (double)uniform[0];
			double uniformErr = (double)uniform[1];
			out.printf("%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%d,%d", e, ourErr, uniformErr,
					ourCoreset.size(), uniformCoreset.size(),
					objTime, ourTime, uniformTime,
					ourConsTime, uniformConsTime);
			out.println();
		}
		
		out.close();
	}
	
	void evaluateFairTree(List<WeightedPoint> instance, int p, int q, int k, String name) throws Exception
	{
		PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File("output/" + name + "_tree.csv"))));
		out.println("ori_obj,ori_fairlet_time,ori_clust_time,ori_node_num,cor_obj,cor_fairlet_time,cor_clust_time,cor_node_num,cor_cons_time");

		FairTree ft = new FairTree(instance);
		Object[] ret = ft.getResult(p, q, k, false, FairTree.fair);
		double obj = (Double)ret[0];
		double fairletTime = (Double)ret[1];
		double clustTime = (Double)ret[2];
		int node = (Integer)ret[3];
		
		double t1 = System.currentTimeMillis();
		List<WeightedPoint> coreset = new CoresetKMedian(instance).getCoreset(0.5, k);
		t1 = System.currentTimeMillis() - t1;
		t1 /= 1000.0;
		ft = new FairTree(coreset);
		//ft = new FairTree(instance);
		ret = ft.getResult(p, q, k, true, FairTree.fairwt);
		double corObj = (Double)ret[0];
		double corFairletTime = (Double)ret[1];
		double corClustTime = (Double)ret[2];
		int corNode = (Integer)ret[3];
		// corClustTime += t1;
		
		out.printf("%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f,%d,%.8f", obj, fairletTime, clustTime, node, corObj, corFairletTime,
				corClustTime, corNode, t1);
		out.close();
	}
	
	void evaluateFairLP(List<WeightedPoint> instance, DataDescriptor dd, String name) throws Exception
	{
		PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(new File("output/" + name + "_lp.csv"))));
		out.println("ori_obj,cor_obj,ori_time,cor_time,cor_cons_time");
		FairLP flp = new FairLP(instance, dd);
		Object[] ret = flp.getResult(false);
		double obj = (Double)ret[0];
		double time = (Double)ret[1];

		double t1 = System.currentTimeMillis();
		List<WeightedPoint> coreset = new CoresetKMeans(instance).getCoreset(0.5, 3);
		t1 = System.currentTimeMillis() - t1;
		t1 /= 1000.0;
		flp = new FairLP(coreset, dd);
		ret = flp.getResult(true);
		double corObj = (Double)ret[0];
		double corTime = (Double)ret[1];
		
		out.printf("%.8f,%.8f,%.8f,%.8f,%.8f", obj, corObj, time, corTime, t1);
		out.close();
	}
	
	public void run() throws Exception
	{
        final int k = 3;
        final int cases = 500; 
		double[] eps = new double[] {0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
		
		ArrayList<WeightedPoint> instance = null;

		instance = adult();
		normalize(instance);
		evaluateFairMeans(instance, k, cases, eps, "adultnorm");
		evaluateFairMedian(instance, k, cases, eps, "adultnorm");
		
		instance = bank();
		normalize(instance);
		evaluateFairMeans(instance, k, cases, eps, "banknorm");
		evaluateFairMedian(instance, k, cases, eps, "banknorm");
		
		instance = athlete();
		normalize(instance);
		evaluateFairMeans(instance, k, cases, eps, "athletenorm");
		evaluateFairMedian(instance, k, cases, eps, "athletenorm");

		instance = diabetes();
		normalize(instance);
		evaluateFairMeans(instance, k, cases, eps, "diabetesnorm");
		evaluateFairMedian(instance, k, cases, eps, "diabetesnorm");
		
        
		
		instance = bankBinary();
		evaluateFairMeans(instance, k, cases, eps, "bankbin");
		evaluateFairMedian(instance, k, cases, eps, "bankbin");
		
		instance = adultBinary();
		evaluateFairMeans(instance, k, cases, eps, "adultbin");
		evaluateFairMedian(instance, k, cases, eps, "adultbin");
		
		
		instance = diabetesBinary();
		evaluateFairMeans(instance, k, cases, eps, "diabetesbin");
		evaluateFairMedian(instance, k, cases, eps, "diabetesbin");
		
		// athlete is binary type, so we don't do another athletebin exp.
		
		instance = adult();
		evaluateFairMeans(instance, k, cases, eps, "adult");
		evaluateFairMedian(instance, k, cases, eps, "adult");
		
		instance = bank();
		evaluateFairMeans(instance, k, cases, eps, "bank");
		evaluateFairMedian(instance, k, cases, eps, "bank");

		instance = diabetes();
		evaluateFairMeans(instance, k, cases, eps, "diabetes");
		evaluateFairMedian(instance, k, cases, eps, "diabetes");
		
		instance = athlete();
		evaluateFairMeans(instance, k, cases, eps, "athlete");
		evaluateFairMedian(instance, k, cases, eps, "athlete");
		
		


		instance = adult();
		this.evaluateFairLP(instance, DataDescriptor.adult(), "adult");
		
		instance = bank();
		this.evaluateFairLP(instance, DataDescriptor.bank(), "bank");
		
		instance = athlete();
		this.evaluateFairLP(instance, DataDescriptor.athlete(), "athlete");
		
		instance = diabetes();
		this.evaluateFairLP(instance, DataDescriptor.diabetes(), "diabetes");
		
		instance = census1990();
		this.evaluateFairLP(instance, DataDescriptor.census1990(), "census1990");
		


		
		instance = adultBinary();
		evaluateFairTree(instance, 2, 5, k, "adult");
		
		instance = bankBinary();
		evaluateFairTree(instance, 2, 5, k, "bank");
		
		instance = athlete();
		evaluateFairTree(instance, 2, 5, k, "athlete");
		
		instance = diabetesBinary();
		evaluateFairTree(instance, 2, 5, k, "diabetes");
		
		instance = census1990Binary();
		evaluateFairTree(instance, 2, 5, k, "census1990");
	}
	
	public static void main(String[] args) throws Exception
	{
		new Main().run();
	}
}

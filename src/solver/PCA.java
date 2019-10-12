package solver;

import java.util.ArrayList;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import base.Point;

// wrapper for PCA using Apache Common Math
public class PCA
{
	private PCA() {}
	public static Point pca(ArrayList<Point> data, int kth)
	{
		double[][] pointArray = new double[data.size()][data.get(0).dim];
		for (int i = 0; i < data.size(); i++)
		{
			Point p = data.get(i);
			for (int j = 0; j < p.dim; j++)
			{
				pointArray[i][j] = p.coor[j];
			}
		}

		//create real matrix
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(pointArray);
		realMatrix = realMatrix.transpose().multiply(realMatrix);

		//create covariance matrix of points, then find eigen vectors
		//see https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

		//Covariance covariance = new Covariance(realMatrix);
		//RealMatrix covarianceMatrix = covariance.getCovarianceMatrix();
		EigenDecomposition ed = new EigenDecomposition(realMatrix);
		return new Point(ed.getEigenvector(kth).toArray());
	}
}

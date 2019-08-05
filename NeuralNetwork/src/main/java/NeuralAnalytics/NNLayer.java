package NeuralAnalytics;

import java.util.Random;

public class NNLayer {
	
	double weightArr[][];
	public double hiddenLayer[][];
	public NNLayer(int neuronCount, int inputCount) {
		
		weightArr = new double[inputCount][neuronCount];
		Random generator = new Random(0);
		
		for(int i=0;i<inputCount;i++)
		{
			for(int j=0;j<neuronCount;j++)
			{
				//weightArr[i][j] = (2 * Math.random()) - 1;
				weightArr[i][j] = (2 * generator.nextDouble()) - 1;
			}
		}
		
	}
	
	public void demonValue(double[][] demonVal) {
        this.weightArr = NNUtil.matrixAddition(weightArr, demonVal);
    }
}

package NeuralAnalytics;

public class NNUtil {
	
	public static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivativeFunction(double x) {
        return x * (1 - x);
    }
    
    public static double scaleByValue(double x, double y)
    {
    	return x = y*x;
    }
	public static double[][] matrixMultiplication(double[][] input, double[][] weights)
	{
		double output[][] = new double[input.length][weights[0].length];
		
		for(int i=0;i<input.length;i++)
		{
			for(int j = 0;j<weights[0].length;j++ )
			{	
				double sum = 0.0;
				for(int k = 0;k < input[0].length;k++)
				{
					sum+= (input[i][k] * weights[k][j]);
				}
				output[i][j] = sum;
			}
		} return output;
	}
	
	public static double[][] applySigmoidFunc(double[][] matrix)
	{
		double[][] sigmoidOutput = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
            	sigmoidOutput[i][j] = sigmoidFunction(matrix[i][j]);
            }
        } return sigmoidOutput;
	}
	
	public static double[][] applySigmoidDerivativeFunc(double[][] matrix)
	{
		double[][] sigmoidDerivativeOutput = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
            	sigmoidDerivativeOutput[i][j] = sigmoidDerivativeFunction(matrix[i][j]);
            }
        } 
        return sigmoidDerivativeOutput;
	}
	
	public static double[][] applyLearningRateFunc(double[][] matrix, double y)
	{
		double[][] adjustmentArr = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
            	adjustmentArr[i][j] = matrix[i][j]*y;// scaleByValue(, y);
            }
        } return adjustmentArr;
	}
	
	public static double[][] matrixSubtraction(double[][] output, double[][] hiddenLayer) {
		
		double subtractedOutput[][] = new double[output.length][output[0].length];

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
            	subtractedOutput[i][j] = output[i][j] - hiddenLayer[i][j];
            }
        } return subtractedOutput;
	}
	
	public static double[][] matrixAddition(double[][] weight, double[][] demonValue) {
		
		double adjustedOutput[][] = new double[weight.length][weight[0].length];

        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
            	adjustedOutput[i][j] = weight[i][j] + demonValue[i][j];
            }
        } return adjustedOutput;
	}
	
	public static double[][] scalarMultiplication(double[][] deltaarray, double[][] sigderivative)
	{
		double deltaOutput[][] = new double[deltaarray.length][deltaarray[0].length];
        for (int i = 0; i < deltaarray.length; ++i) {
            for (int j = 0; j < deltaarray[i].length; ++j) {
            	deltaOutput[i][j] = deltaarray[i][j] * sigderivative[i][j];
            }
        }
        return deltaOutput;
	}
	
	public static double[][] matrixTranspose(double[][] deltaarray)
	{
		double[][] adjustedArray = new double[deltaarray[0].length][deltaarray.length];
        for (int i = 0; i < deltaarray.length; i++) {
            for (int j = 0; j < deltaarray[i].length; ++j) {
            	adjustedArray[j][i] = deltaarray[i][j];
            }
        } return adjustedArray;
	}

}

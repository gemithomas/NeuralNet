package NeuralAnalytics;

import NeuralAnalytics.*;

import java.util.ArrayList;

public class NNet {
	
	private ArrayList<NNLayer> list;
	private ArrayList<AdjustableWeight> adjustmentList;
	private double learningRate;
	public NNet(ArrayList<NNLayer> list, double learningRate) {
		this.list = list;
		adjustmentList = new ArrayList<AdjustableWeight>();
		this.learningRate = learningRate;
	}
	
	public void forwardpropagation(double[][] input)
	{	
		double[][] previous_output = input;
		for(int i=0;i<list.size();i++)
		{
			list.get(i).hiddenLayer = NNUtil.applySigmoidFunc(NNUtil.matrixMultiplication(previous_output, list.get(i).weightArr));
			previous_output = list.get(i).hiddenLayer;
		}
	}
	
	public void train(double[][] input, double[][] output, int iterations)
	{
		long fwd_time = 0,back_time = 0;
		for(int iter=0;iter<iterations;iter++)
		{
			long start_fwd = System.currentTimeMillis();
			forwardpropagation(input);
			long end_fwd = System.currentTimeMillis();
			fwd_time  += end_fwd- start_fwd;
		
			double[][] errorlastLayer = NNUtil.matrixSubtraction(output, list.get(list.size()-1).hiddenLayer);
			double[][] current_error;
			double[][] previous_delta = null;
			adjustmentList.clear();
			for(int i=list.size()-1;i>=0;i--)
			{	
				if(i==list.size()-1)
				{
					current_error = errorlastLayer;
					if(iter == iterations -1)
					{
						System.out.println("Cost =" +CostFunctionCalibration(errorlastLayer));
					}
				}
				else
				{	//Error at L-1 Layer is calculated by Matrix Multiplication of delta & weights at Lth layer
					current_error = NNUtil.matrixMultiplication( previous_delta, NNUtil.matrixTranspose(list.get(i+1).weightArr));
				}
				double[][] deltaLayer = NNUtil.scalarMultiplication(current_error, NNUtil.applySigmoidDerivativeFunc(list.get(i).hiddenLayer));
				double[][] adjustmentLayer;
				if(i != 0)
					adjustmentLayer = NNUtil.matrixMultiplication(NNUtil.matrixTranspose(list.get(i-1).hiddenLayer), deltaLayer);
				else
					adjustmentLayer = NNUtil.matrixMultiplication(NNUtil.matrixTranspose(input), deltaLayer);
				
				previous_delta = deltaLayer;
				
				adjustmentLayer = NNUtil.applyLearningRateFunc(adjustmentLayer, learningRate);
				AdjustableWeight obj = new AdjustableWeight(adjustmentLayer);
				adjustmentList.add(obj); 
				//adjustmentList needed to add all AdjustmentLayer Matrices at this stage.
				// adjustFunction - will adjust the  weights of current layer. 
				// If we adjust the current layer weights for Error calculation at the L-1 Layer will be wrong.
				// Hence the adjustmentList
			}
			long end_back = System.currentTimeMillis();
			back_time  += end_back - end_fwd;
			for(int i = 0;i < list.size(); i++)
			{	
				int j = list.size()-1-i;
				list.get(i).demonValue(adjustmentList.get(j).getAdjustableWeight());
			}
		}
		
		//System.out.println("Time taken for fwd  in (sec) "+ fwd_time/1000);
		//System.out.println("Time taken for back  in (sec) "+ back_time/1000);
	}
	
	// AdjustableWeight Class added to create objects and add to adjustmentList!
	private double CostFunctionCalibration(double[][] errorLastLayer)
	{	
		double sum = 0;
		int count = 0;
		double cost = 0;
		for(int i=0;i<errorLastLayer.length;i++)
		{
			for(int j=0;j<errorLastLayer[i].length;j++)
			{
				double val = errorLastLayer[i][j];
				sum = sum + Math.pow(val, 2);
			} count++;
		} 
		return Math.sqrt(sum/count);
	}
	private static class AdjustableWeight
	{
		double[][] adjustableWeight;
		public AdjustableWeight(double[][] adjustableWeight) {
			
			this.adjustableWeight = adjustableWeight;
		}
		public double[][] getAdjustableWeight() {
			return adjustableWeight;
		}
		public void setAdjustableWeight(double[][] adjustableWeight) {
			this.adjustableWeight = adjustableWeight;
		}
	}
	
	public double[][] getOutput() {
        return list.get(list.size()-1).hiddenLayer;
    }
}

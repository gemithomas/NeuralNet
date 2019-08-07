package NeuralProjects;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import NeuralAnalytics.NNLayer;
import NeuralAnalytics.NNet;

public class DigitRecognition {

	public static void main(String[] args) throws IOException {
		

        //create hidden layer that has 4 neurons and 3 inputs per neuron
        NNLayer layer1 = new NNLayer(10, 784);

        // create output layer that has 1 neuron representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NNLayer layer2 = new NNLayer(10, 10);
        
        NNLayer layer3 = new NNLayer(10, 10);
        ArrayList<NNLayer> list = new ArrayList<NNLayer>();
        list.add(layer1);
        list.add(layer2);
        list.add(layer3);
        
        NNet net = new NNet(list, 0.1);
        
//	   double[][] inputs = new double[][]{
//	            {0, 0, 1, 1},
//	            {1, 1, 1, 1},
//	            {1, 0, 1, 0},
//	            {0, 1, 1, 1},
//	            {0, 0, 0, 0},
//	            {1, 0, 0, 1},
//
//	    };
//	    double[][] outputs = new double[][]{
//            {0,1},
//            {1,1},
//            {1,0},
//            {0,1},
//            {0,0},
//            {1,1},
//    };
	    FileUtil fu = new FileUtil("Input.txt", true);
	    double[][] inputFromFile = fu.getInput();
	    double[][] outputFromFile = fu.getOutput();
	    
	    System.out.println("Training the neural net");
	    long start = System.currentTimeMillis();
	    net.backpropagation(inputFromFile, outputFromFile, 5000);
	    long finish = System.currentTimeMillis();
	    long timeElapsed = finish - start;
	    System.out.println("Finished training in (sec) "+ timeElapsed/1000);
	    
	    // calculate the predictions on unknown data
        // 1, 0, 0
        //predict(new double[][]{{0, 1, 0, 1}}, net);

        // 0, 1, 0
        //predict(new double[][]{{1, 1, 1, 0}}, net);
	    
	    FileUtil ftu = new FileUtil("Test_Input.txt", false);
	    double[][] testInputFromFile = ftu.getInput();
	    double[][] testOutputFromFile = ftu.getOutput();
	    
	    predict(testInputFromFile, testOutputFromFile, net);

    }
	
    public static void predict(double[][] testInput, double[][] ExpectedOutput, NNet net) {
        net.forwardpropagation(testInput);
        double output[][] = net.getOutput();
        
        double predictedOutput[][] = predictedOutput(output);
        // Precision
        double precisionCount = 0;
        for(int i = 0;i< predictedOutput.length;i++)
        {  	
        		if(predictedOutput[i][0] == ExpectedOutput[i][0] )
        		{
        			precisionCount++;
        		}
        }
        double precision = precisionCount/predictedOutput.length;
        System.out.println(precision);           
    }
    
    public static double[][] predictedOutput( double output[][])
	{	
		
		double finalOutput[][] = new double[output.length][1];
		for (int i = 0; i < output.length; i++) {
			int index = 0;
			int count = 0;
            for (int j = 0; j < output[i].length; j++) {
            	if(output[i][j] > 0.5)
            	{	
            		count++;
            		if(count > 1) 
            			{
            				finalOutput[i][0] = -1;
            				break;
            			}
            		output[i][j] = 1;
            		index = j;
            	} else
            	{
            		output[i][j] = 0;
            	}           		
            } if(count == 1)
            {
            	finalOutput[i][0] = index;
            }
        } return finalOutput;
	}
}

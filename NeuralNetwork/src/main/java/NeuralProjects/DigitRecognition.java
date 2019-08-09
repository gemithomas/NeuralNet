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
		
		System.out.println("Start of Training:");
		long start_pgm = System.currentTimeMillis();
		int num_neuron_layer1 = Integer.valueOf(args[0]);
		int num_neuron_layer2 = Integer.valueOf(args[1]);
		int num_iteration = Integer.valueOf(args[2]);
		double learningRate = Double.valueOf(args[3]);
		
        NNLayer layer1 = new NNLayer(num_neuron_layer1, 784);
        

        // create output layer that has 1 neuron representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NNLayer layer2 = new NNLayer(num_neuron_layer2, num_neuron_layer1);
        NNLayer layer3 = new NNLayer(10, num_neuron_layer2);
        
        //NNLayer layer4 = new NNLayer(10, 10);
        ArrayList<NNLayer> list = new ArrayList<NNLayer>();
        list.add(layer1);
        list.add(layer2);
        list.add(layer3);
        //list.add(layer4);
        System.out.println("Learning Rate is = "+learningRate);
        NNet net = new NNet(list, learningRate);
        
	    FileUtil fu = new FileUtil("Input.txt", true);
	    double[][] inputFromFile = fu.getInput();
	    double[][] outputFromFile = fu.getOutput();
	    
	    System.out.println("Training the neural net with num_iteration "+ num_iteration);
	    long start = System.currentTimeMillis();
	    net.train(inputFromFile, outputFromFile, num_iteration);
	    //check the Precision for output from training 
	    //double output[][] = net.getOutput();
	    predictionCheckFromBackProp(inputFromFile, outputFromFile, net);
	    long finish = System.currentTimeMillis();
	    long timeElapsed = finish - start;
	    System.out.println("Finished training in (sec) "+ timeElapsed/1000);
	    
	    FileUtil ftu = new FileUtil("Test_Input.txt", false);
	    double[][] testInputFromFile = ftu.getInput();
	    double[][] testOutputFromFile = ftu.getOutput();
	    
	    predict(testInputFromFile, testOutputFromFile, net);
	    long end_pgm = System.currentTimeMillis();
	    System.out.println("Finished training in (sec) "+ (end_pgm-start_pgm)/1000);
	    
    }
	
	public static void predictionCheckFromBackProp(double[][] testInput, double[][] ExpectedOutput, NNet net) {
        double output[][] = net.getOutput();
        
        double predictedOutput[][] = convertBinaryRepresentationToInt(output);
        double actualOutput[][] = convertBinaryRepresentationToInt(ExpectedOutput);
        // Precision
        double precisionCount = 0;
        for(int i = 0;i< predictedOutput.length;i++)
        {  	
        		if(predictedOutput[i][0] == actualOutput[i][0] )
        		{
        			precisionCount++;
        		}
        }
        double precision = precisionCount/predictedOutput.length;
        System.out.println("Precision from training data set " + precision);           
    }
	
    public static void predict(double[][] testInput, double[][] ExpectedOutput, NNet net) {
        net.forwardpropagation(testInput);
        double output[][] = net.getOutput();
        
        double predictedOutput[][] = convertBinaryRepresentationToInt(output);
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
        System.out.println("Precision from test data set " + precision);          
    }
    
    public static double[][] convertBinaryRepresentationToInt( double output[][])
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

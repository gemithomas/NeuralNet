package NeuralProjects;

import java.util.ArrayList;

import NeuralAnalytics.NNLayer;
import NeuralAnalytics.NNet;

public class RecognizeColumn {

	public static void main(String[] args) {
		

        //create hidden layer that has 4 neurons and 3 inputs per neuron
        NNLayer layer1 = new NNLayer(5, 4);

        // create output layer that has 1 neuron representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NNLayer layer2 = new NNLayer(2, 5);
        ArrayList<NNLayer> list = new ArrayList<NNLayer>();
        list.add(layer1);
        list.add(layer2);
        
        NNet net = new NNet(list, 0.1);
        
	   double[][] inputs = new double[][]{
	            {0, 0, 1, 1},
	            {1, 1, 1, 1},
	            {1, 0, 1, 0},
	            {0, 1, 1, 1},
	            {0, 0, 0, 0},
	            {1, 0, 0, 1},

	    };
	
	    double[][] outputs = new double[][]{
	            {0,1},
	            {1,1},
	            {1,0},
	            {0,1},
	            {0,0},
	            {1,1},
	    };
	
	    System.out.println("Training the neural net");
	    net.train(inputs, outputs, 10000);
	    System.out.println("Finished training");
	    
	    // calculate the predictions on unknown data
        // 1, 0, 0
        predict(new double[][]{{0, 1, 0, 1}}, net);

        // 0, 1, 0
        predict(new double[][]{{1, 1, 1, 0}}, net);

    }

    public static void predict(double[][] testInput, NNet net) {
        net.forwardpropagation(testInput);

        // then 
        System.out.println("Prediction on data "
                + testInput[0][0] + " "
                + testInput[0][1] + " "
                + testInput[0][2] + " "
                + testInput[0][3] + " -> "
                + net.getOutput()[0][0] + ", "+ net.getOutput()[0][1] + " expected -> " + testInput[0][0]);
    }

	}

package NeuralAnalytics;

import NeuralProjects.FileUtil;
import org.junit.Test;
import java.io.IOException;
import java.util.ArrayList;

import static org.junit.Assert.*;

public class NNUtilTest
{	

    @Test
    public void matrixMultiplication()
    {
        double weightArr[][];

        try
        {
            FileUtil fu = new FileUtil("Input.txt", true);

            double[][] inputFromFile = fu.getInput();

            weightArr = new double[784][20];

            double output[][] = new double[inputFromFile.length][weightArr[0].length];

            output = NNUtil.matrixMultiplication(inputFromFile, weightArr);

            assertNotNull(output);

        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    @Test
    public void SigmoidFunctionOnMatrix()
    {
        double weightArr[][];

        try
        {
            FileUtil fu = new FileUtil("Input.txt", true);

            double[][] inputFromFile = fu.getInput();

            weightArr = new double[784][20];

            double output[][] = new double[inputFromFile.length][weightArr[0].length];

            output = NNUtil.matrixMultiplication(inputFromFile, weightArr);

            double[][] ans = NNUtil.applySigmoidFunc(output);

            assertNotNull(ans);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    @Test
    public void SigmoidDerivativeFunctionOnMatrix()
    {
        double weightArr[][];

        try
        {
            FileUtil fu = new FileUtil("Input.txt", true);

            double[][] inputFromFile = fu.getInput();

            weightArr = new double[784][20];

            double output[][] = new double[inputFromFile.length][weightArr[0].length];

            output = NNUtil.matrixMultiplication(inputFromFile, weightArr);

            double[][] matrix = NNUtil.applySigmoidFunc(output);

            double[][] ans = NNUtil.applySigmoidDerivativeFunc(matrix);

            assertNotNull(ans);

        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

	@Test
    public void NeuralNetworkCheck()
    {
    	NNLayer layer1 = new NNLayer(5, 4);
        NNLayer layer2 = new NNLayer(1, 5);
        ArrayList<NNLayer> list = new ArrayList<NNLayer>();
        list.add(layer1);
        list.add(layer2);
        
        NNet net = new NNet(list, 0.1);
        
        double[][] input = new double[][]{
            {0, 0, 1, 1},
            {1, 1, 1, 1},
            {1, 0, 1, 0},
            {0, 1, 1, 1},
            {0, 0, 0, 0},
            {1, 0, 0, 1},

    };

    double[][] output = new double[][]{
            {0},
            {1},
            {1},
            {0},
            {0},
            {1},
    };
    net.train(input, output, 10000);
    
    
    double[][] testInput = new double[][]{{1, 1, 1, 0}};
    double expectedOutput = 1;
    net.forwardpropagation(testInput);
    double testOuptput = net.getOutput()[0][0];
    
    double delta = 0.2;
	assertEquals(expectedOutput, testOuptput, delta);
    }
	
	@Test
	public void sigmoidCalibration()
	{
		double y = 0.23;
		
		double actualOutput = 1 / (1 + Math.exp(-y));
		double expectedOutput = NNUtil.sigmoidFunction(y);
		
		double delta = 0.2;
		assertEquals(expectedOutput, actualOutput, delta);
	}  
	
	@Test
    public void sigmoidFunction()
    {
        //1 / (1 + Math.exp(-x));
        assertEquals(1, NNUtil.sigmoidFunction(1), 1);
        assertEquals(1, NNUtil.sigmoidFunction(3), 1);
        assertEquals(1, NNUtil.sigmoidFunction(4), 1);

    }

    @Test
    public void sigmoidDerivativeFunction()
    {
        //x * (1 - x);
        assertEquals(-2.0, NNUtil.sigmoidDerivativeFunction(2), 1);
        assertEquals(-12.0, NNUtil.sigmoidDerivativeFunction(4), 1);
        assertEquals(-42.0, NNUtil.sigmoidDerivativeFunction(7), 1);


    }

    @Test
    public void scaleByValue()
    {
        //x = y * x;
        assertEquals(6.0, NNUtil.scaleByValue(2, 3), 1);
        assertEquals(12.0, NNUtil.scaleByValue(4, 3), 1);
        assertEquals(30.0, NNUtil.scaleByValue(10, 3), 1);


    }

}
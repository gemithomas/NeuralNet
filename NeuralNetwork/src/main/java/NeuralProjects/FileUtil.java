package NeuralProjects;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import NeuralAnalytics.NNet;

public class FileUtil {
	
	double[][] input;
	double[][] output;
	int count_output = 0;
	int count_input = 0;
	int numberOfLines = 0;
	boolean flagConvertExpectedOutput;
	public FileUtil(String filename, boolean flagConvertExpectedOutput) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String fileLine = null;
		while((fileLine = br.readLine())!= null)
		{
			numberOfLines++;
		}
		this.flagConvertExpectedOutput = flagConvertExpectedOutput;
		input = new double[numberOfLines][784];
		if(flagConvertExpectedOutput)
		{
			output = new double[numberOfLines][10];
		}else {
			output = new double[numberOfLines][1];
		}
		
		ReadFile(filename, flagConvertExpectedOutput);
	}
	public void ReadFile(String filename, boolean convertOutput) throws NumberFormatException, IOException
    {	
		BufferedReader br = new BufferedReader(new FileReader(filename));
		
		String fileLine = null;
		while((fileLine = br.readLine())!= null)
		{	
			String[] fields = fileLine.split(",");
			if(convertOutput)
			{
				int output_integer = new Integer(fields[0]);
				convertExpectedOutputtoBinary(output_integer);
			}else {
				int output_integer = new Integer(fields[0]);
				output[count_input][0] = output_integer;
			}
			for(int i=1;i<=784;i++)
			{	
				double normalized_value = normalize(new Double(fields[i]));
				input[count_input][i-1] = normalized_value;
				
			}
			count_input++;		
		}
    }
	public void convertExpectedOutputtoBinary(int output_integer)
	{	
		for(int i = 0; i<10;i++)
		{
			output[count_output][i] = 0;
		}
		output[count_output][output_integer] = 1;
		count_output++;
	}
		
	public double[][] getInput() {
		return input;
	}
	public void setInput(double[][] input) {
		this.input = input;
	}
	public double[][] getOutput() {
		return output;
	}
	public void setOutput(double[][] output) {
		this.output = output;
	}
	
	public double normalize(Double array_value){
		
		return array_value/255;
	}
	
}

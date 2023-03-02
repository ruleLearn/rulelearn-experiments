package org.rulelearn.experiments.helpers;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ConsistencyFormatter {
	static final String percentDecimalFormat = "%.1f"; //tells number of decimal places in percentages

	public static void main(String[] args) throws IOException { //pass e.g. "temp/input.csv" "temp/output.csv"
		if (args.length != 2) {
			System.out.println("Synopsis: ConsistencyFormatter csvInputFilePath csvOutputFilePath");
			return;
		}
		
		String cellSeparator = "\t";
		String inputValueSeparator = ", ";
		String outputValueSeparator = "|";
		
		String csvInputFilePath = args[0];
		String csvOutputFilePath = args[1];
		
		List<String> csvLines = new ArrayList<>(16);
		
        try (FileReader fileReader = new FileReader(csvInputFilePath);
        	BufferedReader reader = new BufferedReader(fileReader)) {
            String inputLine;
            
            while ((inputLine = reader.readLine()) != null) {
                csvLines.add(inputLine);
            }
        } catch (IOException exception) {
            System.out.println("Cannot read CSV file: "+exception.getMessage());
            return;
        }
        
        String[] cells;
        String[] values;
        StringBuilder outputBuilder = new StringBuilder();
        
        int linesCount = 0;
        for (String csvLine : csvLines) {
        	cells = csvLine.split(cellSeparator);
        	linesCount++;
        	
        	int cellsCount = 0;
        	for (String cell : cells) {
        		values = cell.split(inputValueSeparator);
//        		cellsCount++;
        		
        		int valuesCount = 0;
        		String[] savedValues = new String[3];
        		for (String value : values) {
        			value = value.strip();

    				value = value.replace("prepost2: ", "");
    				value = value.replace("post2: ", "");
    				value = value.replace("pre: ", "");
    				
    				value = value.substring(0);
        			
        			double number = Double.valueOf(value);
        			String newValue = round(number);
        			
        			savedValues[valuesCount] = newValue;
        			//outputBuilder.append(newValue);
        			
        			valuesCount++; //number of processed values
        			
        			if (valuesCount == 3) {
        				outputBuilder.append(savedValues[1]).append(outputValueSeparator).append(savedValues[2]);
        			}
        			
//        			if (valuesCount < values.length) {
//        				outputBuilder.append(outputValueSeparator);
//        			} else if (cellsCount < cells.length) {
//        				outputBuilder.append(cellSeparator);
//        			}
        		} //for values
        		cellsCount++;
        		if (cellsCount < cells.length) {
    				outputBuilder.append(cellSeparator);
    			}
        	} //for cells
        	if (linesCount < csvLines.size()) {
        		outputBuilder.append(System.lineSeparator());
        	}
        }
        
        //print to console
        System.out.println(outputBuilder.toString());
        
        //write to file
        try (FileOutputStream fileStream = new FileOutputStream(csvOutputFilePath);
        		OutputStreamWriter writer = new OutputStreamWriter(fileStream)) {
        	writer.write(outputBuilder.toString());
        } catch (IOException exception) {
        	System.out.println("Could not write to file "+csvOutputFilePath);
			return;
        }
	}
	
	public static String round(double number) {
		number = number * 100;
		
		if (number == 100) {
			return "100";
		} else if (number == 0) {
			return "0";
		} else {
			return String.format(Locale.US, percentDecimalFormat, number);
		}
	}
}

/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/**
 *
 * File: Main.java
 *
 * This is the main class of the algorithm.
 * It gets the configuration script, builds the classifier and executes it.
 *
 * @author Written by Javier García (University of Jaén)  30/05/2015
 * @version 1.1
 * @since JDK1.5
 */

package keel.Algorithms.Monotonic_Classification.MoNGEL;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import org.rulelearn.data.AttributePreferenceType;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.arff.ArffReader;
import org.rulelearn.experiments.AttributeRanges;
import org.rulelearn.experiments.AttributeRanges.AttributeRange;
import org.rulelearn.experiments.InformationTable2InstanceSet;

import keel.Dataset.InstanceSet;

public class Main {
	
	//The classifier
	private static MoNGEL classifier;
	
	/**
	 * The main method of the class
	 *
	 * @param args Arguments of the program (a configuration script, generally)
	 *
	 */
	public static void main (String args[]) {
		//BEGIN MSz
		long seed = 0L;
		InstanceSet trainData = null;
		InstanceSet referenceData = null;
		InstanceSet testData = null;
		String[] outFiles = new String[] {
				"src/main/resources/data/MoNGEL-results/train-results.txt",
				"src/main/resources/data/MoNGEL-results/test-results.txt",
				"src/main/resources/data/MoNGEL-results/summary-results.txt"
		};
		List<AttributeRanges.AttributeRange> attributeRangesList = new ArrayList<AttributeRanges.AttributeRange>();
		attributeRangesList.add(new AttributeRange(0, 4.3, 7.9));
		attributeRangesList.add(new AttributeRange(1, 2.0, 4.4));
		attributeRangesList.add(new AttributeRange(2, 1.0, 6.9));
		attributeRangesList.add(new AttributeRange(3, 0.1, 2.5));
		AttributeRanges attributeRanges = new AttributeRanges(attributeRangesList);
		
		InformationTable informationTable;
		
		ArffReader arffReader = new ArffReader();
		try {
			informationTable = arffReader.read("src/main/resources/data/arff/iris-0-train.arff", AttributePreferenceType.GAIN);
		} catch (FileNotFoundException exception) {
			exception.printStackTrace();
			return;
		}
		
		trainData = InformationTable2InstanceSet.convert(informationTable, "iris-train", attributeRanges);
		referenceData = InformationTable2InstanceSet.convert(informationTable, "iris-reference", attributeRanges);
		
		try {
			informationTable = arffReader.read("src/main/resources/data/arff/iris-0-test.arff", AttributePreferenceType.GAIN);
		} catch (FileNotFoundException exception) {
			exception.printStackTrace();
			return;
		}
		
		testData = InformationTable2InstanceSet.convert(informationTable, "iris-test", attributeRanges);
		//END MSz
		
		classifier = new MoNGEL(seed, trainData, referenceData); //MSz
        classifier.initializeRules();     // Initializing the rules structures
		classifier.getRules();
		classifier.loadTestData(testData);//MSz
		classifier.setOutFiles(outFiles); //MSz
		classifier.execute();			  // Executing the method MoNGEL
        classifier.printOutput();	   	  // Processing the output following the Keel requirements
	} //end-method 
        
        
  
} //end-class



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
 * File: HyperrectanglesAlgorithm.java
 *
 * A general framework for Hyperrectangles Algorithms.
 * This class contains all common operations in the development of a
 * Hyperrectangles algorithm. Any Hyperrectangles algorithm can extend this
 * class and, by implementing the abstract methods "readParameters", "evaluate"
 * and "writeRules" getting most of its work already done.
 *
 * @author Written by Joaquin Derrac (University of Granada) 8/7/2009
 * @author Modified by Joaquin Derrac (University of Granada) 17/10/2009
 * @version 1.2
 * @since JDK1.5
 *
 */
package keel.Algorithms.Monotonic_Classification.Basic;

import java.util.Arrays;
import java.util.StringTokenizer;

import keel.Algorithms.Classification.Classifier;
import keel.Dataset.Attribute;
import keel.Dataset.Instance;
import keel.Dataset.InstanceSet;

import org.core.Files;

public abstract class HyperrectanglesAlgorithm implements Classifier {

	//Files

//	protected String outFile[];
//	protected String testFile;
//	protected String trainFile;
//	protected String referenceFile;
	
	//Instance Sets
	
	protected InstanceSet train;
	protected InstanceSet test;
	protected InstanceSet reference;
	
	protected Instance temp;	
	
	//Data
	
	protected int inputAtt; //number of input attributes
	protected Attribute[] inputs; //array with input attributes
	protected Attribute output; //single output attribute
	protected boolean[] nulls;
	
	protected double trainData[][];
	protected int trainOutput[];
	protected double testData[][];
	protected int testOutput[];
	protected double referenceData[][];
	protected int referenceOutput[];
	protected String relation;
	
	protected int nClasses; //number of decision classes
	protected int nInstances[];
	
	//Timing
	
	protected long initialTime;
	
	protected double modelTime;
	protected double trainingTime;
	protected double testTime;
	
	//Naming
	
	protected String name;
	
	//Random seed
	
//	protected long seed;
	
	//Results
	protected int confMatrix[][];
	protected int unclassified;
	protected int realClass[][];
	protected int prediction[][];
	protected int trainConfMatrix[][];
	protected int trainUnclassified;
	protected int trainRealClass[][];
	protected int trainPrediction[][];
	protected String ruleSetText;
        protected double MAE;
        protected double trainMAE;
        private double antimontest;
        private double antimontrain;
	
//	/** 
//	 * Read the configuration and data files, and process it.
//	 * 
//	 * @param script Name of the configuration script  
//	 * 
//	 */
//	protected void readDataFiles(String script){
//	    
//		//Read of the script file
//		readConfiguracion(script);   
//		readParameters(script);
//
//		//Read of training data files
//	    try {
//			train = new InstanceSet();
//
//			train.readSet(trainFile, true);
//
//			train.setAttributesAsNonStatic();
//
//		    inputAtt = train.getAttributeDefinitions().getInputNumAttributes();
//		    inputs = train.getAttributeDefinitions().getInputAttributes();
//		    output = train.getAttributeDefinitions().getOutputAttribute(0);
//		    
//			//Normalize the data
//		     
//			normalizeTrain();
//			
//	    } catch (Exception e) {
//			System.err.println(e);
//			System.exit(1);
//	    }
//
//	    //Read of test data files
//	    try {
//			test = new InstanceSet();
//			test.readSet(testFile, false);
//		    test.setAttributesAsNonStatic();
//			//Normalize the data
//			normalizeTest();
//			
//	    } catch (Exception e) {
//			System.err.println(e);
//			System.exit(1);
//	    }
//  
//	    Attributes.clearAll();
//	    
//		//Read of reference data files
//		try {
//			reference = new InstanceSet();					
//			reference.readSet(referenceFile, true);
//			reference.setAttributesAsNonStatic();
//
//			//Normalize the data
//			normalizeReference();
//					
//		} catch (Exception e) {
//			System.err.println(e);
//			System.exit(1);
//		}
//		
//		//Now, the data is loaded and preprocessed
//
//	    //Get the number of classes
//	    nClasses=train.getAttributeDefinitions().getOutputAttribute(0).getNumNominalValues();
//	    
//	    //And the number of instances on each class
//	    
//	    nInstances=new int[nClasses];
//	    for(int i=0;i<nClasses;i++){
//	    	nInstances[i]=0;
//		}
//	    for(int i=0;i<trainOutput.length;i++){
//	    	nInstances[trainOutput[i]]++;
//	    }
//	    
//	}//end-method
	
	
	/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
	/** 
	 * Reads learning (training and reference) data and processes them.
	 * 
	 * @param trainData training data
	 * @param referenceData reference data  
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	protected void loadLearningData(InstanceSet trainData, InstanceSet referenceData) {
		//Read training data
	    try {
			train = trainData;

		    inputAtt = train.getAttributeDefinitions().getInputNumAttributes();
		    inputs = train.getAttributeDefinitions().getInputAttributes();
		    output = train.getAttributeDefinitions().getOutputAttribute(0);
		    
			//Normalize the data
			normalizeTrain();
	    } catch (Exception e) {
			System.err.println(e);
			System.exit(1);
	    }

		//Read reference data
		try {
			reference = referenceData;

			//Normalize the data
			normalizeReference();
		} catch (Exception e) {
			System.err.println(e);
			System.exit(1);
		}
		
		//Now, the data is loaded and preprocessed

	    //Get the number of classes
	    nClasses=train.getAttributeDefinitions().getOutputAttribute(0).getNumNominalValues();
	    
	    //and the number of instances in each class
	    nInstances = new int[nClasses];
	    for (int i = 0; i < nClasses; i++) {
	    	nInstances[i] = 0;
		}
	    for (int i = 0; i < trainOutput.length; i++) {
	    	nInstances[trainOutput[i]]++;
	    }
	    
	}
	
	/** 
	 * Reads test data and processes them.
	 * 
	 * @param testData test data
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public void loadTestData(InstanceSet testData) {
		//Read of test data files
	    try {
			test = testData;
			//Normalize the data
			normalizeTest();
	    } catch (Exception e) {
			System.err.println(e);
			System.exit(1);
	    }
	}
	
//	public void setOutFiles(String[] outFiles) {
//		this.outFile = outFiles;
//	}
	/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

//	/** 
//	 * Reads configuration script, and extracts its contents.
//	 * 
//	 * @param script Name of the configuration script  
//	 * 
//	 */	
//	protected void readConfiguracion (String script) {
//
//		String fichero, linea, token;
//		StringTokenizer lineasFichero, tokens;
//		byte line[];
//	    int i, j;
//
//	    outFile = new String[3];
//
//	    fichero = Files.readFile (script);
//	    lineasFichero = new StringTokenizer (fichero,"\n\r");
//
//	    lineasFichero.nextToken();
//	    linea = lineasFichero.nextToken();
//
//	    tokens = new StringTokenizer (linea, "=");
//	    tokens.nextToken();
//	    token = tokens.nextToken();
//
//	    //Getting the names of training and test files
//	    //reference file will be used as comparision
//	    
//	    line = token.getBytes();
//	    for (i=0; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    trainFile = new String (line,i,j-i);
//	    for (i=j+1; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    referenceFile = new String (line,i,j-i);
//	    for (i=j+1; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    testFile = new String (line,i,j-i);
//
//	    //Getting the path and base name of the results files
//	    
//	    linea = lineasFichero.nextToken();
//	    tokens = new StringTokenizer (linea, "=");
//	    tokens.nextToken();
//	    token = tokens.nextToken();
//
//	    //Getting the names of output files
//	    
//	    line = token.getBytes();
//	    for (i=0; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    outFile[0] = new String (line,i,j-i);
//	    for (i=j+1; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    outFile[1] = new String (line,i,j-i);
//	    for (i=j+1; line[i]!='\"'; i++);
//	    i++;
//	    for (j=i; line[j]!='\"'; j++);
//	    outFile[2] = new String (line,i,j-i);
//	    
//	} //end-method

//	/** 
//	 * Reads the parameters of the algorithm. 
//	 * Must be implemented in the subclass.
//	 * 
//	 * @param script Configuration script
//	 * 
//	 */
//	protected abstract void readParameters(String script);
	
	/** 
	 * This function builds the data matrix for training data and normalizes inputs values
	 */	
	protected void normalizeTrain() throws DataException {

	    StringTokenizer tokens;
	    double minimum[];
	    double range[];

	    //Check if dataset corresponding with a classification problem
	    
	    if (train.getAttributeDefinitions().getOutputNumAttributes() < 1) {
			throw new DataException ("This dataset haven�t outputs, so it not corresponding to a classification problem.");
	    } else if (train.getAttributeDefinitions().getOutputNumAttributes() > 1) {
			throw new DataException ("This dataset have more of one output.");
	    }

	    if (train.getAttributeDefinitions().getOutputAttribute(0).getType() == Attribute.REAL) {
			throw new DataException ("This dataset have an input attribute with float values, so it not corresponding to a classification 	problem.");
	    }

	    //Copy the data
	    
	    tokens = new StringTokenizer (train.getHeader()," \n\r");
	    tokens.nextToken();
	    relation = tokens.nextToken();

	    trainData = new double[train.getNumInstances()][inputAtt];
	    trainOutput = new int[train.getNumInstances()];
	    
	    for (int i=0; i<train.getNumInstances(); i++) {
		
	    	temp = train.getInstance(i);
			trainData[i] = temp.getAllInputValues();
			trainOutput[i] = (int)temp.getOutputRealValues(0);
			nulls = temp.getInputMissingValues();
			
			//Clean missing values
	        for (int j=0; j<nulls.length; j++){
	        	if (nulls[j]) {
	        		trainData[i][j]=0.0;
		        }
	        }
	    }
	    
	    //Normalice the data
	    
	    minimum=new double[inputAtt];
	    range=new double[inputAtt];	    

	    for (int i=0; i<inputAtt; i++) {
            if (train.getAttributeDefinitions().getInputAttribute(i).getType() != Attribute.NOMINAL) {
		    	minimum[i]=train.getAttributeDefinitions().getInputAttribute(i).getMinAttribute();
		    	range[i]=train.getAttributeDefinitions().getInputAttribute(i).getMaxAttribute()-minimum[i];
            }
	    }
	    
	    //Both real and nominal data are normaliced in [0,1]
	    
	    for (int i=0; i<train.getNumInstances(); i++) {
	    	for (int j = 0; j < inputAtt; j++) {
	            if (train.getAttributeDefinitions().getInputAttribute(j).getType() == Attribute.NOMINAL) {
	            	if(train.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()>1){
	            		trainData[i][j] /= train.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()-1;
	            	}
	            }else{
	                trainData[i][j] -= minimum[j];
	                trainData[i][j] /= range[j];
	            }
	    	}
	    }
	    
	} //end-method 

	/** 
	 * This function builds the data matrix for test data and normalizes inputs values
	 */	
	protected void normalizeTest() throws DataException {

	    StringTokenizer tokens;
	    double minimum[];
	    double range[];

	    //Check if dataset corresponding with a classification problem
	    
	    if (test.getAttributeDefinitions().getOutputNumAttributes() < 1) {
			throw new DataException ("This dataset haven�t outputs, so it not corresponding to a classification problem.");
	    } else if (test.getAttributeDefinitions().getOutputNumAttributes() > 1) {
			throw new DataException ("This dataset have more of one output.");
	    }

	    if (test.getAttributeDefinitions().getOutputAttribute(0).getType() == Attribute.REAL) {
			throw new DataException ("This dataset have an input attribute with float values, so it not corresponding to a classification 	problem.");
	    }

	    //Copy the data
	    
	    tokens = new StringTokenizer (test.getHeader()," \n\r");
	    tokens.nextToken();
	    tokens.nextToken();

	    testData = new double[test.getNumInstances()][inputAtt];
	    testOutput = new int[test.getNumInstances()];
	    
	    for (int i=0; i<test.getNumInstances(); i++) {
		
	    	temp = test.getInstance(i);
	    	testData[i] = temp.getAllInputValues();
	    	testOutput[i] = (int)temp.getOutputRealValues(0);
			nulls = temp.getInputMissingValues();
			
			//Clean missing values
	        for (int j=0; j<nulls.length; j++){
	        	if (nulls[j]) {
	        		testData[i][j]=0.0;
		        }
	        }
	    }
	    
	    //Normalice the data
	    
	    minimum=new double[inputAtt];
	    range=new double[inputAtt];	    

	    for (int i=0; i<inputAtt; i++) {
            if (test.getAttributeDefinitions().getInputAttribute(i).getType() != Attribute.NOMINAL) {
		    	minimum[i]=train.getAttributeDefinitions().getInputAttribute(i).getMinAttribute();
		    	range[i]=train.getAttributeDefinitions().getInputAttribute(i).getMaxAttribute()-minimum[i];
            }
	    }
	    
	    //Both real and nominal data are normaliced in [0,1]
	    
	    for (int i=0; i<test.getNumInstances(); i++) {
	    	for (int j = 0; j < inputAtt; j++) {
	            if (test.getAttributeDefinitions().getInputAttribute(j).getType() == Attribute.NOMINAL) {
	            	if(test.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()>1){
	            		testData[i][j] /= test.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()-1;
	            	}
	            }
	            else{
	            	testData[i][j] -= minimum[j];
	            	testData[i][j] /= range[j];
	            }
	    	}
	    }
	    
	} //end-method 
	
	/** 
	 * This function builds the data matrix for reference data and normalizes inputs values
	 */	
	protected void normalizeReference() throws DataException {

	    StringTokenizer tokens;
	    double minimum[];
	    double range[];

	    //Check if dataset corresponding with a classification problem
	    
	    if (reference.getAttributeDefinitions().getOutputNumAttributes() < 1) {
			throw new DataException ("This dataset haven�t outputs, so it not corresponding to a classification problem.");
	    } else if (reference.getAttributeDefinitions().getOutputNumAttributes() > 1) {
			throw new DataException ("This dataset have more of one output.");
	    }

	    if (reference.getAttributeDefinitions().getOutputAttribute(0).getType() == Attribute.REAL) {
			throw new DataException ("This dataset have an input attribute with float values, so it not corresponding to a classification 	problem.");
	    }

	    //Copy the data
	    
	    tokens = new StringTokenizer (reference.getHeader()," \n\r");
	    tokens.nextToken();
	    tokens.nextToken();

	    referenceData = new double[reference.getNumInstances()][inputAtt];
	    referenceOutput = new int[reference.getNumInstances()];
	    
	    for (int i=0; i<reference.getNumInstances(); i++) {
		
	    	temp = reference.getInstance(i);
	    	referenceData[i] = temp.getAllInputValues();
	    	referenceOutput[i] = (int)temp.getOutputRealValues(0);
			nulls = temp.getInputMissingValues();
			
			//Clean missing values
	        for (int j=0; j<nulls.length; j++){
	        	if (nulls[j]) {
	        		referenceData[i][j]=0.0;
		        }
	        }
	    }
	    
	    //Normalice the data
	    
	    minimum=new double[inputAtt];
	    range=new double[inputAtt];	    

	    for (int i=0; i<inputAtt; i++) {
            if (reference.getAttributeDefinitions().getInputAttribute(i).getType() != Attribute.NOMINAL) {
		    	minimum[i]=train.getAttributeDefinitions().getInputAttribute(i).getMinAttribute();
		    	range[i]=train.getAttributeDefinitions().getInputAttribute(i).getMaxAttribute()-minimum[i];
            }
	    }
	    
	    //Both real and nominal data are normaliced in [0,1]
	    
	    for (int i=0; i<reference.getNumInstances(); i++) {
	    	for (int j = 0; j < inputAtt; j++) {
	            if (reference.getAttributeDefinitions().getInputAttribute(j).getType() == Attribute.NOMINAL) {
	            	if(reference.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()>1){
	            		referenceData[i][j] /= reference.getAttributeDefinitions().getInputAttribute(j).getNominalValuesList().size()-1;
	            	}
	            }else{
	            	referenceData[i][j] -= minimum[j];
	            	referenceData[i][j] /= range[j];
	            }
	    	}
	    }
	    
	} //end-method 

	/** 
	 * Executes the classification of train and test data sets
	 * 
	 */	
	public void execute(){
		
		modelTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
//		System.out.println(name+" "+ relation + " Model " + modelTime + "s");
		
		trainRealClass = new int[trainData.length][1];
		trainPrediction = new int[trainData.length][1];	
		
		//Check  time		
		setInitialTime();
		//Working on training
		for (int i=0; i<trainRealClass.length; i++) {
			trainRealClass[i][0] = trainOutput[i];
			trainPrediction[i][0]=evaluate(trainData[i]);
		}
                antimontrain=0.0;  
                for (int i=0; i<trainData.length; i++) {
                    for (int j=i+1; j<trainData.length; j++) {
                        int aux=comparacion(trainData[i],trainData[j]);
			if((aux!=-2)&&(aux *(trainPrediction[i][0]-trainPrediction[j][0])<0))
                            antimontrain=antimontrain+1;
                    }
                }    
                antimontrain=2*antimontrain/(trainPrediction.length*trainPrediction.length-trainPrediction.length);    
		trainingTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
		
		//Writing results
//		writeOutput(outFile[0], trainRealClass, trainPrediction);
//		System.out.println(name+" "+ relation + " Training " + trainingTime + "s");
		
		//Working on test
		realClass = new int[testData.length][1];
		prediction = new int[testData.length][1];	
		
		//Check  time		
		setInitialTime();
		
                for (int i=0; i<realClass.length; i++) {
			realClass[i][0] = testOutput[i];
			prediction[i][0]=evaluate(testData[i]);
		}
                antimontest=0.0;  
                for (int i=0; i<testData.length; i++) {
                    for (int j=i+1; j<testData.length; j++) {
                        int aux=comparacion(testData[i],testData[j]);
			if((aux!=-2)&&(aux *(prediction[i][0]-prediction[j][0])<0))
                            antimontest=antimontest+1;
                    }
                }    
                antimontest=2*antimontest/(prediction.length*prediction.length-prediction.length);
                testTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
		//Writing results
//		writeOutput(outFile[1], realClass, prediction);	
//		System.out.println(name+" "+ relation + " Test " + testTime + "s");
		
//		printOutput();

	}//end-method 

//	/** 
//	 * Executes the classification of reference and test data sets
//	 * 
//	 */
//	public void executeReference(){
//		
//		modelTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
//		System.out.println(name+" "+ relation + " Model " + modelTime + "s");
//		
//		trainRealClass = new int[referenceData.length][1];
//		trainPrediction = new int[referenceData.length][1];	
//		
//		//Check  time		
//		setInitialTime();
//		
//		//Working on training
//		for (int i=0; i<trainRealClass.length; i++) {
//			trainRealClass[i][0] = referenceOutput[i];
//			trainPrediction[i][0]=evaluate(referenceData[i]);
//		}
//
//		trainingTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
//		
//		//Writing results
////		writeOutput(outFile[0], trainRealClass, trainPrediction);
////		System.out.println(name+" "+ relation + " Training " + trainingTime + "s");
//		
//		//Working on test
//		realClass = new int[testData.length][1];
//		prediction = new int[testData.length][1];	
//		
//		//Check  time		
//		setInitialTime();
//		
//		for (int i=0; i<realClass.length; i++) {
//			realClass[i][0] = testOutput[i];
//			prediction[i][0]=evaluate(testData[i]);
//		}
//
//		testTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
//		
//		//Writing results
////		writeOutput(outFile[1], realClass, prediction);	
////		System.out.println(name+" "+ relation + " Test " + testTime + "s");
//		
////		printOutput();
//		
//	}//end-method 
	
	/** 
	 * Evaluates a instance to predict its class. 
	 * Must be implemented in the subclass.
	 * 
	 * @param example Instance evaluated 
	 * @return The class predicted. -1 if the instance remains "Unclassified"
	 * 
	 */
	protected abstract int evaluate(double example[]);
	
	/** 
	 * Calculates the Euclidean distance between two instances
	 * 
	 * @param instance1 First instance 
	 * @param instance2 Second instance
	 * @return The Euclidean distance
	 * 
	 */
	protected double euclideanDistance(double instance1[],double instance2[]){
		
		double length=0.0;

		for (int i=0; i<instance1.length; i++) {
			length += (instance1[i]-instance2[i])*(instance1[i]-instance2[i]);
		}
			
		length = Math.sqrt(length); 
				
		return length;
		
	} //end-method
	
	/** 
	 * Calculates the Manhattan distance between two instances
	 * 
	 * @param instance1 First instance 
	 * @param instance2 Second instance
	 * @return The Euclidean distance
	 * 
	 */
	protected double manhattanDistance(double instance1[],double instance2[]){
		
		double length=0.0;

		for (int i=0; i<instance1.length; i++) {
			length += Math.abs(instance1[i]-instance2[i]);
		}
				
		return length;
		
	} //end-method
	
	/** 
	 * Checks if two instances are the same
	 * 
	 * @param a First instance 
	 * @param b Second instance
	 * @return True if both instances are equal.
	 * 
	 */
	protected boolean same(double a[],double b[]){

		for(int i=0;i<a.length;i++){
			if(a[i]!=b[i]){
				return false;
			}
		}
		
		return true;
		
	}//end-method 
	
	/** 
	 * Generates a string with the contents of the instance
	 * 
	 * @param instance Instance to print. 
	 * 
	 * @return A string, with the values of the instance
	 * 
	 */	
	public static String printInstance(int instance[]){
		
		String exit="";
		
		for(int i=0;i<instance.length;i++){
			exit+=instance[i]+" ";
		}
		
		return exit;
		
	}//end-method 
	
	/** 
	 * Sets the time counter
	 * 
	 */
	protected void setInitialTime(){
		
		initialTime = System.currentTimeMillis();
		
	}//end-method
	
//	/**
//	 * Prints output files.
//	 * 
//	 * @param filename Name of output file
//	 * @param realClass Real output of instances
//	 * @param prediction Predicted output for instances
//	 */
//	private void writeOutput(String filename, int [][] realClass, int [][] prediction) {
//	
//		String text = "";
//		
//		/*Printing input attributes*/
//		text += "@relation "+ relation +"\n";
//
//		for (int i=0; i<inputs.length; i++) {
//			
//			text += "@attribute "+ inputs[i].getName()+" ";
//			
//		    if (inputs[i].getType() == Attribute.NOMINAL) {
//		    	text += "{";
//		        for (int j=0; j<inputs[i].getNominalValuesList().size(); j++) {
//		        	text += (String)inputs[i].getNominalValuesList().elementAt(j);
//		        	if (j < inputs[i].getNominalValuesList().size() -1) {
//		        		text += ", ";
//		        	}
//		        }
//		        text += "}\n";
//		    } else {
//		    	if (inputs[i].getType() == Attribute.INTEGER) {
//		    		text += "integer";
//		        } else {
//		        	text += "real";
//		        }
//		        text += " ["+String.valueOf(inputs[i].getMinAttribute()) + ", " +  String.valueOf(inputs[i].getMaxAttribute())+"]\n";
//		    }
//		}
//
//		/*Printing output attribute*/
//		text += "@attribute "+ output.getName()+" ";
//
//		if (output.getType() == Attribute.NOMINAL) {
//			text += "{";
//			
//			for (int j=0; j<output.getNominalValuesList().size(); j++) {
//				text += (String)output.getNominalValuesList().elementAt(j);
//		        if (j < output.getNominalValuesList().size() -1) {
//		        	text += ", ";
//		        }
//			}		
//			text += "}\n";	    
//		} else {
//		    text += "integer ["+String.valueOf(output.getMinAttribute()) + ", " + String.valueOf(output.getMaxAttribute())+"]\n";
//		}
//
//		/*Printing data*/
//		text += "@data\n";
//
//		Files.writeFile(filename, text);
//		
//		if (output.getType() == Attribute.INTEGER) {
//			
//			text = "";
//			
//			for (int i=0; i<realClass.length; i++) {
//			      
//			      for (int j=0; j<realClass[0].length; j++){
//			    	  text += "" + realClass[i][j] + " ";
//			      }
//			      for (int j=0; j<realClass[0].length; j++){
//			    	  text += "" + prediction[i][j] + " ";
//			      }
//			      text += "\n";			      
//			      if((i%10)==9){
//			    	  Files.addToFile(filename, text);
//			    	  text = "";
//			      }     
//			}			
//			
//			if((realClass.length%10)!=0){
//				Files.addToFile(filename, text);
//			}
//		}
//		else{
//			
//			text = "";
//			
//			for (int i=0; i<realClass.length; i++) {
//			      
//			      for (int j=0; j<realClass[0].length; j++){
//			    	  text += "" + (String)output.getNominalValuesList().elementAt(realClass[i][j]) + " ";
//			      }
//			      for (int j=0; j<realClass[0].length; j++){
//			    	  if(prediction[i][j]>-1){
//			    		  text += "" + (String)output.getNominalValuesList().elementAt(prediction[i][j]) + " ";
//			    	  }
//			    	  else{
//			    		  text += "" + "Unclassified" + " ";
//			    	  }
//			      }
//			      text += "\n";
//			      
//			      if((i%10)==9){
//			    	  Files.addToFile(filename, text);
//			    	  text = "";
//			      } 
//			}			
//			
//			if((realClass.length%10)!=0){
//				Files.addToFile(filename, text);
//			}		
//		}
//		
//	}//end-method 
	
	/**
	 * Prints the additional output file
	 */
	public void printOutput(){
		
		double redIS;
		double redFS;
		double redIFS;
		int nRules;

		String text="";		
		
		computeConfussionMatrixes();
		
		//Accuracy
		text+="Accuracy: "+getAccuracy()+"\n";
		text+="Accuracy (Training): "+getTrainAccuracy()+"\n";
		
		//Kappa
		text+="Kappa: "+getKappa()+"\n";
		text+="Kappa (Training): "+getTrainKappa()+"\n";
		
		//Unclassified
		text+="Unclassified instances: "+unclassified+"\n";
		text+="Unclassified instances (Training): "+trainUnclassified+"\n";		
		
		//Reduction
		
		redIS=(1.0 - ( (double)trainData.length/(double)referenceData.length ));
		redFS=(1.0 - ( (double)inputAtt/(double)referenceData[0].length ));
		
		redIFS= (1.0 -   ( 
				( (double)trainData.length/(double)referenceData.length ) *   
				( (double)inputAtt/(double)referenceData[0].length      )
				));
		
		//Reduction IS	
		text+= "Reduction (instances): " +redIS+ "\n";
		
		//Reduction FS
		text+= "Reduction (features): "+redFS+"\n";
		
		//Reduction IFS
		text+= "Reduction (both): "+redIFS+"\n";
		
		//Model time
		text+= "Model time: "+modelTime+" s\n";
		
		//Training time
		text+= "Training time: "+trainingTime+" s\n";
		
		//Test time
		text+= "Test time: "+testTime+" s\n";
		
		//Number of rules
		nRules=writeRules();
		
		text+= "Number of rules: "+nRules+" \n";
                
                //Mae of test
                text+="MAE: "+Double.toString(this.computeMAE())+" \n";
                
                //Mae of train
                text+="MAE Training: "+Double.toString(this.computeTrainMAE())+" \n";
                
                //Monotonic index of the rule set
                text+="Index NonMonotonic DataTrain : "+Double.toString(this.antimontrain)+" \n";
                
                //Monotonic index of the rule set
                text+="Index NonMonotonic DataTest : "+Double.toString(this.antimontest)+" \n";
                
                //Confusion matrix
		text+="Confussion Matrix:\n";
		for(int i=0;i<nClasses;i++){
			
			for(int j=0;j<nClasses;j++){
				text+=confMatrix[i][j]+"\t";
			}
			text+="\n";
		}
		text+="\n";
		
		text+="Training Confussion Matrix:\n";
		for(int i=0;i<nClasses;i++){
			
			for(int j=0;j<nClasses;j++){
				text+=trainConfMatrix[i][j]+"\t";
			}
			text+="\n";
		}
		text+="\n";
		
		text+="***Rules obtained****";
		
		text+=ruleSetText;
		
		//Finish additional output file
//		Files.writeFile (outFile[2], text);
		System.out.println(text);
	}//end-method 

	/** 
	 * Writes the final ruleset obtained, in the ruleSetText variable.
	 * Must be implemented in the subclass.
	 * 
	 * @return The number of rules of the final rule set
	 */
	protected abstract int writeRules();
	
	/**
	 * Computes the confusion matrixes
	 * 
	 */
	private void computeConfussionMatrixes(){
		
		confMatrix= new int [nClasses][nClasses];
		trainConfMatrix= new int [nClasses][nClasses];
		
		unclassified=0;
		
		for(int i=0;i<nClasses;i++){
			Arrays.fill(confMatrix[i], 0);
		}
		
		for(int i=0;i<prediction.length;i++){
			if(prediction[i][0]==-1){
				unclassified++;
			}else{
				confMatrix[prediction[i][0]][realClass[i][0]]++;
			}
		}
		
		trainUnclassified=0;
		
		for(int i=0;i<nClasses;i++){
			Arrays.fill(trainConfMatrix[i], 0);
		}
		
		for(int i=0;i<trainPrediction.length;i++){
			if(trainPrediction[i][0]==-1){
				trainUnclassified++;
			}else{
				trainConfMatrix[trainPrediction[i][0]][trainRealClass[i][0]]++;
			}
		}
		
	}//end-method 
	
	/**
	 * Computes the accuracy obtained on test set
	 * 
	 * @return Accuracy on test set
	 */
	private double getAccuracy(){
		
		double acc;
		int count=0;
		
		for(int i=0;i<nClasses;i++){			
			count+=confMatrix[i][i];
		}
		
		acc=((double)count/(double)test.getNumInstances());
		
		return acc;
		
	}//end-method 
	
	/**
	 * Computes the accuracy obtained on the training set
	 * 
	 * @return Accuracy on test set
	 */
	private double getTrainAccuracy(){
		
		double acc;
		int count=0;
		
		for(int i=0;i<nClasses;i++){			
			count+=trainConfMatrix[i][i];
		}
		
		acc=((double)count/(double)train.getNumInstances());
		
		return acc;
		
	}//end-method 
	
	/**
	 * Computes the Kappa obtained on test set
	 * 
	 * @return Kappa on test set
	 */	
	private double getKappa(){
		
		double kappa;
		double agreement,expected;
		int count,count2;
		double prob1,prob2;
		
		count=0;
		for(int i=0;i<nClasses;i++){			
			count+=confMatrix[i][i];
		}
		
		agreement=((double)count/(double)test.getNumInstances());
		
		expected=0.0;
		
		for(int i=0;i<nClasses;i++){			
			
			count=0;
			count2=0;
			
			for(int j=0;j<nClasses;j++){
				count+=confMatrix[i][j];
				count2+=confMatrix[j][i];
			}
			
			prob1=((double)count/(double)test.getNumInstances());
			prob2=((double)count2/(double)test.getNumInstances());
			
			expected+=(prob1*prob2);
		}

		kappa=(agreement-expected)/(1.0-expected);
		
		return kappa;
		
	}//end-method 

	/**
	 * Computes the Kappa obtained on test set
	 * 
	 * @return Kappa on test set
	 */	
	private double getTrainKappa(){
		
		double kappa;
		double agreement,expected;
		int count,count2;
		double prob1,prob2;
		
		count=0;
		for(int i=0;i<nClasses;i++){			
			count+=trainConfMatrix[i][i];
		}
		
		agreement=((double)count/(double)train.getNumInstances());
		
		expected=0.0;
		
		for(int i=0;i<nClasses;i++){			
			
			count=0;
			count2=0;
			
			for(int j=0;j<nClasses;j++){
				count+=trainConfMatrix[i][j];
				count2+=trainConfMatrix[j][i];
			}
			
			prob1=((double)count/(double)train.getNumInstances());
			prob2=((double)count2/(double)train.getNumInstances());
			
			expected+=(prob1*prob2);
		}

		kappa=(agreement-expected)/(1.0-expected);
		
		return kappa;
		
	}//end-method 
        
        /**
	 * Computes the MAE
	 * 
	 */
	private double computeMAE(){
		
		MAE=0.0;
		for(int i=0;i<prediction.length;i++){
			if(prediction[i][0]==-1){
				unclassified++;
			}else{
				MAE=MAE+Math.abs(prediction[i][0]-realClass[i][0]);
			}
		}
                MAE=MAE/(double)test.getNumInstances();
                return MAE;
	}//end-method 

        /**
	 * Computes the MAE
	 * 
	 */
	private double computeTrainMAE(){
		
		trainMAE=0.0;
		for(int i=0;i<prediction.length;i++){
			if(prediction[i][0]==-1){
				unclassified++;
			}else{
				trainMAE=trainMAE+Math.abs(trainPrediction[i][0]-trainRealClass[i][0]);
                        }
		}
                trainMAE=trainMAE/(double)train.getNumInstances();
                return trainMAE;
	}//end-method 

        protected abstract double indexNonMonotonic();

        public int comparacion(double x[],double y[]){
                int numAtt = x.length;
                int comprobacion[];
                comprobacion = new int[3];
                for(int i=0;i<3;i++){
                    comprobacion[i]=0;
                }
                for(int i=0; i < numAtt; i++)
                {
                  if(x[i] < y[i]) 
                     comprobacion[2]++;
                  else if(x[i] == y[i]) 
                      comprobacion[1]++;
                  else
                      comprobacion[0]++;
                  if((comprobacion[0]>0)&&(comprobacion[2]>0))
                          return -2;
                }
                //Son iguales
                if (comprobacion[1]==numAtt)
                    return 0;

                else if ((comprobacion[2]+comprobacion[1])==numAtt)
                    return 1;
                else if ((comprobacion[0]+comprobacion[1])==numAtt)
                    return -1;
                else
                    return -2;


        }
}//end-class


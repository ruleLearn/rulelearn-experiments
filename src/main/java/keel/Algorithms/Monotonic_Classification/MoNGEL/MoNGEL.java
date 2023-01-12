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
 * File: MoNGEL.java
 *
 * The MoNGEL Algorithm.
 * The algorithm tries to show that is enough to consider a small neighbourhood
 * to achieve classification accuracy comparable to an algorithm considering the
 * whole learning set, combining this k-nearest neighbours method and a
 * rule-based algorithm.
 *
 * @author Written by Javier Garcia Fernandez (University of Jaen) 8/7/2014
 * @version 1.2
 * @since JDK1.5
 *
 */

package keel.Algorithms.Monotonic_Classification.MoNGEL;

import java.util.*;
import java.util.StringTokenizer;

import org.core.Fichero;
import org.core.Files;
import org.core.Randomize;
import keel.Dataset.Attribute;
import keel.Dataset.Attributes;
import keel.Algorithms.Monotonic_Classification.Basic.HyperrectanglesAlgorithm;
import keel.Dataset.InstanceSet;




public class MoNGEL extends HyperrectanglesAlgorithm{
	
    
        Rule ruleset[];
        int type_clasify=0;
        private int MatrixAntiMon[][];
        private int MaxRuleAntiMon[];
        
        private int numCover[];

	
	/** 
	 * The main method of the class
	 * 
	 * @param script Name of the configuration script  
	 * 
	 */
	public MoNGEL (String script) {
		
        initialTime = System.currentTimeMillis();
		
		readDataFiles(script);
		
		//Naming the algorithm
		name="MoNGEL";
			
		Rule.setSize(inputAtt);
		Rule.setAttributes(inputs);
		
		for(int i=0;i<inputAtt;i++){
			if(inputs[i].getType()==Attribute.NOMINAL){
				Rule.setNumValue(Attributes.getInputAttribute(i).getNumNominalValues(),i);
			}
			else{
				Rule.setNumValue(1,i);
			}
		}

		
		
		//Initialization of random generator
	    
	    Randomize.setSeed(seed);
		
		//Initialization stuff ends here. So, we can start time-counting
		
		setInitialTime(); 

	} //end-method 
	
	/** 
	 * Reads configuration script, to extract the parameter's values.
	 * 
	 * @param script Name of the configuration script  
	 * 
	 */	
    protected void readParameters (String script) {

        String file;
        String line;
        StringTokenizer fileLines, tokens;
        String mode;

        file = Fichero.leeFichero (script);
        fileLines = new StringTokenizer (file,"\n\r");

        //Discard in/out files definition
        fileLines.nextToken();
        fileLines.nextToken();
        fileLines.nextToken();

        //Getting the seed
        line = fileLines.nextToken();
        tokens = new StringTokenizer (line, "=");
        tokens.nextToken();
        seed = Long.parseLong(tokens.nextToken().substring(1));
        }//end-method
    
     /**
     * Extract the rules from the training set. This is the main part of the
     * MoNGEL algorithm.
     */
	public void initializeRules(){
		
            Rule newSet[];
            modelTime=System.currentTimeMillis();    
            this.nClasses=this.reference.getAttributeDefinitions().getOutputAttributes()[0].getNumNominalValues();
            ruleset=new Rule[this.referenceData.length];
            for(int i=0;i<referenceData.length;i++){			
                                    ruleset[i]=new Rule(referenceData[i],referenceOutput[i]);		
                    }
            // Hyperrectangles are ordered by the output
            List aux2 = new LinkedList();
            for(int k=0;k<ruleset.length;k++)
                aux2.add(ruleset[k]);
            Collections.sort(aux2);
            Collections.reverse(aux2);
            for(int k=0;k<ruleset.length;k++)
                ruleset[k]=(Rule)aux2.get(k);
            //Repeated rules are removed
            for(int i=0;i<ruleset.length;i++){
                    for(int j=i+1;j<ruleset.length;j++){
                        //If the rules are the same I remove one of them
                        if (ruleset[i].equals(ruleset[j])){
                            newSet=new Rule[ruleset.length-1];
                            System.arraycopy(ruleset, 0, newSet, 0, j);
                            System.arraycopy(ruleset, j+1, newSet, j, (ruleset.length-j-1));
                            ruleset=new Rule[newSet.length];
                            System.arraycopy(newSet, 0, ruleset, 0, newSet.length);
                            j--;
                        }
                    }
            }        
        }
    
    
    /**
     * Extract the rules from the training set. This is the main part of the
     * MoNGEL algorithm.
     */
	public void getRules(){
		
		boolean canMerge,testing;
		int indexMerge,row;
		double distMerge,auxDist;
		Rule newRule;
		Rule newSet [];
        //merging process
		canMerge=true;
        while(canMerge){
			canMerge=false;
		    for(int i=0;i<ruleset.length&& !canMerge;i++){
				//find their nearest hyperrectangle
				indexMerge=-1;
				distMerge=Double.MAX_VALUE;
				for(int j=i+1;j<ruleset.length;j++){
							if((ruleset[i].getOutput()==ruleset[j].getOutput())&&(ruleset[i].compareInput(ruleset[j])!=-2)){
							auxDist=ruleset[i].distanceRule(ruleset[j]);
							if(distMerge>auxDist){
									distMerge=auxDist;
									indexMerge=j;
							}
					}
				}
				//try to merge
				if(indexMerge>-1){
					newRule=ruleset[i].clone();					
					newRule.merge(ruleset[indexMerge]);
					testing=true;
					for(int j=0;j<ruleset.length&&testing;j++){
							if((j!=i)&&(j!=indexMerge)&&(newRule.getOutput()!=ruleset[j].getOutput())){
									if(newRule.overlap(ruleset[j])){
											testing=false;
									}				
							}
					}
					if(testing){

							ruleset[i]=newRule.clone();

							newSet=new Rule[ruleset.length-1];

							System.arraycopy(ruleset, 0, newSet, 0, indexMerge);
							System.arraycopy(ruleset, indexMerge+1, newSet, indexMerge, (ruleset.length-indexMerge-1));
							ruleset=new Rule[newSet.length];
							System.arraycopy(newSet, 0, ruleset, 0, newSet.length);
							canMerge=true;
					}
				}
			}
		}
		matrix_AntiMonotonic();
		NumInstancesCov();
		while(SumMaxAntMon()>0){
			row=MatrixRowMax();
			newSet=new Rule[ruleset.length-1];
			System.arraycopy(ruleset, 0, newSet, 0, row);
			System.arraycopy(ruleset, row+1, newSet, row, ruleset.length-row-1);
			ruleset=new Rule[newSet.length];
			System.arraycopy(newSet, 0, ruleset, 0, newSet.length);
			DeleteRowColumn(row);
		}
		modelTime=System.currentTimeMillis()-modelTime;         
	}//end-method
    
    /**
	 * Classifies an instance using the ruleset
	 *
     * @param instance Instance to classify
     * @return Class assigned to the instance
	 */
	protected int evaluate(double instance[]){
		
        if(type_clasify==0){
                double minArea=Double.MAX_VALUE;
		double minDist=Double.MAX_VALUE;
		int selected=-1;
		
		for(int i=0;i<ruleset.length;i++){
                       if(ruleset[i].distance(instance)==minDist){
				if(ruleset[i].getArea()<minArea){
					minArea=ruleset[i].getArea();
					selected=i;
				}
			}
			if(ruleset[i].distance(instance)<minDist){
				minDist=ruleset[i].distance(instance);
				minArea=ruleset[i].getArea();
				selected=i;
			}
               }		
		selected=ruleset[selected].getOutput();
		return selected;
            }
            
            else{
                for(int i=0; i < ruleset.length; i++){
                    if(ruleset[i].compareInput(new Rule(instance,1))>0){
                              return (ruleset[i].getOutput());
                    }
                }
                return (ruleset[ruleset.length-1].getOutput());
            }

	}//end-method
    
	/** 
	 * Writes the final ruleset obtained, in the ruleSetText variable.
	 * 
	 * @return The number of rules of the final rule set
	 */
	protected int writeRules(){
		
		String text="";
		
		text+="\n";
		
		for(int i=0;i<ruleset.length;i++){
			text+="\n";
			text+=ruleset[i];
		}
		
		ruleSetText=text;
		
		return ruleset.length;
	}//end-method
        
        /**
	 * Prints output files.
	 * 
	 * @param filename Name of output file
	 * @param realClass Real output of instances
	 * @param prediction Predicted output for instances
	 */
	private void writeOutput(String filename, int [][] realClass, int [][] prediction) {
	
		String text = "";
		
		/*Printing input attributes*/
		text += "@relation "+ relation +"\n";

		for (int i=0; i<inputs.length; i++) {
			
			text += "@attribute "+ inputs[i].getName()+" ";
			
		    if (inputs[i].getType() == Attribute.NOMINAL) {
		    	text += "{";
		        for (int j=0; j<inputs[i].getNominalValuesList().size(); j++) {
		        	text += (String)inputs[i].getNominalValuesList().elementAt(j);
		        	if (j < inputs[i].getNominalValuesList().size() -1) {
		        		text += ", ";
		        	}
		        }
		        text += "}\n";
		    } else {
		    	if (inputs[i].getType() == Attribute.INTEGER) {
		    		text += "integer";
		        } else {
		        	text += "real";
		        }
		        text += " ["+String.valueOf(inputs[i].getMinAttribute()) + ", " +  String.valueOf(inputs[i].getMaxAttribute())+"]\n";
		    }
		}

		/*Printing output attribute*/
		text += "@attribute "+ output.getName()+" ";

		if (output.getType() == Attribute.NOMINAL) {
			text += "{";
			
			for (int j=0; j<output.getNominalValuesList().size(); j++) {
				text += (String)output.getNominalValuesList().elementAt(j);
		        if (j < output.getNominalValuesList().size() -1) {
		        	text += ", ";
		        }
			}		
			text += "}\n";	    
		} else {
		    text += "integer ["+String.valueOf(output.getMinAttribute()) + ", " + String.valueOf(output.getMaxAttribute())+"]\n";
		}

		/*Printing data*/
		text += "@data\n";

		Files.writeFile(filename, text);
		
		if (output.getType() == Attribute.INTEGER) {
			
			text = "";
			
			for (int i=0; i<realClass.length; i++) {
			      
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + realClass[i][j] + " ";
			      }
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + prediction[i][j] + " ";
			      }
			      text += "\n";			      
			      if((i%10)==9){
			    	  Files.addToFile(filename, text);
			    	  text = "";
			      }     
			}			
			
			if((realClass.length%10)!=0){
				Files.addToFile(filename, text);
			}
		}
		else{
			
			text = "";
			
			for (int i=0; i<realClass.length; i++) {
			      
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + (String)output.getNominalValuesList().elementAt(realClass[i][j]) + " ";
			      }
			      for (int j=0; j<realClass[0].length; j++){
			    	  if(prediction[i][j]>-1){
			    		  text += "" + (String)output.getNominalValuesList().elementAt(prediction[i][j]) + " ";
			    	  }
			    	  else{
			    		  text += "" + "Unclassified" + " ";
			    	  }
			      }
			      text += "\n";
			      
			      if((i%10)==9){
			    	  Files.addToFile(filename, text);
			    	  text = "";
			      } 
			}			
			
			if((realClass.length%10)!=0){
				Files.addToFile(filename, text);
			}		
		}
		
	}//end-method 
        
	 /**
	 * Calculate the non monotonicity index of the rule set.
	 * 	
	 * @return The non monotonicity index of the rule set
	 */	
		
	public  double indexNonMonotonic(){
        double count=0.0;
        int countPairs=0;
        for(int i=0;i<this.ruleset.length;i++){
            for(int j=i+1;j<this.ruleset.length;j++){
                countPairs++;
                if (ruleset[i].isAntiMonotonic(ruleset[j]))
                    count=count+1.0;
            }           
        }
        return (count/(double)countPairs);
    };
    
	 /**
	 * Calculate the anti monotonicity matrix used to calculate the anti monotonic index.
	 * 	
	 */	
	void matrix_AntiMonotonic(){
        
		MatrixAntiMon=new int[this.ruleset.length][this.ruleset.length];
		MaxRuleAntiMon=new int[this.ruleset.length];
		Arrays.fill(MaxRuleAntiMon,0);
		for(int i=0;i<this.ruleset.length;i++)
			Arrays.fill(MatrixAntiMon[i],0);
		for(int i=0;i<this.ruleset.length;i++)
			for(int j=i+1;j<this.ruleset.length;j++)
				if(ruleset[i].isAntiMonotonic(ruleset[j])){
					 MatrixAntiMon[i][j]=1;
					 MatrixAntiMon[j][i]=1;
					 MaxRuleAntiMon[i]=MaxRuleAntiMon[i]+1;
					 MaxRuleAntiMon[j]=MaxRuleAntiMon[j]+1;
				}
    }
    
	/**
	 *  Calculate the row with the rule maximal anti monotonic value.
	 * 	
	 * @return The number of the row with the maximal anti monotonic value
	 */	
	    
	public int MatrixRowMax(){
		int max=0;
		int row=-1;
		for(int j=0;j<ruleset.length;j++)
			if(MaxRuleAntiMon[j]>max){
				row=j;
				max=MaxRuleAntiMon[j];
			}
			else if((MaxRuleAntiMon[j]!=0)&&(MaxRuleAntiMon[j]==max)){
				if(this.numCover[j]<this.numCover[row])
					row=j;
			}
		return row;
	}
	
	/**
	 *  Calculate the sum of the maximal anti monotonic values of the rule set.
	 * 	
	 * @return The sum of the maximal anti monotonic values of the rule set
	 */	
	public int SumMaxAntMon(){
		int sum=0;
		for(int j=0;j<ruleset.length;j++)
				sum=sum+MaxRuleAntiMon[j];
		return sum;
	}
	
	/**
	 *  Delete row column.
	 * 	
	 * @param The number of the row to be removed
	 */	
	
	void DeleteRowColumn(int i){
		int aux[][]=new int[this.MatrixAntiMon.length-1][this.MatrixAntiMon.length-1];
		int aux2[]=new int[this.MatrixAntiMon.length-1]; 
		int aux3[]=new int[this.MatrixAntiMon.length-1]; 
		int h=0,m=0;
		for(int j=0;j<this.MatrixAntiMon.length-1;j++){
			if(j==i)
				h++;
			aux2[j]=this.MaxRuleAntiMon[h]-MatrixAntiMon[h][i];
			aux3[j]=this.numCover[h];
			m=j+1;
			for(int k=j+1;k<this.MatrixAntiMon.length-1;k++){
				if(k==i)
					m++;
				aux[j][k]=this.MatrixAntiMon[h][m];
				m++;
			}
			h++;
		}
		this.MatrixAntiMon=aux;
		this.MaxRuleAntiMon=aux2;
		this.numCover=aux3;
	}
	
	/**
	 *  Calculates the number of instances covered
	 * 	
	 */		
	
	void NumInstancesCov(){
		numCover=new int[ruleset.length];
		Arrays.fill(numCover, 0);
		for(int i=0;i<ruleset.length;i++)
			for(int j=0;j<trainData.length;j++){
				if (ruleset[i].distance(trainData[j])==0)
					 numCover[i]++;
				
			}        
	}
	
	/**
	 *  Calculate the row with the rule minimal anti monotonic value.
	 * 	
	 * @return The number of the row with the minimal anti monotonic value
	 */	
	
	int MinRow(){
		int min=trainData.length+1;
		int row=-1;
		for(int j=0;j<ruleset.length;j++)
			if((this.MaxRuleAntiMon[j]>0)&(numCover[j]<min)){
				row=j;
				min=numCover[j];
			}    
		return row;
	
	
	}
	
	public InstanceSet gettrain(){return train;};
	public InstanceSet gettest(){return test;};
	public InstanceSet reference(){return reference;};  
	public int numRule(){
		return this.ruleset.length;
	}
	
	public Rule[] gettrule(){return this.ruleset;};  
	
	public String[] getfichsalidas(){return this.outFile;};  
        
} //end-class 


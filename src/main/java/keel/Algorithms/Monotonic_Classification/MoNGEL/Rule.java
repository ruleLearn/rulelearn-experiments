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
 * File: Rule.java
 *
 * Auxiliary class to repressent rules for the MoNGEL algorithm
 *
 * @author Written by Joaquin Derrac (University of Granada) 8/7/2009
 * @version 1.1
 * @since JDK1.5
 *
 *
 * @modified by Javier García (University of Jaén)  30/05/2015
 */
package keel.Algorithms.Monotonic_Classification.MoNGEL;

import keel.Dataset.Attribute;

public class Rule implements Comparable<Rule> {
	
	private int size;            //number of input attributes of the rule
	private boolean isNominal[]; //nominal attributes
	private int nValues[];       //different values in nominal attributes
	
	private double valueMin[];          //for numeric attributes
	private double valueMax[];          //for numeric attributes
	private boolean valueNom[][];       //for nominal attributes
	
	private double area;                //area of the rule
	
	private int output;                 //output attribute
        
	Attribute[] inputs; //input attributes //MSz
	
	/**
     * Sets the size of the rule
     *
     * @param value Number of attributes of the rule
     *
     */
	public void setSize(int value){
		
		size=value;
		nValues=new int[size];
        
	}//end-method

    /**
     * Test which attributes are nominal
     *
     * @param inputs Attributes' descriptions
     *
     */
	public void testForNominalAttributes(Attribute[] inputs){
		
		isNominal=new boolean[size];
		
		for(int i=0;i<size;i++){
			if(inputs[i].getType()==Attribute.NOMINAL){
				isNominal[i]=true;
			}
			else{
				isNominal[i]=false;				
			}
		}
		
	}//end-method

	/**
     * Sets the number of different values for an attribute
     *
     * @param value Number of values
     * @param pos Index of the attribute
     *
     */
	public void setNumValue(int value,int pos){
		
		nValues[pos]=value;
		
	}//end-method

	/**
     * Default builder. Generates a void rule.
     * 
     * @param attributes the attributes for which this rule is generated
     */
	public Rule(Attribute[] inputAttributes){ //MSz: added inputAttributes
		//***** BEGIN MSz fix *****
		this.inputs = inputAttributes;
		setSize(inputAttributes.length);
		testForNominalAttributes(inputAttributes);
		
		for (int i = 0; i < size; i++) {
			if (inputs[i].getType() == Attribute.NOMINAL) {
				setNumValue(inputs[i].getNumNominalValues(), i);
			} else {
				setNumValue(1, i);
			}
		}
		//***** END MSz fix *****
		
		valueMin=new double[size];
		valueMax=new double[size];
		valueNom=new boolean[size][];	
		
		for(int i=0;i<size;i++){
			valueNom[i]=new boolean [nValues[i]];
			for(int j=0;j<nValues[i];j++){
				valueNom[i][j]=false;
			}
		}
			
		output=-1;
		area=0;

	}//end-method

    /**
     * Builder. Generates a rule covering only a point
     *
     * @param instance Basic instance
     * @param out Ouput of the instance
     * @param attributes the attributes for which this rule is generated
     */
	public Rule(double instance[], int out, Attribute[] inputAttributes){ //MSz: added inputAttributes
		//***** BEGIN MSz fix *****
		this.inputs = inputAttributes;
		setSize(inputAttributes.length);
		testForNominalAttributes(inputAttributes);
		
		for (int i = 0; i < size; i++) {
			if (inputs[i].getType() == Attribute.NOMINAL) {
				setNumValue(inputs[i].getNumNominalValues(), i);
			} else {
				setNumValue(1, i);
			}
		}
		//***** END MSz fix *****
		
		int nomRep;
		
		valueMin=new double[size];
		valueMax=new double[size];
		valueNom=new boolean[size][];	
		
		for(int i=0;i<size;i++){
			valueNom[i]=new boolean [nValues[i]];
			for(int j=0;j<nValues[i];j++){
				valueNom[i][j]=false;
			}
		}

		for(int i=0;i<size;i++){
			if(isNominal[i]){
				nomRep=(int)(instance[i]*(nValues[i]-1));
				valueNom[i][nomRep]=true;
			}
			else{
				valueMax[i]=instance[i];
				valueMin[i]=instance[i];				
			}
		}
		
		output=out;
		
		computeArea();

	}//end-method

    /**
     * Reinitialices a rule, loading it with the contents of a single instance
     *
     * @param instance Basic instance
     * @param out Ouput of the instance
     */
	public void loadRule(double instance[],int out){
		
		int nomRep;
	
		for(int i=0;i<size;i++){
			if(isNominal[i]){
				nomRep=(int)(instance[i]*(nValues[i]-1));
				valueNom[i][nomRep]=true;
			}
			else{
				valueMax[i]=instance[i];
				valueMin[i]=instance[i];				
			}
		}
		
		output=out;
		
		computeArea();

	}//end-method

    /**
     * Computes the area of the rule
     */
	private void computeArea(){
		
		int count;
		
		area=0.0;
		
		for(int i=0;i<size;i++){
			if(isNominal[i]){
				count=0;
				for(int j=0;j<valueNom[i].length;j++){
					if(valueNom[i][j]){
						count++;
					}
				}
				area+=(double)((double)count/(double)valueNom[i].length);
			}
			else{
				area+=valueMax[i]-valueMin[i];				
			}			
		}
		
	}//end-method
     
	@Override
	public int compareTo(Rule other){
//            if(!(other instanceof Rule))
//                throw new ClassCastException("Invalid object in Rule:compareTo");
            if(this.getOutput()>((Rule)other).getOutput())
                return 1;
            else if(this.getOutput()<((Rule)other).getOutput())
                return -1;
            else
                return 0;
        }

    /**
     * Clone method
     *
     * @return A intialized copy of the rule
     */
	public Rule clone(){
		
		Rule clon=new Rule(inputs);	
		
		for(int i=0;i<size;i++){
			if(isNominal[i]){
				System.arraycopy(valueNom[i], 0, clon.valueNom[i], 0, nValues[i]);
			}
			else{
				clon.valueMax[i]=valueMax[i];
				clon.valueMin[i]=valueMin[i];				
			}
		}
		
		clon.output=output;
		clon.area=area;
		
		return clon;

	}//end-method

    /**
     * Equals method
     *
     * @param rul Another rule
     * @return True of both rules are equal. False, if not
     */
	@Override
	public boolean equals(Object rul) {
		
		Rule another=(Rule)rul;
		
		boolean isEqual=true;
		
		if(output!=another.output){
			isEqual=false;
		}
		
		if(area!=another.area){
			isEqual=false;
		}
			
		for(int i=0;i<size && isEqual;i++){
			if(isNominal[i]){
				for(int j=0;j<nValues[i]&& isEqual;j++){
					if(valueNom[i][j]!=another.valueNom[i][j]){
						isEqual=false;
					}
				}
			}
			else{
				if(valueMin[i]!=another.valueMin[i]){
					isEqual=false;
				}
				if(valueMax[i]!=another.valueMax[i]){
					isEqual=false;
				}
			}
		}
		
		return isEqual;

	}//end-method

    /**
     * To String method
     *
     * @return A text string representing the contents of the rule
     */
	@Override
	public String toString() {

		String text="";
                for(int i=0;i<size;i++){
			
			if(isNominal[i]){
                                int numelem=0;
                                text+="{";
				for(int j=0;j<nValues[i]-1;j++){
			              if(this.valueNom[i][j]==true){
                                          if(numelem>0)
                                              text+=",";    
                                          text+=inputs[i].getNominalValue(j);
                                          numelem++;
                                      }  
				}
                                if(this.valueNom[i][nValues[i]-1]==true){
                                    if(numelem>0)
                                            text+=",";    
                                    text+=inputs[i].getNominalValue(nValues[i]-1);
                                }    
                                text+="}";
			}
			else{
                                String aux=Double.toString(desnormalize(valueMin[i],i)); 
				text+="["+aux.substring(0,Math.min(6,aux.length()))+"-";
                                aux=Double.toString(desnormalize(valueMax[i],i)); 
				text+=aux.substring(0,Math.min(6,aux.length()))+"] ";				
                                while (text.length()<16*(i+1))
                                    text+=" ";
			}
		}
		
		text+="Output= "+output;
                String aux=Double.toString(area);
		text+=" Area: "+aux.substring(0,Math.min(6,aux.length()));
		
		return text;

	}//end-method

     
        
    /**
     * Returns the output class of the rule
     *
     * @return Output class of the rule
     */
	public int getOutput(){
		
		return output;

	}//end-method

    /**
     * Returns the area of the rule
     *
     * @return Area of the rule
     */
	public double getArea(){
		
		return area;

	}//end-method
	
    /**
     * Computes the distance between a given instance and the rule.
     *
     * @param instance Instance to be tested
     *
     * @return Distance computed
     */
	public double distance(double instance[]){
		
		double dist=0.0;
		double inc;
		int nomRep;
		
		for(int i=0;i<size;i++){
			if(isNominal[i]){
				nomRep=(int)(instance[i]*(nValues[i]-1));
				if(valueNom[i][nomRep]==false){
					dist+=1.0;
				}
			}
			else{
				if(instance[i]<valueMin[i]){
					inc=(valueMin[i]-instance[i]);
					dist+=(inc*inc);
				}
				if(instance[i]>valueMax[i]){
					inc=(instance[i]-valueMax[i]);
					dist+=(inc*inc);
				}
			}

		}	
		
		return dist;

	}//end-method

    /**
     * Computes the distance between two rules.
     *
     * @param another Second rule to be tested
     *
     * @return Distance computed
     */
	public double distanceRule(Rule another){
		
		double dist=0.0;
		int count;
		double inc;
		double a,b;
		
		for(int i=0;i<size;i++){
			if(isNominal[i]){ //compute the proportion of examples which differs
				count=0;
				for(int j=0;j<nValues[i];j++){
					if(valueNom[i][j]!=another.valueNom[i][j]){
						count++;
					}
				}
				inc=(double)((double)count/(double)nValues[i]);
			}
			else{ //compute distance between the centroids

				a=(valueMax[i]+valueMin[i])/2.0;
				b=(another.valueMax[i]+another.valueMin[i])/2.0;
                            if(a>b){
					inc=a-b;
				}
				else{
					inc=b-a;
				}

			}
			dist+=inc*inc;
		}	
		
		return dist;

	}//end-method

      /**
     * Compares the input of two rules.
     *
     * @param another Second rule to be compared
     *
     * @return Result of the comparison
     */
    public int compareInput(Rule other){
      int numAtt = other.size;
      int testing[];
      testing = new int[3];
      for(int i=0;i<3;i++)
          testing[i]=0;
      for(int i=0; i < numAtt; i++){
         if(this.isNominal[i]==true){
            int cont1=0,cont2=0,contcomunes=0;
            for(int j=0;j<other.valueNom[i].length;j++){
                if(valueNom[i][j]==true){
                     cont1++;
                    if(other.valueNom[i][j]==true){
                        cont2++;
                        contcomunes++;
                    }
                }
                else if(other.valueNom[i][j]==true)
                    cont2++;
            }
            //Are equal
            if((cont1==cont2)&&(contcomunes==cont1))
                testing[1]++;
            //It is contained in another
            else if (cont1==contcomunes)
                testing[0]++;
            else if (cont2==contcomunes)
                testing[2]++;
            else
                return -2;
        }
         else if((other.valueMax[i] < valueMin[i])&&(this.isNominal[i]==false)) 
           testing[2]++;
        else if((valueMax[i]<other.valueMin[i])&&(this.isNominal[i]==false)) 
            testing[0]++;
	else if((other.valueMax[i] == valueMax[i])&&(other.valueMin[i] == valueMin[i])&&(this.isNominal[i]==false)) 
            testing[1]++;
        if((testing[0]>0)&&(testing[2]>0))
                return -2;
      }
      //Are equal
      if (testing[1]==numAtt)
          return 0;
      else if ((testing[2]+testing[1])==numAtt)
          return 1;
      else if ((testing[0]+testing[1])==numAtt)
          return -1;
      else
          return -2;
    }

    
      /**
     * Compares the output of two rules.
     *
     * @param another Second rule to be compared
     *
     * @return Result of the comparison
     */
    
    private int CompareOutput(Rule other)
    {
      if (this.output>other.output)
        return 1;
      else if (this.output==other.output)
          return 0;
      else 
      return -1;
    }

     /**
     * Compares if the rule is Monotonic in relation to the parameter Rule.
     *
     * @param another Second rule to be compared
     *
     * @return Result of the comparison
     */
    boolean isMonotonic(Rule other){
        
        if ((Math.abs(this.compareInput(other))<=1)&&(this.compareInput(other)*(this.output-other.output)>=0))
            return true;
       else
            return false;
    
    }
   
   
    /**
     * Compares if the rule is AntiMonotonic in relation to the parameter Rule.
     *
     * @param another Second rule to be compared
     *
     * @return Result of the comparison
     */   
    boolean isAntiMonotonic(Rule other){
        

        if ((Math.abs(this.compareInput(other))==1)&&(this.compareInput(other)*(this.output-other.output)<0))
            return true;
        else if ((Math.abs(this.compareInput(other))==0)&&(this.output!=other.output))
            return true;    
       else
            return false;
    
    }
    
	   /**
     * Compares if the rule is AntiMonotonic in relation to the parameter Rules.
     *
     * @param another Second rule to be compared
     *
     * @return Result of the comparison
     */ 
    boolean isAntiMonotonic(Rule other[]){
            for (Rule other1 : other) {
                if (this.isAntiMonotonic(other1)) {
                    return true;
                }
            }
        return false;
    }
    
    
    
    
     /**
     * Test if two rules are overlapped
     *
     * @param another Second rule to test
     *
     * @return True if the rules are overlapped. False, if not.
     */
	public boolean overlap(Rule another){
		
		boolean over=true;
		boolean test;
		
		for(int i=0;i<size && over;i++){
			if(isNominal[i]){
				test=false;
				for(int j=0;j<nValues[i]&&!test;j++){
					if((valueNom[i][j]==true)&&(another.valueNom[i][j]==true)){
						test=true;
					}
				}
				if(!test){
					over=false;
				}
			}
			else{
				test=false;
				//left overlap
				if((another.valueMax[i]>=valueMin[i])&&(another.valueMax[i]<=valueMax[i])){
					test=true;
				}
				else{
					//right overlap
					if((valueMax[i]>=another.valueMin[i])&&(valueMax[i]<=another.valueMax[i])){
						test=true;
					}							
				}
				if(!test){
					over=false;
				}	
			}
		}

		return over;

	}//end-method

    /**
     * Merge two rules
     *
     * @param another Second rule to merge
     */
	public void merge(Rule another){

		for(int i=0;i<size;i++){
			if(isNominal[i]){
				for(int j=0;j<nValues[i];j++){
					if(another.valueNom[i][j]==true){
						valueNom[i][j]=true;
					}
				}
			}
			else{
				valueMin[i]=Math.min(valueMin[i],another.valueMin[i]);	
				valueMax[i]=Math.max(valueMax[i],another.valueMax[i]);	
			}
		}
                output=Math.max(output,another.output);
		computeArea();

	}//end-method
        
     /** 
	 * This function builds the data matrix for test data and normalizes inputs values
	 */	
	protected double desnormalize(double Z,int j){

	    double minimum;
	    double range,X=0;

	    //Normalize the data
	    
	    
	    if (inputs[j].getType() != Attribute.NOMINAL) {
		    	minimum=inputs[j].getMinAttribute();
		    	range=inputs[j].getMaxAttribute()-minimum;
                        X= Z*range;
                        X=  X+ minimum;
            }
	    
	    //Both real and nominal data are desnormalized in [0,1]
	    
	    
            else  {
                if(inputs[j].getNominalValuesList().size()>1)
                        X = Z* (inputs[j].getNominalValuesList().size()-1);
            }
            return X;
      }
        
        public int getSize(){return size;};   
        
        public double[] getMins(){return valueMin;};
        
        public double[] getMaxs(){return valueMax;};
        
        public boolean[][] getNominal(){return valueNom;};

}//end-class



/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Objects;

import org.rulelearn.core.UnknownValueException;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;

import weka.core.Instances;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class Data {
	
	InformationTable data;
	Instances instances = null; //not always used - calculated only when getter is invoked for the first time
	String name;
	String groupName; //name of a group of different data sets that this data belongs to (data sets within a group may differ, e.g., by missing values or different columns, but should contain the same objects!)
	long seed;
	boolean hasSeed = false;
	long informationTableTransformationTime = 0L; //time of transformation from InformationTable to InformationTableWithDecisionDistributions, if the information table contains decision distributions
	
	Data(InformationTable data, String name, String groupName, long seed) {
		this.data = data;
		this.name = name;
		this.groupName = groupName;
		this.seed = seed;
		this.hasSeed = true;
	}
	
	Data(InformationTableWithDecisionDistributions data, String name, String groupName, long seed, long informationTableTransformationTime) {
		this.data = data;
		this.name = name;
		this.groupName = groupName;
		this.seed = seed;
		this.hasSeed = true;
		this.informationTableTransformationTime = informationTableTransformationTime;
	}
	
	Data(InformationTable data, String name, String groupName) {
		this.data = data;
		this.name = name;
		this.groupName = groupName;
		this.hasSeed = false;
	}
	
	Data(InformationTableWithDecisionDistributions data, String name, String groupName, long informationTableTransformationTime) {
		this.data = data;
		this.name = name;
		this.groupName = groupName;
		this.hasSeed = false;
		this.informationTableTransformationTime = informationTableTransformationTime;
	}
	
	public InformationTable getInformationTable() {
		return data;
	}
	
	//SIC! replaces data with new reference
	//next call to getInformationTable() will in fact return an instance of InformationTableWithDecisionDistributions!
	public void extendInformationTableWithDecisionDistributions() { //does not check if the information table already contains decision distributions
		long start = System.currentTimeMillis();
		data = new InformationTableWithDecisionDistributions(data, true, true);
		this.informationTableTransformationTime = System.currentTimeMillis() - start;
	}
	
	public Instances getInstances() { //builds instances on the first call
		if (instances == null) {
			instances = InformationTable2Instances.convert(data, name);
		}
		return instances;
	}
	
	public String getName() {
		return name;
	}
	
	public String getGroupName() {
		return groupName;
	}
	
	public long getSeed() {
		if (hasSeed) {
			return seed;
		} else {
			throw new UnknownValueException("Data do not store a cross-validation seed.");
		}
	}
	
	public long getInformationTableTransformationTime() {
		return informationTableTransformationTime;
	}

	@Override
	public int hashCode() {
		return Objects.hash(this.getClass(), getName());
	}
	
	@Override
	public boolean equals(Object other) {
		return (other instanceof Data) && getName().equals(((Data)other).getName());
	}
	
}

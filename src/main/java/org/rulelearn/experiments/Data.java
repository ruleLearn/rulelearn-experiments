/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Objects;

import org.rulelearn.core.UnknownValueException;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class Data {
	
	InformationTable data;
	String name;
	long seed;
	boolean hasSeed = false;
	
	Data(InformationTable data, String name, long seed) {
		this.data = data;
		this.name = name;
		this.seed = seed;
		this.hasSeed = true;
	}
	
	Data(InformationTable data, String name) {
		this.data = data;
		this.name = name;
		this.hasSeed = false;
	}
	
	public InformationTable getInformationTable() {
		return data;
	}
	
	//SIC! replaces data with new reference
	//next call to getInformationTable() with in fact return an instance of InformationTableWithDecisionDistributions!
	public void etendInformationTableWithDecisionDistributions() {
		data = new InformationTableWithDecisionDistributions(data);
	}
	
	public String getName() {
		return name;
	}
	
	public long getSeed() {
		if (hasSeed) {
			return seed;
		} else {
			throw new UnknownValueException("Full data do not store a cross-validation seed.");
		}
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

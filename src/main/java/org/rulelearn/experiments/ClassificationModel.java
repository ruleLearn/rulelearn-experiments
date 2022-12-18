/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.data.SimpleDecision;

/**
 * Classification model learned from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface ClassificationModel {
	
	public ModelValidationResult validate(Data testData);
	SimpleDecision classify(int i, Data data); //gets simple decision of a single object from data
	String getModelDescription();
	String getModelLearnerDescription();
	String getValidationSummary();
	
}

/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Classification model learned from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface ClassificationModel {
	
	public ModelValidationResult validate(Data testData);
	String getModelDescription();
	String getValidationSummary();
	
}

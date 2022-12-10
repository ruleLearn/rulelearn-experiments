/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Classification model learned from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface ClassificationModel {
	
	public OrdinalMisclassificationMatrix validate(Data testData);
	String getModelDescription();
	String getValidationSummary();
	
}

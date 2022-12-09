/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Cross validation fold.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface CrossValidationFold {
	Data getTrainData();
	Data getTestData();
	int getIndex();
	void done(); //called when processing of fold is done - can be used to free resources
}

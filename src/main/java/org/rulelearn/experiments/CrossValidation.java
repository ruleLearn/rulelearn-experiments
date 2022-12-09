/**
 * 
 */
package org.rulelearn.experiments;

import java.util.List;

/**
 * K-fold cross validation.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface CrossValidation {
	void setSeed(long seed);
	long getSeed();
	void setNumberOfFolds(int k);
	int getNumberOfFolds();
	List<CrossValidationFold> getStratifiedFolds(Data data); //should calculate stratified folds each time this method is called (no caching, as this method is called only once!)
}

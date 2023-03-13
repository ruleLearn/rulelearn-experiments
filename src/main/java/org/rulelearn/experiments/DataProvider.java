/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Provides {@Data data} for each cross-validation (with seed concerning that cross-validation),
 * seeds for subsequent cross-validations, and number of folds in each cross-validation.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface DataProvider {
	
	Data provide(int crossValidationNumber); //provides data for particular cross-validation (with seed concerning that cross-validation)
	Data provideOriginalData(); //provides original data
	Data previewOriginalData(); //previews original data - returns Data but does not store information table neither calculates its decision distributions
	String getDataName();
	String getDataGroup();
	long[] getSeeds(); //determines number of cross-validations
	int getNumberOfFolds(); //number of folds in each cross-validation
	public void done(); //called when processing of fold is done - can be used to free resources
	public void reset(); //frees resources by setting this provider to its initial state (the state once the provider object has been created)
}

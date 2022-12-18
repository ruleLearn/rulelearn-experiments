/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.sampling.CrossValidator;

/**
 * Cross validation parameterized by a seed and a number of folds, always giving the same folds for the same data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class RepeatableCrossValidation implements CrossValidation {
	
	long seed;
	int k = -1;
	boolean seedSet = false;
	boolean kSet = false;

	@Override
	public long getSeed() {
		return seed;
	}
	@Override
	public void setSeed(long seed) {
		this.seed = seed;
		seedSet = true;
	}
	@Override
	public int getNumberOfFolds() {
		return k;
	}
	@Override
	public void setNumberOfFolds(int k) {
		this.k = k;
		kSet = true;
	}

	/**
	 * Gets stratified folds. Does not store any data, so each invocation of this method will cause calculation of folds.
	 * 
	 * @param data full data set
	 * @throws InvalidValueException if seed or number of folds has not been set prior to this call
	 */
	@Override
	public List<CrossValidationFold> getStratifiedFolds(Data data) {
		if (seedSet && kSet) {
			CrossValidator crossValidator = new CrossValidator(new Random());
			crossValidator.setSeed(seed);
			
			InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = (data.getInformationTable() instanceof InformationTableWithDecisionDistributions ?
					(InformationTableWithDecisionDistributions)data.getInformationTable() : new InformationTableWithDecisionDistributions(data.getInformationTable(), true));
			
			List<org.rulelearn.sampling.CrossValidator.CrossValidationFold<InformationTable>> folds = crossValidator.splitStratifiedIntoKFolds(informationTableWithDecisionDistributions, true, k); 
			List<CrossValidationFold> crossValidationFolds = new ArrayList<CrossValidationFold>(k);
			
			int foldIndex = 0;
			for (org.rulelearn.sampling.CrossValidator.CrossValidationFold<InformationTable> fold : folds) {
				Data trainData = new Data(fold.getTrainingTable(), data.getName()+"_"+seed+"_train_"+foldIndex, data.getSeed());
				Data testData = new Data(fold.getValidationTable(), data.getName()+"_"+seed+"_test_"+foldIndex, data.getSeed());
				crossValidationFolds.add(new RepeatableCrossValidationFold(trainData, testData, foldIndex++));
			}
			
			return crossValidationFolds;
		} else {
			throw new InvalidValueException("Seed or number of folds not set in repeatable cross validation.");
		}
	}

}

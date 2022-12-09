/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Single fold of a {@link RepeatableCrossValidation}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class RepeatableCrossValidationFold implements CrossValidationFold {
	
	Data trainData;
	Data testData;
	int foldIndex;
	
	boolean done = false;
	
	public RepeatableCrossValidationFold(Data trainData, Data testData, int foldIndex) {
		this.trainData = trainData;
		this.testData = testData;
		this.foldIndex = foldIndex;
	}

	/**
	 * @throws UnsupportedOperationException if this cross validation fold has already {@link #done() done} his job.
	 */
	@Override
	public Data getTrainData() {
		if (!done) {
			return trainData;
		} else {
			throw new UnsupportedOperationException("Repeatable cross validation fold has already done his job.");
		}
	}

	/**
	 * @throws UnsupportedOperationException if this cross validation fold has already {@link #done() done} his job.
	 */
	@Override
	public Data getTestData() {
		if (!done) {
			return testData;
		} else {
			throw new UnsupportedOperationException("Repeatable cross validation fold has already done his job.");
		}
	}

	/**
	 * @throws UnsupportedOperationException if this cross validation fold has already {@link #done() done} his job.
	 */
	@Override
	public int getIndex() {
		if (!done) {
			return foldIndex;
		} else {
			throw new UnsupportedOperationException("Repeatable cross validation fold has already done his job.");
		}
	}

	@Override
	public void done() {
		trainData = null;
		testData = null;
		done = true;
	}

}

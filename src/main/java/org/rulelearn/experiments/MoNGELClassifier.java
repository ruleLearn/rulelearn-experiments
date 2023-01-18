/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.data.SimpleDecision;

import keel.Algorithms.Monotonic_Classification.MoNGEL.MoNGEL;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class MoNGELClassifier extends KEELClassifier {
	
	MoNGEL aMoNGEL;

	/**
	 * Class constructor.
	 * 
	 * @param modelLearningStatistics
	 */
	public MoNGELClassifier(ModelLearningStatistics modelLearningStatistics) {
		super(modelLearningStatistics);
		// TODO Auto-generated constructor stub
	}

	@Override
	public ModelValidationResult validate(Data testData) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SimpleDecision classify(int i, Data data) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ModelDescription getModelDescription() {
		// TODO Auto-generated method stub
		return null;
	}

}

/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.data.SimpleDecision;

/**
 * Generic KEEL classifier.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class KEELClassifier implements ClassificationModel {
	ModelLearningStatistics modelLearningStatistics;
	ModelDescription modelDescription = null;
	
	public KEELClassifier(ModelLearningStatistics modelLearningStatistics) {
		this.modelLearningStatistics = modelLearningStatistics;
	}
	
	@Override
	public abstract ModelValidationResult validate(Data testData);
	
	@Override
	public abstract SimpleDecision classify(int i, Data data);
	
	@Override
	public abstract ModelDescription getModelDescription();
	
	@Override
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}
}

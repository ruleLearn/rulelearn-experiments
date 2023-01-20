package org.rulelearn.experiments;

/**
 * Generic KEEL classifier.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class KEELClassifier implements ClassificationModel {
	ModelLearningStatistics modelLearningStatistics;
	
	public KEELClassifier(ModelLearningStatistics modelLearningStatistics) {
		this.modelLearningStatistics = modelLearningStatistics;
	}
	
	@Override
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}
}

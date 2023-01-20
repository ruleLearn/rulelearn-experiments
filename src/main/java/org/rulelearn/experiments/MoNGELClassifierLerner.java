package org.rulelearn.experiments;

import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;

import keel.Algorithms.Classification.Classifier;
import keel.Algorithms.Monotonic_Classification.MoNGEL.MoNGEL;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class MoNGELClassifierLerner extends KEELClassifierLerner {
	
	/**
	 * Sole constructor.
	 */
	public MoNGELClassifierLerner() {
		super(() -> new MoNGEL()); //use provider that provides MoNGEL as KEEL classifier
	}

	@Override
	MoNGELClassifier constructKEELClassifier(Classifier trainedClassifer, AttributeRanges attributeRanges, ModelLearningStatistics modelLearningStatistics) {
		return new MoNGELClassifier((MoNGEL)trainedClassifer, attributeRanges, modelLearningStatistics);
	}

}

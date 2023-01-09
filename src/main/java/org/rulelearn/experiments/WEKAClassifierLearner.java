/**
 * 
 */
package org.rulelearn.experiments;

import java.util.function.Supplier;

import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAClassifierLearner extends AbstractLearningAlgorithm {
	
	/**
	 * Supplies new instance of an {@link AbstractClassifier}.
	 */
	Supplier<AbstractClassifier> wekaClassifierProvider;
	
	public WEKAClassifierLearner(Supplier<AbstractClassifier> wekaClassifierProvider) {
		this.wekaClassifierProvider = wekaClassifierProvider;
	}

	@Override
	public WEKAClassifer learn(Data data, LearningAlgorithmDataParameters parameters) { //parameters can be null, if not used (i.e., WEKA algorithm is used with default options)
		Instances train = InformationTable2Instances.convert(data.getInformationTable(), data.getName());
		AbstractClassifier wekaClassifier = wekaClassifierProvider.get();

		try {
			String options;
			if (parameters != null && (options = parameters.getParameter(WEKAAlgorithmOptions.optionsParameterName)) != null) {
				wekaClassifier.setOptions(weka.core.Utils.splitOptions(options));
			}
			wekaClassifier.buildClassifier(train); //train the classifier
		} catch (Exception e) {
			e.printStackTrace();
			return null; //TODO: handle exception?
		}
		
		//calculate ModelLearningStatistics
		long start = System.currentTimeMillis();
		int numberOfLearningObjects = data.getInformationTable().getNumberOfObjects();
		int numberOfConsistentLearningObjects = ClassificationModel.getNumberOfConsistentObjects(data.getInformationTable(), 0.0);
		double consistencyThreshold = -1.0;
		int numberOfConsistentLearningObjectsForConsistencyThreshold = -1;
		String modelLearnerDescription = (new StringBuilder(getName())).append("(").append(parameters).append(")").toString();
		long statisticsCountingTime = System.currentTimeMillis() - start;
		
		ModelLearningStatistics modelLearningStatistics = new ModelLearningStatistics(
				numberOfLearningObjects, numberOfConsistentLearningObjects, consistencyThreshold, numberOfConsistentLearningObjectsForConsistencyThreshold,
				modelLearnerDescription, statisticsCountingTime);
		
		return new WEKAClassifer(wekaClassifier, modelLearningStatistics);
	}

	@Override
	public String getName() {
		return getAlgorithmName(wekaClassifierProvider.get().getClass());
	}
	
	public static String getAlgorithmName(Class<? extends AbstractClassifier> basicClassifierClass) {
		return WEKAClassifierLearner.class.getSimpleName()+"("+basicClassifierClass.getSimpleName()+")";
	}

}

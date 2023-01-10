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
		//+++++
		long start = System.currentTimeMillis();
		int numberOfLearningObjects = data.getInformationTable().getNumberOfObjects();
		
		Integer numberOfConsistentLearningObjectsObj = NumberOfConsistentObjectsCache.getInstance().getNumberOfConsistentObjects(data.getName(), 0.0);
		int numberOfConsistentLearningObjects;
		if (numberOfConsistentLearningObjectsObj != null) { //number of objects already in cache
			numberOfConsistentLearningObjects = numberOfConsistentLearningObjectsObj.intValue();
		} else { //number of objects not yet in cache
			numberOfConsistentLearningObjects = ClassificationModel.getNumberOfConsistentObjects(data.getInformationTable(), 0.0);
			NumberOfConsistentObjectsCache.getInstance().putNumberOfConsistentObjects(data.getName(), 0.0, numberOfConsistentLearningObjects); //store calculated number of objects in cache
		}
		
		double consistencyThreshold = -1.0;
		int numberOfConsistentLearningObjectsForConsistencyThreshold = -1;
		String modelLearnerDescription = (new StringBuilder(getName())).append("(").append(parameters).append(")").toString();
		long statisticsCountingTime = System.currentTimeMillis() - start;
		
		ModelLearningStatistics modelLearningStatistics = new ModelLearningStatistics(
				numberOfLearningObjects, numberOfConsistentLearningObjects, consistencyThreshold, numberOfConsistentLearningObjectsForConsistencyThreshold,
				modelLearnerDescription, statisticsCountingTime);
		//+++++
		
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

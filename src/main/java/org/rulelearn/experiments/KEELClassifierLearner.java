package org.rulelearn.experiments;

import java.util.function.Supplier;

import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;

import keel.Algorithms.Classification.Classifier;
import keel.Dataset.InstanceSet;

/**
 * Generic KEEL classifier learner.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class KEELClassifierLearner extends AbstractLearningAlgorithm {
	
	/**
	 * Supplies new instance of an {@link Classifier}.
	 */
	Supplier<Classifier> keelClassifierProvider;

	public KEELClassifierLearner(Supplier<Classifier> keelClassifierProvider) {
		this.keelClassifierProvider = keelClassifierProvider;
	}
	
	abstract KEELClassifier constructKEELClassifier(Classifier trainedClassifer, AttributeRanges attributeRanges, ModelLearningStatistics modelLearningStatistics);
	
	@Override
	public ClassificationModel learn(Data data, LearningAlgorithmDataParameters parameters) { //parameters should be an instance of KeelAlgorithmDataParameters
		AttributeRanges attributeRanges = ((KEELAlgorithmDataParameters)parameters).getAttributeRanges();
		
		InstanceSet trainInstanceSet = InformationTable2InstanceSet.convert(data.getInformationTable(), data.getName(), attributeRanges);
		
		Classifier trainedClassifer = keelClassifierProvider.get();
		trainedClassifer.buildClassifier(trainInstanceSet);
		
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
				modelLearnerDescription, 0L, 0L, statisticsCountingTime); //no data transformation, no retrieving model from cache
				//+++++
		
		return constructKEELClassifier(trainedClassifer, attributeRanges, modelLearningStatistics);
	}
	
	@Override
	public String getName() {
		return getAlgorithmName(keelClassifierProvider.get().getClass());
	}
	
	public static String getAlgorithmName(Class<? extends Classifier> basicClassifierClass) {
		return KEELClassifierLearner.class.getSimpleName()+"("+basicClassifierClass.getSimpleName()+")";
	}

}

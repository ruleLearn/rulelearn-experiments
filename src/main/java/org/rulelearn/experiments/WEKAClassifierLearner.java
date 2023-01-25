/**
 * 
 */
package org.rulelearn.experiments;

import java.util.function.Supplier;

import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;

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
		
		Filter[] filters = null;
		
		try {
			String options;
			if (parameters != null) {
				if ((options = parameters.getParameter(WEKAAlgorithmOptions.optionsParameterName)) != null) {
					wekaClassifier.setOptions(weka.core.Utils.splitOptions(options));
				}
				
				Supplier<Filter[]> filtersProvider = ((WEKAAlgorithmOptions)parameters).getFiltersProvider();
				
				if (filtersProvider != null) {
					filters = filtersProvider.get();
					if (filters != null) { //there may be some filters
						for (Filter filter : filters) { //use subsequent filters, if there are any filters meant to be used (array is not empty)
							filter.setInputFormat(train);
							train = Filter.useFilter(train, filter);
						}
					}
				}
			}
			
			wekaClassifier.buildClassifier(train); //train the classifier
		} catch (Exception exception) {
			exception.printStackTrace();
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
				modelLearnerDescription, 0L, 0L, statisticsCountingTime); //no data transformation, no retrieving model from cache
		//+++++
		
		return new WEKAClassifer(wekaClassifier, filters, modelLearningStatistics);
	}

	@Override
	public String getName() {
		return getAlgorithmName(wekaClassifierProvider.get().getClass());
	}
	
	public static String getAlgorithmName(Class<? extends AbstractClassifier> basicClassifierClass) {
		return WEKAClassifierLearner.class.getSimpleName()+"("+basicClassifierClass.getSimpleName()+")";
	}

}

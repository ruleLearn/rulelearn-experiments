/**
 * 
 */
package org.rulelearn.experiments;

import java.util.function.Supplier;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAClassifierLearner extends AbstractLearningAlgorithm {
	
	Supplier<AbstractClassifier> wekaClassifierProvider;
	
	public WEKAClassifierLearner(Supplier<AbstractClassifier> wekaClassifierProvider) {
		this.wekaClassifierProvider = wekaClassifierProvider;
	}

	@Override
	public ClassificationModel learn(Data data, LearningAlgorithmDataParameters parameters) {
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
			return null; //TODO
		}
		
		//Arrays.asList(j48.getOptions()).stream().forEach(System.out::println);
		return new WEKAClassifer(wekaClassifier);
	}

	@Override
	public String getName() {
		return getAlgorithmName(wekaClassifierProvider.get().getClass());
	}
	
	public static String getAlgorithmName(Class<? extends AbstractClassifier> basicClassifierClass) {
		return WEKAClassifierLearner.class.getSimpleName()+"("+basicClassifierClass.getSimpleName()+")";
	}

}

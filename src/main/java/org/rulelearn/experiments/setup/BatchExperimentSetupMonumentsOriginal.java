package org.rulelearn.experiments.setup;

import java.util.Arrays;
import java.util.List;

import org.rulelearn.experiments.BasicDataProvider;
import org.rulelearn.experiments.DataProcessor;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.LearningAlgorithmDataParametersContainer;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearner;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearnerDataParameters;
import org.rulelearn.experiments.WEKAAlgorithmOptions;
import org.rulelearn.experiments.WEKAClassifierLearner;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

import weka.classifiers.bayes.NaiveBayes;

/**
 * Batch experiment setup for monuments ('zabytki' in Polish) data set, concerning original data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupMonumentsOriginal extends BatchExperimentSetupMonuments {
	
	public BatchExperimentSetupMonumentsOriginal(long[] seeds, int k, DataProcessor dataProcessor) {
		super(seeds, k, dataProcessor);
	}
	
	@Override
	public List<LearningAlgorithm> getLearningAlgorithms() {
		if (learningAlgorithms == null) {
			learningAlgorithms = getLearningAlgorithmsForOriginalData();
		}
		
		return learningAlgorithms;
	}
	
	@Override
	public LearningAlgorithmDataParametersContainer getLearningAlgorithmDataParametersContainer() {
		if (parametersContainer == null) {
			parametersContainer = new LearningAlgorithmDataParametersContainer();
			
			parametersContainer
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR VC-DRSA RULES
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
								//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
						))
						
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV_K9_K10,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
								//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
						))
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01,
						Arrays.asList(
								//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
						))
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
						));

			parametersContainer
			//%%%%%%%%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR NAIVE BAYES
			//%%%%%%%%%%%%%%%%%%%%%%%%%%
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				//------
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV_K9_K10,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				//------
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV01,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				//------
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))); //option -D means discretize numeric attributes
			
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			//SORT PARAMETERS LISTS
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			parametersContainer.sortParametersLists(); //assure parameters for VCDomLEMModeRuleClassifierLearnerDataParameters algorithm are in ascending order w.r.t. consistency threshold
		}
		
		return parametersContainer;
	}

	@Override
	protected DataProvider getDataProviderMonuments(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
				"data/csv/zabytki-data-noMV.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10.json",
				"data/csv/zabytki-data-noMV.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
				"data/csv/zabytki-data-noMV-0-1.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10.json",
				"data/csv/zabytki-data-noMV-0-1.csv",
				false, ';',
				dataSetName, seeds, k);
	}
	
}

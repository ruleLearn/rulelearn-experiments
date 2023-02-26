package org.rulelearn.experiments.setup;

import java.util.Arrays;
import java.util.List;

import org.rulelearn.experiments.BasicDataProvider;
import org.rulelearn.experiments.DataProcessor;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.LearningAlgorithmDataParameters;
import org.rulelearn.experiments.LearningAlgorithmDataParametersContainer;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearner;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearnerDataParameters;
import org.rulelearn.experiments.WEKAAlgorithmOptions;
import org.rulelearn.experiments.WEKAClassifierLearner;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

import weka.classifiers.bayes.NaiveBayes;

/**
 * Batch experiment setup for churn4000v8 data set, concerning original data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupChurn4000v8Original extends BatchExperimentSetupChurn4000v8 {
	
	public BatchExperimentSetupChurn4000v8Original(long[] seeds, int k, DataProcessor dataProcessor) {
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
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8,
						Arrays.asList(
								/*new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),*/
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
								/*new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)*/ ))
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList()
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("support >= 1"), "1")
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						//getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", false),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", true),
//								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0",
										new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) ));
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
//						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())

			parametersContainer
			//%%%%%%%%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR NAIVE BAYES
			//%%%%%%%%%%%%%%%%%%%%%%%%%%
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D") )); //option -D means discretize numeric attributes
		
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			//SORT PARAMETERS LISTS
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			parametersContainer.sortParametersLists(); //assure parameters for VCDomLEMModeRuleClassifierLearnerDataParameters algorithm are in ascending order w.r.t. consistency threshold
		}
		
		return parametersContainer;
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata.json",
				"data/json-objects/bank-churn-4000-v8 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataSetName, seeds, k);
	}
	
	static List<LearningAlgorithmDataParameters> getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList() {
		return Arrays.asList(
		//parameter space search 1
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
		//parameter space search 2
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0")
		//parameter space search 3
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0"),
		//parameter space search 4
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0"),
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0")

		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
				
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)

		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
	
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0",	new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
	
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0",	new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true) //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		);
	}

}

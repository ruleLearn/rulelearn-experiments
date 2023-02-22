package org.rulelearn.experiments.setup;

import java.util.Arrays;
import java.util.List;

import org.rulelearn.experiments.BasicDataProvider;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.LearningAlgorithmDataParametersContainer;
import org.rulelearn.experiments.WEKAAlgorithmOptions;
import org.rulelearn.experiments.WEKAClassifierLearner;

import weka.classifiers.misc.OSDL;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Batch experiment setup for churn4000v8 data set, concerning data format compatible with OLM and OSDL algorithms.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupChurn4000v8OLM_OSDL extends BatchExperimentSetupChurn4000v8 {

	public BatchExperimentSetupChurn4000v8OLM_OSDL(long[] seeds, int k) {
		super(seeds, k);
	}
	
	@Override
	public List<LearningAlgorithm> getLearningAlgorithms() {
		if (learningAlgorithms == null) {
			learningAlgorithms = getLearningAlgorithmsForOLM_OSDLData();
		}
		
		return learningAlgorithms;
	}
	
	@Override
	public LearningAlgorithmDataParametersContainer getLearningAlgorithmDataParametersContainer() {
		if (parametersContainer == null) {
			parametersContainer = new LearningAlgorithmDataParametersContainer();
			
			parametersContainer
			//%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR OSDL
			//%%%%%%%%%%%%%%%%%%%
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						))
				
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(
								new WEKAAlgorithmOptions(null, () -> new Filter[] {new ReplaceMissingValues(), new Discretize()})
								//, new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize(), new ReplaceMissingValues()})
						));
		}
		
		return parametersContainer;
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata.json",
				"data/csv/OLM/bank-churn-4000-v8-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.05-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.05-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.10-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.10-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.15-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.15-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.20-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.20-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv2(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.25-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv15(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.25-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
}

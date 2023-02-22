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

/**
 * Batch experiment setup for monuments ('zabytki' in Polish) data set, concerning data format compatible with OLM and OSDL algorithms.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupMonumentsOLM_OSDL extends BatchExperimentSetupMonuments {
	
	public BatchExperimentSetupMonumentsOLM_OSDL(long[] seeds, int k) {
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
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameMonumentsNoMV,
						Arrays.asList(new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize()})))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameMonumentsNoMV_K9_K10,
						Arrays.asList(new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize()})))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameMonumentsNoMV01,
						Arrays.asList(new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize()})))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(OSDL.class), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(new WEKAAlgorithmOptions(null, () -> new Filter[] {new Discretize()})));
		}
		
		return parametersContainer;
	}

	@Override
	protected DataProvider getDataProviderMonuments(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/zabytki-metadata-Y1-K-enum-ordinal-Year1CG.json",
				"data/csv/zabytki-data-noMV-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/zabytki-metadata-Y1-K-enum-ordinal-K9-K10-Year1CG.json",
				"data/csv/zabytki-data-noMV-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/zabytki-metadata-Y1-K-enum-ordinal-0-1-Year1CG.json",
				"data/csv/zabytki-data-noMV-0-1-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/OLM/zabytki-metadata-Y1-K-enum-ordinal-0-1-K9-K10-Year1CG.json",
				"data/csv/zabytki-data-noMV-0-1-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}
	
}

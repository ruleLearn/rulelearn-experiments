package org.rulelearn.experiments.setup;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.rulelearn.experiments.AttributeRanges;
import org.rulelearn.experiments.BasicDataProvider;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.KEELAlgorithmDataParameters;
import org.rulelearn.experiments.KEELClassifierLearner;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.LearningAlgorithmDataParametersContainer;

import keel.Algorithms.Monotonic_Classification.MoNGEL.MoNGEL;

/**
 * Batch experiment setup for monuments ('zabytki' in Polish) data set, concerning data format compatible with MoNGEL algorithm.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupMonumentsMoNGEL extends BatchExperimentSetupMonuments {
	
	public BatchExperimentSetupMonumentsMoNGEL(long[] seeds, int k) {
		super(seeds, k);
	}
	
	@Override
	public List<LearningAlgorithm> getLearningAlgorithms() {
		if (learningAlgorithms == null) {
			learningAlgorithms = getLearningAlgorithmsForMoNGELData();
		}
		
		return learningAlgorithms;
	}
	
	@Override
	public LearningAlgorithmDataParametersContainer getLearningAlgorithmDataParametersContainer() {
		if (parametersContainer == null) {
			parametersContainer = new LearningAlgorithmDataParametersContainer();
			
//			if (learningAlgorithms.stream().filter(a -> a.getName().equals(KEELClassifierLearner.getAlgorithmName(MoNGEL.class))).collect(Collectors.toList()).size() > 0) { // MoNGEL is on the list of algorithms
			//%%%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR MONGEL
			//%%%%%%%%%%%%%%%%%%%%%
			
			List<DataProvider> tempDataProvidersList;
			final String[] dataNameBox = new String[1]; //regular String was not enough to satisfy stream constraint of a final or effectively final variable; unpack by checking index 0!
			
			dataNameBox[0] = dataNameMonumentsNoMV;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameMonumentsNoMV_K9_K10;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameMonumentsNoMV01;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameMonumentsNoMV01_K9_K10;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
//			} //if
		}
		
		return parametersContainer;
	}

	@Override
	protected DataProvider getDataProviderMonuments(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/zabytki-metadata-Y1-K-numeric-ordinal-Year1CG.json",
				"data/csv/zabytki-data-noMV-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10-Year1CG.json",
				"data/csv/zabytki-data-noMV-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/zabytki-metadata-Y1-K-numeric-ordinal-Year1CG.json",
				"data/csv/zabytki-data-noMV-0-1-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

	@Override
	protected DataProvider getDataProviderMonuments01_K9_K10(String dataSetName, long[] seeds, int k) {
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10-Year1CG.json",
				"data/csv/zabytki-data-noMV-0-1-Year1CG.csv",
				false, ';',
				dataSetName, seeds, k);
	}

}

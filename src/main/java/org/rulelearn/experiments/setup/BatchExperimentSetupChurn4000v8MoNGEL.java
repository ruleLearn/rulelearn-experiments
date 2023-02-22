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
 * Batch experiment setup for churn4000v8 data set, concerning data format compatible with MoNGEL algorithm.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentSetupChurn4000v8MoNGEL extends BatchExperimentSetupChurn4000v8 {
	
	public BatchExperimentSetupChurn4000v8MoNGEL(long[] seeds, int k) {
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
			
//			if (getLearningAlgorithms().stream().filter(a -> a.getName().equals(KEELClassifierLearner.getAlgorithmName(MoNGEL.class))).collect(Collectors.toList()).size() > 0) { // MoNGEL is on the list of algorithms
			//%%%%%%%%%%%%%%%%%%%%%
			//PARAMETERS FOR MONGEL
			//%%%%%%%%%%%%%%%%%%%%%
			
			getDataProviders(); //forces setting of dataProviders field (calculated upon first call)!
			
			List<DataProvider> tempDataProvidersList;
			final String[] dataNameBox = new String[1]; //regular String was not enough to satisfy stream constraint of a final or effectively final variable; unpack by checking index 0!
			
			dataNameBox[0] = dataNameChurn4000v8;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_05_mv2;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_05_mv15;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_10_mv2;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_10_mv15;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_15_mv2;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_15_mv15;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_20_mv2;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_20_mv15;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_25_mv2;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
			dataNameBox[0] = dataNameChurn4000v8_0_25_mv15;
			if ( (tempDataProvidersList = dataProviders.stream().filter(p -> p.getDataName().equals(dataNameBox[0])).limit(1).collect(Collectors.toList()) ).size() > 0) { //data will be provided
				parametersContainer.putParameters(KEELClassifierLearner.getAlgorithmName(MoNGEL.class), dataNameBox[0],
						Arrays.asList(new KEELAlgorithmDataParameters(new AttributeRanges(tempDataProvidersList.get(0).previewOriginalData().getInformationTable() ))) );
			}
//			} //if
		}
		
		return parametersContainer;
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.00_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv2(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_05_mv15(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv2(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_10_mv15(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv2(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_15_mv15(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv2(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_20_mv15(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv2(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
	@Override
	protected DataProvider getDataProviderChurn4000v8_0_25_mv15(String dataSetName, long[] seeds, int k) { //concerns data set version MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER
		return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
	}
	
}

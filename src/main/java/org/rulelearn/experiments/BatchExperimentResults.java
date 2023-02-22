/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import org.rulelearn.data.Decision;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentResults {
	
	private class FoldsResults { //results for all folds within single cross-validation
		ModelValidationResult[] foldModelValidationResults; //result for each fold
		Decision[] orderOfDecisions; //order taken from entire data
		ModelValidationResult aggregatedModelValidationResult; //result aggregated over all folds
		
		public FoldsResults(Decision[] orderOfDecisions, int foldsCount) {
			this.orderOfDecisions = orderOfDecisions;
			this.foldModelValidationResults = new ModelValidationResult[foldsCount];
			this.aggregatedModelValidationResult = null;
		}
	}
	
	public static class FullDataResults { //full information table results for a single data set
		Map<Double, Double> consistencyThreshold2QualityOfApproximation;
		Map<String, FullDataModelValidationResult> algorithmNameWithParameters2Results; //maps "algorithm-name(parameters)" to full data model validation result
		
		public FullDataResults(Map<Double, Double> consistencyThreshold2QualityOfApproximation, Map<String, FullDataModelValidationResult> algorithmNameWithParameters2Results) { //linked hash maps should be passed!
			this.consistencyThreshold2QualityOfApproximation = consistencyThreshold2QualityOfApproximation;
			this.algorithmNameWithParameters2Results = algorithmNameWithParameters2Results;
		}
	}
	
	public static class Builder {
		int dataSetsCount = -1;
		int learningAlgorithmsCount = -1;
		int maxParametersCount = -1;
		int maxCrossValidationsCount = -1;
		
		public Builder() {}
		
		public Builder dataSetsCount(int dataSetsCount) {
			this.dataSetsCount = dataSetsCount;
			return this;
		}
		public Builder learningAlgorithmsCount(int learningAlgorithmsCount) {
			this.learningAlgorithmsCount = learningAlgorithmsCount;
			return this;
		}
		public Builder maxParametersCount(int maxParametersCount) {
			this.maxParametersCount = maxParametersCount;
			return this;
		}
		public Builder maxCrossValidationsCount(int maxCrossValidationsCount) {
			this.maxCrossValidationsCount = maxCrossValidationsCount;
			return this;
		}
		
		public BatchExperimentResults build() {
			BatchExperimentResults batchExperimentResult = new BatchExperimentResults(dataSetsCount, learningAlgorithmsCount, maxParametersCount, maxCrossValidationsCount);
			return batchExperimentResult;
		}
	}
	
	public static class DataAlgorithmParametersSelector { //selects data+algorithm+parameters triple
		int dataSetNumber = -1;
		int learningAlgorithmNumber = -1;
		int parametersNumber = -1;
		
		public DataAlgorithmParametersSelector() {}
		
		public DataAlgorithmParametersSelector(DataAlgorithmParametersSelector selector) { //copying constructor
			this.dataSetNumber = selector.dataSetNumber;
			this.learningAlgorithmNumber = selector.learningAlgorithmNumber;
			this.parametersNumber = selector.parametersNumber;
		}
		
		public DataAlgorithmParametersSelector dataSetNumber(int dataSetNumber) {
			this.dataSetNumber = dataSetNumber;
			return this;
		}
		public DataAlgorithmParametersSelector learningAlgorithmNumber(int learningAlgorithmNumber) {
			this.learningAlgorithmNumber = learningAlgorithmNumber;
			return this;
		}
		public DataAlgorithmParametersSelector parametersNumber(int parametersNumber) {
			this.parametersNumber = parametersNumber;
			return this;
		}
		
	}
	
	public static class CVSelector extends DataAlgorithmParametersSelector { //selects data+algorithm+CV triple
		int crossValidationNumber = -1;
		
		public CVSelector() {}
		
		@Override
		public CVSelector dataSetNumber(int dataSetNumber) {
			this.dataSetNumber = dataSetNumber;
			return this;
		}
		@Override
		public CVSelector learningAlgorithmNumber(int learningAlgorithmNumber) {
			this.learningAlgorithmNumber = learningAlgorithmNumber;
			return this;
		}
		@Override
		public CVSelector parametersNumber(int parametersNumber) {
			this.parametersNumber = parametersNumber;
			return this;
		}
		public CVSelector crossValidationNumber(int crossValidationNumber) {
			this.crossValidationNumber = crossValidationNumber;
			return this;
		}
	}
	
	public static class FullDataModelValidationResult {
		DataAlgorithmParametersSelector selector;
		ModelValidationResult modelValidationResult;
		
		public FullDataModelValidationResult(DataAlgorithmParametersSelector selector, ModelValidationResult modelValidationResult) {
			this.selector = selector;
			this.modelValidationResult = modelValidationResult;
		}

		public DataAlgorithmParametersSelector getSelector() {
			return selector;
		}

		public ModelValidationResult getModelValidationResult() {
			return modelValidationResult;
		}
		
	}
	
	/**
	 * A pair of total calculation times, with one total time concerning training of a classifier, and the other total time concerning validation of a classifier.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public class CalculationTimes {
		private long totalTrainingTime = 0L; //duration of training (learning by) a classifier, in [ms]
		private int totalTrainingTimeIncreaseCount = 0; //tells how many times total training time was increased
		
		private long totalValidationTime = 0L; //duration of validating a classifier, in [ms]
		private int totalValidationTimeIncreaseCount = 0; //tells how many times total validation time was increased
		
		synchronized void increaseTotalTrainingTime(long increase) {
			totalTrainingTime += increase;
			totalTrainingTimeIncreaseCount++;
		}
		synchronized void increaseTotalValidationTime(long increase) {
			totalValidationTime += increase;
			totalValidationTimeIncreaseCount++;
		}
		
		long getTotalTrainingTime() {
			return totalTrainingTime;
		}
		
		long getTotalValidationTime() {
			return totalValidationTime;
		}
		
		double getAverageTrainingTime() {
			return (double)totalTrainingTime / totalTrainingTimeIncreaseCount;
		}
		
		double getAverageValidationTime() {
			return (double)totalValidationTime / totalValidationTimeIncreaseCount;
		}
	}
	
	FoldsResults[][][][] foldsResults; //usage: foldsResults[dataSetNumber][learningAlgorithmNumber][parametersNumber][crossValidationNumber] = foldsResults
	
	int dataSetsCount = -1;
	int learningAlgorithmsCount = -1;
	int maxParametersCount = -1;
	int maxCrossValidationsCount = -1;
	
	Map<String, FullDataResults> dataName2FullDataResults;
	
	CalculationTimes[][][] totalFoldCalculationTimes; //usage: foldCalculationTimes[dataSetNumber][learningAlgorithmNumber][parametersNumber]; cumulated times over all folds (in any cross-validation)
	CalculationTimes[][][] fullDataCalculationTimes; //usage: fullDataCalculationTimes[dataSetNumber][learningAlgorithmNumber][parametersNumber]; time concerning single training and validation on full data
	
	private BatchExperimentResults(int dataSetsCount, int learningAlgorithmsCount, int maxParametersCount, int maxCrossValidationsCount) {
		this.dataSetsCount = dataSetsCount;
		this.learningAlgorithmsCount = learningAlgorithmsCount;
		this.maxParametersCount = maxParametersCount;
		this.maxCrossValidationsCount = maxCrossValidationsCount;
		this.dataName2FullDataResults = new HashMap<String, FullDataResults>();
		
		this.foldsResults = new FoldsResults[dataSetsCount][learningAlgorithmsCount][maxParametersCount][maxCrossValidationsCount];
		
		this.fullDataCalculationTimes = new CalculationTimes[dataSetsCount][learningAlgorithmsCount][maxParametersCount];
		this.totalFoldCalculationTimes = new CalculationTimes[dataSetsCount][learningAlgorithmsCount][maxParametersCount];
		
		//initialize calculation times
		for (int i = 0; i < dataSetsCount; i++) {
			for (int j = 0; j < learningAlgorithmsCount; j++) {
				for (int k = 0; k < maxParametersCount; k++) {
					this.fullDataCalculationTimes[i][j][k] = new CalculationTimes();
					this.totalFoldCalculationTimes[i][j][k] = new CalculationTimes();
				}
			}
		}
	} //accessible only by the builder
	
	public void storeFullDataResults (String dataName, FullDataResults fullDataResults) {
		dataName2FullDataResults.put(dataName, fullDataResults);
	}
	
	public String reportFullDataResults(String dataName) { //OUTPUT
		StringBuilder sb = new StringBuilder(256);
		FullDataResults fullDataResults = dataName2FullDataResults.get(dataName);
		Map<Double, Double> consistencyThreshold2QualityOfApproximation = fullDataResults.consistencyThreshold2QualityOfApproximation;
		Map<String, FullDataModelValidationResult> algorithmNameWithParameters2Results = fullDataResults.algorithmNameWithParameters2Results;
		
		if (consistencyThreshold2QualityOfApproximation != null) {
			consistencyThreshold2QualityOfApproximation.forEach((consistencyThreshold, qualityOfApproximation) -> {
				sb.append("Quality of approximation for ('").append(dataName).append("', consistency threshold=").append(consistencyThreshold).append("): ")
				.append(qualityOfApproximation).append(".").append(System.lineSeparator());
			});
		}
		
		sb.append("--").append(System.lineSeparator());
		
		algorithmNameWithParameters2Results.forEach(
			(algorithmNameWithParameters, result) -> {
				ClassificationStatistics classificationStatistics = result.getModelValidationResult().getClassificationStatistics();
				
				sb.append(String.format(Locale.US, "Train data result for ('%s', %s):%n"
						+ "  Accuracy: %s (overall: %s, avg: %s) # %s # %s (%s|%s). Main model decisions ratio: %s.%n"
						+ "  True positive rates: %s. Gmean: %s.%n"
						+ "  [Learning]: %s.%n"
						+ "%s%n"
						+ "  [Model]: %s.%n"
						+ "  [Times]: training: %d [ms], validation: %d [ms].",
						dataName, algorithmNameWithParameters,
						BatchExperiment.round(result.getModelValidationResult().getOrdinalMisclassificationMatrix().getAccuracy()),
						BatchExperiment.round(classificationStatistics.getOverallAccuracy()), //test if the same as above
						BatchExperiment.round(classificationStatistics.getAvgAccuracy()), //test if the same as above
						BatchExperiment.round(classificationStatistics.getMainModelAccuracy()),
						BatchExperiment.round(classificationStatistics.getDefaultModelAccuracy()),
						BatchExperiment.round(classificationStatistics.getDefaultClassAccuracy()),
						BatchExperiment.round(classificationStatistics.getDefaultClassifierAccuracy()),
						BatchExperiment.round(classificationStatistics.getMainModelDecisionsRatio()),
						BatchExperiment.getTruePositiveRates(result.getModelValidationResult().getOrdinalMisclassificationMatrix()),
						BatchExperiment.round(result.getModelValidationResult().getOrdinalMisclassificationMatrix().getGmean()),
						result.getModelValidationResult().getModelLearningStatistics().toString(),
						Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
						.map(line -> (new StringBuilder("  ")).append(line).toString())
						.collect(Collectors.joining(System.lineSeparator())), //print validation summary in several lines
						result.getModelValidationResult().getModelDescription().toShortString(), //print one line model description
						getFullDataCalculationTimes(result.getSelector()).getTotalTrainingTime(),
						getFullDataCalculationTimes(result.getSelector()).getTotalValidationTime()
				))
				.append(System.lineSeparator());
			}
		);
		
		return sb.toString();
	}
	
	//must be called before storeFoldMisclassificationMatrix!
	public void initializeFoldResults(CVSelector selector, Decision[] orderOfDecisions, int foldsCount) {
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber] = new FoldsResults(orderOfDecisions, foldsCount);
	}
	
	public void storeFoldModelValidationResult(CVSelector selector, int foldNumber, ModelValidationResult modelValidationResult) { //do initializeFoldResults before!
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber]
				.foldModelValidationResults[foldNumber] = modelValidationResult;
	}
	
	//gets aggregated over folds model validation result for selected single cross-validation, or null (if there are no fold results stored for given CV selector)
	public ModelValidationResult getAggregatedCVModelValidationResult(CVSelector selector) {
		FoldsResults _foldResults = foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber];
		
		if (_foldResults != null) {
			if (_foldResults.aggregatedModelValidationResult == null) { //there is no aggregated matrix yet
				_foldResults.aggregatedModelValidationResult = new ModelValidationResult(AggregationMode.SUM, _foldResults.orderOfDecisions, _foldResults.foldModelValidationResults);
			}
			return _foldResults.aggregatedModelValidationResult;
		} else {
			return null;
		}
	}
	
	public ModelValidationResult getAggregatedModelValidationResult(DataAlgorithmParametersSelector selector) {
		List<ModelValidationResult> modelValidationResults = new ArrayList<>(maxCrossValidationsCount);
		int numberOfCrossValidations = 0;
		Decision[] orderOfDecisions = null; //order taken from entire data
		
		for (int i = 0; i < maxCrossValidationsCount; i++) {
			ModelValidationResult aggregatedCVModelValidationResult = getAggregatedCVModelValidationResult(
					(new CVSelector()).dataSetNumber(selector.dataSetNumber).learningAlgorithmNumber(selector.learningAlgorithmNumber)
					.parametersNumber(selector.parametersNumber).crossValidationNumber(i));
			if (i == 0) {
				orderOfDecisions = aggregatedCVModelValidationResult.getOrderOfDecisions(); //get order from the first model validation result
			}
			
			if (aggregatedCVModelValidationResult != null) {
				numberOfCrossValidations++;
				modelValidationResults.add(aggregatedCVModelValidationResult);
			}
		} //for
		
		if (numberOfCrossValidations > 0) {
			return new ModelValidationResult(AggregationMode.MEAN_AND_DEVIATION, orderOfDecisions, modelValidationResults.toArray(new ModelValidationResult[0]));
		} else {
			return null;
		}

	}
	
	//used to get reference of CalculationTimes instance to increase time
	public CalculationTimes getFullDataCalculationTimes(DataAlgorithmParametersSelector selector) {
		return fullDataCalculationTimes[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber];
	}
	
	public CalculationTimes getTotalFoldCalculationTimes(DataAlgorithmParametersSelector selector) {
		return totalFoldCalculationTimes[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber];
	}
	
}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import org.rulelearn.data.Decision;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperimentResults {
	
	private class FoldsResults { //results for all folds within single cross-validation
		OrdinalMisclassificationMatrix[] foldMisclassificationMatrices; //matrix for each fold
		Decision[] orderOfDecisions; //taken from entire data
		OrdinalMisclassificationMatrix aggregatedMisclassificationMatrix; //matrix aggregated over all folds
		
		public FoldsResults(Decision[] orderOfDecisions, int foldsCount) {
			this.orderOfDecisions = orderOfDecisions;
			this.foldMisclassificationMatrices = new OrdinalMisclassificationMatrix[foldsCount];
			this.aggregatedMisclassificationMatrix = null;
		}
	}
	
	public static class FullDataResults { //full information table results for a single data set
		Map<Double, Double> consistencyThreshold2QualityOfApproximation;
		Map<String, Double> algorithmNameWithParameters2Accuracy; //maps "algorithm-name(parameters)" to accuracy
		
		public FullDataResults(Map<Double, Double> consistencyThreshold2QualityOfApproximation, Map<String, Double> algorithmNameWithParameters2Accuracy) { //linked hash maps should be passed!
			this.consistencyThreshold2QualityOfApproximation = consistencyThreshold2QualityOfApproximation;
			this.algorithmNameWithParameters2Accuracy = algorithmNameWithParameters2Accuracy;
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
	
	public class AverageEvaluation {
		double average;
		double stdDev; //standard deviation
		
		public AverageEvaluation(double average, double stdDev) {
			this.average = average;
			this.stdDev = stdDev;
		}
		
		public double getAverage() {
			return average;
		}
		public double getStdDev() {
			return stdDev;
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
	
	public String reportFullDataResults(String dataName) {
		StringBuilder sb = new StringBuilder();
		FullDataResults fullDataResults = dataName2FullDataResults.get(dataName);
		Map<Double, Double> consistencyThreshold2QualityOfApproximation = fullDataResults.consistencyThreshold2QualityOfApproximation;
		Map<String, Double> algorithmNameWithParameters2Accuracy = fullDataResults.algorithmNameWithParameters2Accuracy;
		
		if (consistencyThreshold2QualityOfApproximation != null) {
			consistencyThreshold2QualityOfApproximation.forEach((consistencyThreshold, qualityOfApproximation) -> {
				sb.append("Quality of approximation for ('").append(dataName).append("', consistency threshold=").append(consistencyThreshold).append("): ")
				.append(qualityOfApproximation).append(".").append(System.lineSeparator());
			});
		}
		
		algorithmNameWithParameters2Accuracy.forEach(
			(algorithmNameWithParameters, accuracy) ->
				sb.append(String.format(Locale.US, "Train data accuracy for ('%s', %s): %f.", dataName, algorithmNameWithParameters, accuracy)).append(System.lineSeparator())
		);
		
		return sb.toString();
	}
	
	//must be called before storeFoldMisclassificationMatrix!
	public void initializeFoldResults(CVSelector selector, Decision[] orderOfDecisions, int foldsCount) {
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber] = new FoldsResults(orderOfDecisions, foldsCount);
	}
	
	public void storeFoldMisclassificationMatrix(CVSelector selector, int foldNumber, OrdinalMisclassificationMatrix foldResult) { //do initializeFoldResults before!
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber].foldMisclassificationMatrices[foldNumber] = foldResult;
	}
	
	//gets average misclassification matrix or null, if there are no fold results stored for given selector
	public OrdinalMisclassificationMatrix getAverageCVMisclassificationMatrix(CVSelector selector) {
		FoldsResults _foldResults = foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber][selector.crossValidationNumber];
		
//		System.out.println("Getting avg misclassification matrix for: "+selector.dataSetNumber+", "+selector.learningAlgorithmNumber+", "+selector.crossValidationNumber);
		
		if (_foldResults != null) {
			if (_foldResults.aggregatedMisclassificationMatrix == null) { //there is already an aggregated matrix
				_foldResults.aggregatedMisclassificationMatrix = new OrdinalMisclassificationMatrix(_foldResults.orderOfDecisions, _foldResults.foldMisclassificationMatrices);
			}
			return _foldResults.aggregatedMisclassificationMatrix;
		} else {
			return null;
		}
	}
	
	public AverageEvaluation getAverageDataAlgorithmParametersAccuracy(DataAlgorithmParametersSelector selector) {
//		List<OrdinalMisclassificationMatrix> matrices = new ArrayList<>(maxCrossValidationsCount);
		double sumCVAccuracies = 0.0;
		OrdinalMisclassificationMatrix averageCVMisclassificationMatrix;
		
		List<Double> cvAccuracies = new ArrayList<Double>();
		
		int numberOfCrossValidations = 0;
		for (int i = 0; i < maxCrossValidationsCount; i++) {
			averageCVMisclassificationMatrix = getAverageCVMisclassificationMatrix(
					(new CVSelector()).dataSetNumber(selector.dataSetNumber).learningAlgorithmNumber(selector.learningAlgorithmNumber).parametersNumber(selector.parametersNumber).crossValidationNumber(i));
			
			if (averageCVMisclassificationMatrix != null) {
				numberOfCrossValidations++;
//				matrices.add(ordinalMisclassificationMatrix); //TODO: use this list of matrices, if more information is needed
				sumCVAccuracies += averageCVMisclassificationMatrix.getAccuracy();
				cvAccuracies.add(averageCVMisclassificationMatrix.getAccuracy());
			} else {
				break; //there are no more cross-validations stored
			}
		}
		
		double average = 0.0;
		double stdDev = 0.0;
		
		if (numberOfCrossValidations >= 1) {
			average = sumCVAccuracies / numberOfCrossValidations;
			final double streamAverage = average;
			if (numberOfCrossValidations > 1) {
				stdDev = Math.sqrt(((double)1 / (numberOfCrossValidations - 1)) * cvAccuracies.stream().map(a -> Math.pow(a - streamAverage, 2)).collect(Collectors.summingDouble(n -> n))); //divide by (N-1)
			}
		}
		
		AverageEvaluation result = new AverageEvaluation(average, stdDev);
		
		return result;
	}
	
	public CalculationTimes getFullDataCalculationTimes(DataAlgorithmParametersSelector selector) {
		return fullDataCalculationTimes[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber];
	}
	
	public CalculationTimes getTotalFoldCalculationTimes(DataAlgorithmParametersSelector selector) {
		return totalFoldCalculationTimes[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.parametersNumber];
	}
	
}

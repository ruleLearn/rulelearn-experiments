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
		OrdinalMisclassificationMatrix[] foldMisclassificationMatrices;
		Decision[] orderOfDecisions; //taken from entire data
		OrdinalMisclassificationMatrix aggregatedMisclassificationMatrix;
		
		public FoldsResults(Decision[] orderOfDecisions, int foldsCount) {
			this.orderOfDecisions = orderOfDecisions;
			this.foldMisclassificationMatrices = new OrdinalMisclassificationMatrix[foldsCount];
			this.aggregatedMisclassificationMatrix = null;
		}
	}
	
	public static class FullDataResults { //full information table results for a single data set
		double qualityOfDRSAApproximation;
		double qualityOfVCDRSAApproximation;
		double consistencyThresholdForVCDRSAApproximation;
		Map<String, Double> algorithmNameWithParameters2Accuracy;
		public FullDataResults(double qualityOfDRSAApproximation, double qualityOfVCDRSAApproximation,
				double consistencyThresholdForVCDRSAApproximation, Map<String, Double> algorithmNameWithParameters2Accuracy) { //should pass a linked hash map
			this.qualityOfDRSAApproximation = qualityOfDRSAApproximation;
			this.qualityOfVCDRSAApproximation = qualityOfVCDRSAApproximation;
			this.consistencyThresholdForVCDRSAApproximation = consistencyThresholdForVCDRSAApproximation;
			this.algorithmNameWithParameters2Accuracy = algorithmNameWithParameters2Accuracy;
		}
	}
	
	public static class Builder {
		int dataSetsCount = -1;
		int learningAlgorithmsCount = -1;
		int crossValidationsCount = -1;
		
		public Builder() {}
		
		public Builder dataSetsCount(int dataSetsCount) {
			this.dataSetsCount = dataSetsCount;
			return this;
		}
		public Builder learningAlgorithmsCount(int learningAlgorithmsCount) {
			this.learningAlgorithmsCount = learningAlgorithmsCount;
			return this;
		}
		public Builder crossValidationsCount(int crossValidationsCount) {
			this.crossValidationsCount = crossValidationsCount;
			return this;
		}
		
		public BatchExperimentResults build() {
			BatchExperimentResults batchExperimentResult = new BatchExperimentResults(dataSetsCount, learningAlgorithmsCount, crossValidationsCount);
			batchExperimentResult.foldsResults = new FoldsResults[dataSetsCount][learningAlgorithmsCount][crossValidationsCount];
			return batchExperimentResult;
		}
	}
	
	public static class DataAlgorithmSelector { //selects data+algorithm pair
		int dataSetNumber = -1;
		int learningAlgorithmNumber = -1;
		
		public DataAlgorithmSelector() {}
		
		public DataAlgorithmSelector dataSetNumber(int dataSetNumber) {
			this.dataSetNumber = dataSetNumber;
			return this;
		}
		public DataAlgorithmSelector learningAlgorithmNumber(int learningAlgorithmNumber) {
			this.learningAlgorithmNumber = learningAlgorithmNumber;
			return this;
		}
	}
	
	public static class CVSelector extends DataAlgorithmSelector { //selects data+algorithm+CV triple
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
		public CVSelector crossValidationNumber(int crossValidationNumber) {
			this.crossValidationNumber = crossValidationNumber;
			return this;
		}
	}
	
	public class AverageEvaluation {
		double average;
		double standardDeviation;
		public AverageEvaluation(double average, double standardDeviation) {
			this.average = average;
			this.standardDeviation = standardDeviation;
		}
		public double getAverage() {
			return average;
		}
		public double getStandardDeviation() {
			return standardDeviation;
		}
	}
	
	FoldsResults[][][] foldsResults;
	int dataSetsCount = -1;
	int learningAlgorithmsCount = -1;
	int maximumCrossValidationsCount = -1;
	
	Map<String, FullDataResults> dataName2FullDataResults;
	
	private BatchExperimentResults(int dataSetsCount, int learningAlgorithmsCount, int maximumCrossValidationsCount) {
		this.dataSetsCount = dataSetsCount;
		this.learningAlgorithmsCount = learningAlgorithmsCount;
		this.maximumCrossValidationsCount = maximumCrossValidationsCount;
		this.dataName2FullDataResults = new HashMap<String, FullDataResults>();
	} //accessible only by the builder
	
	public void storeFullDataResults (String dataName, FullDataResults fullDataResults) {
		dataName2FullDataResults.put(dataName, fullDataResults);
	}
	
	public String reportFullDataResults(String dataName) {
		StringBuilder sb = new StringBuilder();
		FullDataResults fullDataResults = dataName2FullDataResults.get(dataName);
		
		sb.append("Quality of approximation for ('").append(dataName).append("', consistency threshold=0.0): ").append(fullDataResults.qualityOfDRSAApproximation).append(".").append(System.lineSeparator());
		sb.append("Quality of approximation for ('").append(dataName).append("', consistency threshold=").append(fullDataResults.consistencyThresholdForVCDRSAApproximation).append("): ").append(fullDataResults.qualityOfVCDRSAApproximation).append(".").append(System.lineSeparator());
		
		Map<String, Double> algorithmNameWithParameters2Accuracy = fullDataResults.algorithmNameWithParameters2Accuracy;
		algorithmNameWithParameters2Accuracy.forEach(
			(algorithmNameWithParameters, accuracy) ->
				sb.append(String.format(Locale.US, "Train data accuracy for ('%s', %s): %f.", dataName, algorithmNameWithParameters, accuracy)).append(System.lineSeparator())
		);
		
		return sb.toString();
	}
	
	//must be called before storeFoldMisclassificationMatrix!
	public void initializeFoldResults(CVSelector selector, Decision[] orderOfDecisions, int foldsCount) {
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.crossValidationNumber] = new FoldsResults(orderOfDecisions, foldsCount);
	}
	
	public void storeFoldMisclassificationMatrix(CVSelector selector, int foldNumber, OrdinalMisclassificationMatrix foldResult) { //do initializeFoldResults before!
		foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.crossValidationNumber].foldMisclassificationMatrices[foldNumber] = foldResult;
	}
	
	//gets average misclassification matrix or null, if there are no fold results stored for given selector
	public OrdinalMisclassificationMatrix getAverageCVMisclassificationMatrix(CVSelector selector) {
		FoldsResults _foldResults = foldsResults[selector.dataSetNumber][selector.learningAlgorithmNumber][selector.crossValidationNumber];
		
//		System.out.println("Getting avg misclassification matrix for: "+selector.dataSetNumber+", "+selector.learningAlgorithmNumber+", "+selector.crossValidationNumber);
		
		if (_foldResults != null) {
			if (_foldResults.aggregatedMisclassificationMatrix != null) { //there is already an aggregated matrxi
				return _foldResults.aggregatedMisclassificationMatrix;
			} else {
				_foldResults.aggregatedMisclassificationMatrix = new OrdinalMisclassificationMatrix(_foldResults.orderOfDecisions, _foldResults.foldMisclassificationMatrices);
				return _foldResults.aggregatedMisclassificationMatrix;
			}
		} else {
			return null;
		}
	}
	
	public AverageEvaluation getAverageDataAlgorithmAccuracy(DataAlgorithmSelector selector) {
//		List<OrdinalMisclassificationMatrix> matrices = new ArrayList<>(maximumCrossValidationsCount);
		double sumCVAccuracies = 0.0;
		OrdinalMisclassificationMatrix averageCVMisclassificationMatrix;
		
		List<Double> cvAccuracies = new ArrayList<Double>();
		
		int numberOfCrossValidations = 0;
		for (int i = 0; i < maximumCrossValidationsCount; i++) {
			averageCVMisclassificationMatrix = getAverageCVMisclassificationMatrix((new CVSelector()).dataSetNumber(selector.dataSetNumber).learningAlgorithmNumber(selector.learningAlgorithmNumber).crossValidationNumber(i));
			
			if (averageCVMisclassificationMatrix != null) {
				numberOfCrossValidations++;
//				matrices.add(ordinalMisclassificationMatrix); //TODO: use this list of matrices
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
	
}

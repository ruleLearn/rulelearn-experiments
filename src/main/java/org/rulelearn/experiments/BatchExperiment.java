/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.experiments.BatchExperimentResults.AverageEvaluation;
import org.rulelearn.experiments.BatchExperimentResults.AverageEvaluations;
import org.rulelearn.experiments.BatchExperimentResults.CVSelector;
import org.rulelearn.experiments.BatchExperimentResults.CalculationTimes;
import org.rulelearn.experiments.BatchExperimentResults.DataAlgorithmParametersSelector;
import org.rulelearn.experiments.BatchExperimentResults.FullDataResults;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.misc.OSDL;
import weka.classifiers.rules.OLM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

/**
 * Batch repeated cross-validation experiment over multiple data sets, with pre-processing of learning data, and different parameterized learning methods.
 * Each data set used in the experiment need to be an ordinal classification problem (with order among decisions).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperiment {

	List<DataProvider> dataProviders;
	CrossValidationProvider crossValidationProvider;
	DataProcessor trainDataPreprocessor;
	List<LearningAlgorithm> learningAlgorithms;
	LearningAlgorithmDataParametersContainer parametersContainer;
	
	//<BEGIN EXPERIMENT CONFIG>
	static boolean useMainModelAccuracy = false; //true = use main model accuracy; false = use overall accuracy
	static final boolean doFullDataReclassification = true;
	static final boolean doCrossValidations = false; //true = perform CVs; false = skip CVs
	//<END EXPERIMENT CONFIG>
	
	/**
	 * Constructs this experiment.
	 * 
	 * @param dataProviders
	 * @param crossValidations
	 * @param trainDataPreprocessor
	 * @param learningAlgorithms
	 * @param parametersContainer
	 */
	public BatchExperiment(List<DataProvider> dataProviders, CrossValidationProvider crossValidationProvider, DataProcessor trainDataPreprocessor, List<LearningAlgorithm> learningAlgorithms,
			LearningAlgorithmDataParametersContainer parametersContainer) {
		this.dataProviders = dataProviders;
		this.crossValidationProvider = crossValidationProvider;
		this.trainDataPreprocessor = trainDataPreprocessor;
		this.learningAlgorithms = learningAlgorithms;
		this.parametersContainer = parametersContainer;
	}
	
	static String resolveText(String text, Object... params) {
		for (int i = 0; i < params.length; i++) {
			text = text.replaceFirst("%"+(i+1), (params[i] != null ? params[i].toString() : "null"));
		}
		return text;
	}
	
	//e.g.: out("This is a test %1, %2.", 23, "abc") results in System.out.print("This is a test 23, abc.");
	static void out(String text, Object... params) {
		System.out.print(resolveText(text, params));
	}
	
	//e.g.: out("This is a test %1, %2.", 23, "abc") results in System.out.println("This is a test 23, abc.");
	static void outN(String text, Object... params) {
		System.out.println(resolveText(text, params));
	}
	
	static void out(String text) {
		System.out.print(text);
	}
	
	static void outN(String text) {
		System.out.println(text);
	}
	
	static void outN() {
		System.out.println();
	}
	
	private long b(String msg) { //begin
		if (msg != null) {
			outN(msg);
		}
		return System.currentTimeMillis();
	}
	
	private void e(long t, String msg) { //end
		long duration = System.currentTimeMillis() - t;
		outN("%1 [Duration]: %2 [ms].", msg, duration);
	}
	
	private String foldNumber2Spaces(int foldNumber) {
		if (foldNumber == 0) {
			return "";
		}
		StringBuilder sb = new StringBuilder(foldNumber);
		for (int i = 0; i < foldNumber; i++) {
			sb.append("  ");
		}
		return sb.toString();
	}
	
	/**
	 * Calculates quality of approximation using \epsilon-VC-DRSA.
	 * 
	 * @param informationTable the data
	 * @return quality of approximation
	 */
	double calculateQualityOfApproximation(InformationTable informationTable, double consistencyThreshold) {
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = (informationTable instanceof InformationTableWithDecisionDistributions ?
				(InformationTableWithDecisionDistributions)informationTable : new InformationTableWithDecisionDistributions(informationTable, true));
		
		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, 
				   new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
		
		return unions.getQualityOfApproximation();
	}
	
	private static List<LearningAlgorithmDataParameters> processListOfParameters(List<LearningAlgorithmDataParameters> parametersList) {
		if (parametersList == null || parametersList.size() == 0) {
			return Arrays.asList(new LearningAlgorithmDataParameters[]{null}); //put one parameters equal to null, i.e., default parameters
		}
		return parametersList;
	}
	
	/**
	 * Runs this experiment.
	 * 
	 * @return results of this experiment
	 */
	public BatchExperimentResults run() {
		
		//calculate maximum number of parameters for an algorithm
		int maxParametersCount = -1;
		int parametersCount;
		for (DataProvider dataProvider : dataProviders) {
			for (LearningAlgorithm algorithm : learningAlgorithms) {
				List<LearningAlgorithmDataParameters> parametersList = processListOfParameters(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
				//Optional<List<LearningAlgorithmDataParameters>> optional = Optional.ofNullable(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
				//parametersCount = optional.map(list -> list.size()).orElse(0);
				parametersCount = parametersList.size();
				if (parametersCount > maxParametersCount) {
					maxParametersCount = parametersCount;
				}
			}
		}
		outN("Maximum number of algorithm vs data parameters, over all (data, algorithm) pairs: %1.", maxParametersCount); //!
		
		//calculate maximum number of cross validations among all data sets
				int maxCrossValidationsCount = -1;
				for (DataProvider dataProvider : dataProviders) {
					if (dataProvider.getSeeds().length > maxCrossValidationsCount) {
						maxCrossValidationsCount = dataProvider.getSeeds().length;
					}
				}
				outN("Maximum number of cross-validations: %1.", maxCrossValidationsCount); //!
		
		outN(); //!
		
		BatchExperimentResults batchExperimentResults = (new BatchExperimentResults.Builder())
				.dataSetsCount(dataProviders.size()).learningAlgorithmsCount(learningAlgorithms.size()).maxParametersCount(maxParametersCount)
				.maxCrossValidationsCount(maxCrossValidationsCount).build();
		
		List<CrossValidationFold> crossValidationFolds;
		Data data;
		int dataSetNumber;
		int crossValidationsCount;
		CrossValidation crossValidation;
		BatchExperimentResults.CVSelector initializingCVSelector;
		
		long t1, t2;
		boolean doCrossValidationsForProvider;
		List<LearningAlgorithmDataParameters> parametersList;
		
		dataSetNumber = -1;
		for (DataProvider dataProvider : dataProviders) {
			dataSetNumber++;
			final int streamDataSetNumber = dataSetNumber;
			
			doCrossValidationsForProvider = dataProvider.getSeeds().length > 0 && doCrossValidations;
			
			if (doCrossValidationsForProvider || doFullDataReclassification) {
				t1 = b(resolveText("Starting calculations for data %1.", dataProvider.getDataName()));

				//>>>>> PROCESS FULL DATA
				if (doFullDataReclassification) {
					double epsilonDRSAConsistencyThreshold = 0.0;
					double qualityOfDRSAApproximation;
					double qualityOfVCDRSAApproximation;
					Map<String, BatchExperimentResults.Evaluations> algorithmNameWithParameters2Evaluations = new LinkedHashMap<String, BatchExperimentResults.Evaluations>();
					Map<Double, Double> consistencyThreshold2QualityOfApproximation = new LinkedHashMap<Double, Double>();
					
					//print full data set accuracies
					Data fullData = dataProvider.provideOriginalData();
					outN("Quality of approximation for consistency threshold=%1: %2.", epsilonDRSAConsistencyThreshold, qualityOfDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), 0.0));
					consistencyThreshold2QualityOfApproximation.put(Double.valueOf(epsilonDRSAConsistencyThreshold), qualityOfDRSAApproximation);
					
					//if rule classifier is used for any data (thus in particular for the current data)
					if (learningAlgorithms.stream().filter(algorithm -> algorithm.getName().equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())).collect(Collectors.toList()).size() > 0) {
						
						parametersList = parametersContainer.getParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataProvider.getDataName()); //get list of parameters for rule classifier
						
						if (parametersList != null) { //rule classifier is there (i.e., has at least one parameters) :)
							for (LearningAlgorithmDataParameters parameters : parametersList) { //check quality of approximation for all considered consistency thresholds
								double consistencyThreshold = Double.valueOf(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.consistencyThresholdParameterName));
								if (!consistencyThreshold2QualityOfApproximation.containsKey(Double.valueOf(consistencyThreshold))) { //ensure that quality of approximation is calculated for each consistency threshold only once
									outN("Quality of approximation for consistency threshold=%1: %2.", consistencyThreshold, qualityOfVCDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), consistencyThreshold));
									consistencyThreshold2QualityOfApproximation.put(consistencyThreshold, qualityOfVCDRSAApproximation);
								}
							}
						}
					}
					
					//calculate and process full data models for all (algorithm, parameters) pairs
					Data processedFullData = trainDataPreprocessor.process(fullData);
					int algorithmNumber = -1;
					//TODO: optimize the following loop using a parallel stream
					for (LearningAlgorithm algorithm : learningAlgorithms) {
						algorithmNumber++;
						
						parametersList = processListOfParameters(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
						int parameterNumber = -1;
						
						//TODO: optimize the following loop using a parallel stream
						for (LearningAlgorithmDataParameters parameters : parametersList) { //check all parameters from the list of parameters for current algorithm
							parameterNumber++;
	
							/**/long trainingStartTime = System.currentTimeMillis();
							//=====
							ClassificationModel model = algorithm.learn(processedFullData, parameters); //can change result of processedFullData.getInformationTable()
							//=====
							/**/long fullDataTrainingTime = System.currentTimeMillis() - trainingStartTime;
							
							/**/long validationStartTime = System.currentTimeMillis();
							//=====
							ModelValidationResult modelValidationResult = model.validate(fullData);
							//=====
							/**/long fullDataValidationTime = System.currentTimeMillis() - validationStartTime;
	
							/**/BatchExperimentResults.DataAlgorithmParametersSelector selector = (new BatchExperimentResults.DataAlgorithmParametersSelector())
							/**/	.dataSetNumber(dataSetNumber).learningAlgorithmNumber(algorithmNumber).parametersNumber(parameterNumber);
							/**/CalculationTimes fullDataCalculationTimes = batchExperimentResults.getFullDataCalculationTimes(selector);
							/**/fullDataCalculationTimes.increaseTotalTrainingTime(fullDataTrainingTime);
							/**/fullDataCalculationTimes.increaseTotalValidationTime(fullDataValidationTime);
							
							String validationSummary = model.getValidationSummary();
							algorithmNameWithParameters2Evaluations.put(algorithm.getName()+"("+parameters+")",
									new BatchExperimentResults.Evaluations(
											modelValidationResult.getOverallAccuracy(),
											modelValidationResult.getMainModelAccuracy(),
											modelValidationResult.getDefaultModelAccuracy(),
											modelValidationResult.getMainModelDecisionsRatio()
									));
							outN("Train data accuracy for parameterized algorithm '%1(%2)': %3 # %4 # %5. Main model decisions ratio: %6.\n%% %7 [Times]: training: %8 [ms], validation: %9 [ms].",
									algorithm.getName(),
									parameters,
									modelValidationResult.getOverallAccuracy(),
									modelValidationResult.getMainModelAccuracy(),
									modelValidationResult.getDefaultModelAccuracy(),
									modelValidationResult.getMainModelDecisionsRatio(),
									validationSummary,
									fullDataTrainingTime,
									fullDataValidationTime);
							outN("  /");
							outN(" /");
							outN("/");
							outN(model.getModelDescription());
							outN("\\");
							outN(" \\");
							outN("  \\");
							model = null; //facilitate GC
						}
					}
					
					//save quality of approximation and results of all parameterized algorithms for full data
					FullDataResults fullDataResults = new FullDataResults(consistencyThreshold2QualityOfApproximation, algorithmNameWithParameters2Evaluations);
					batchExperimentResults.storeFullDataResults(dataProvider.getDataName(), fullDataResults);
					
					VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(); //release references to allow GC
				}
				//<<<<<

				//>>>>> DO MULTIPLE CROSS-VALIDATIONS
				if (doCrossValidationsForProvider) {
					crossValidationsCount = dataProvider.getSeeds().length; //get number of cross-validations for current data
					for (int crossValidationNumber = 0; crossValidationNumber < crossValidationsCount; crossValidationNumber++) {
						t2 = b(resolveText("  Starting calculations for %1, cross-validation %2.", dataProvider.getDataName(), crossValidationNumber));
						data = dataProvider.provide(crossValidationNumber);
						
						crossValidation = crossValidationProvider.provide();
						crossValidation.setSeed(dataProvider.getSeeds()[crossValidationNumber]);
						crossValidation.setNumberOfFolds(dataProvider.getNumberOfFolds());
						
						final int streamCrossValidationNumber = crossValidationNumber;
						
						//for each (algorithm, parameters) pair initialize storage for results of particular folds
						for (int i = 0; i < learningAlgorithms.size(); i++) {
							parametersList = processListOfParameters(parametersContainer.getParameters(learningAlgorithms.get(i).getName(), dataProvider.getDataName()));
							for (int j = 0; j < parametersList.size(); j++) {
								initializingCVSelector = (new BatchExperimentResults.CVSelector())
										.dataSetNumber(dataSetNumber).learningAlgorithmNumber(i).parametersNumber(j).crossValidationNumber(crossValidationNumber);
								batchExperimentResults.initializeFoldResults(initializingCVSelector, data.getInformationTable().getOrderedUniqueFullyDeterminedDecisions(), crossValidation.getNumberOfFolds());
							}
						}
						
						crossValidationFolds = crossValidation.getStratifiedFolds(data);
						
						//run each fold in parallel!
						crossValidationFolds.parallelStream().forEach(fold -> {
							//long t3;
							//t3 = b("    Starting calculations for fold "+fold.getIndex()+".");
							//t3 = b(null);
							b(null);
							Data processedTrainData = trainDataPreprocessor.process(fold.getTrainData()); //e.g.: over-sampling, under-sampling, bagging
							
							//long t4;
							
							int learningAlgorithmNumber = -1;
							for (LearningAlgorithm algorithm : learningAlgorithms) {
								learningAlgorithmNumber++;
								//t4 = b("      Starting calculations for fold "+fold.getIndex()+", algorithm "+algorithm.getName()+".");
								//t4 = b(null);
								
								long t5;
								
								List<LearningAlgorithmDataParameters> _parametersList = processListOfParameters(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
								int parametersNumber = -1;
								for (LearningAlgorithmDataParameters parameters : _parametersList) { //check all parameters from the list of parameters for current algorithm
									parametersNumber++;
									t5 = b(null);
									
									/**/long trainingStartTime = System.currentTimeMillis();
									//=====
									ClassificationModel model = algorithm.learn(processedTrainData, parameters); //can change result of processedTrainData.getInformationTable()
									//=====
									/**/long foldTrainingTime = System.currentTimeMillis() - trainingStartTime;
									
									/**/long validationStartTime = System.currentTimeMillis();
									//=====
									ModelValidationResult modelValidationResult = model.validate(fold.getTestData());
									//=====
									/**/long foldValidationTime = System.currentTimeMillis() - validationStartTime;
									
									/**/BatchExperimentResults.DataAlgorithmParametersSelector selector = (new BatchExperimentResults.DataAlgorithmParametersSelector())
									/**/	.dataSetNumber(streamDataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber);
									/**/CalculationTimes totalFoldCalculationTimes = batchExperimentResults.getTotalFoldCalculationTimes(selector);
									/**/totalFoldCalculationTimes.increaseTotalTrainingTime(foldTrainingTime);
									/**/totalFoldCalculationTimes.increaseTotalValidationTime(foldValidationTime);
									
									String validationSummary = model.getValidationSummary();
									model = null; //facilitate GC
									
									BatchExperimentResults.CVSelector cvSelector = (new BatchExperimentResults.CVSelector())
											.dataSetNumber(streamDataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber).crossValidationNumber(streamCrossValidationNumber);
									batchExperimentResults.storeFoldModelValidationResult(cvSelector, fold.getIndex(), modelValidationResult);
									
									e(t5, resolveText("      %1End of fold %2, algorithm %3(%4). [Accuracy]: %5 # %6 # %7.\n      %8%% %9",
											foldNumber2Spaces(fold.getIndex()), fold.getIndex(), algorithm.getName(), parameters,
											modelValidationResult.getOverallAccuracy(), modelValidationResult.getMainModelAccuracy(), modelValidationResult.getDefaultModelAccuracy(),
											foldNumber2Spaces(fold.getIndex()), validationSummary));
								}
								
								//e(t4, resolveText("      %1Finishing calculations for fold %2, algorithm %3.", foldNumber2Spaces(fold.getIndex()), fold.getIndex(), algorithm.getName()));
							}
							
							VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(processedTrainData.getName()); //release references to allow GC
							processedTrainData = null; //facilitate GC
		
							//e(t3, "    Finishing calculations for fold "+fold.getIndex()+".");
							fold.done(); //facilitate GC
						});
					
						data = null; //facilitate GC
						crossValidation = null; //facilitate GC
						crossValidationFolds = null; //facilitate GC
						
						e(t2, resolveText("  Finishing calculations for %1, cross-validation %2.", dataProvider.getDataName(), crossValidationNumber));
						outN("  ----------");
						for (int learningAlgorithmNumber = 0; learningAlgorithmNumber < learningAlgorithms.size(); learningAlgorithmNumber++) {
							parametersList = processListOfParameters(parametersContainer.getParameters(learningAlgorithms.get(learningAlgorithmNumber).getName(), dataProvider.getDataName()));
							int parametersNumber = -1;
							for (LearningAlgorithmDataParameters parameters : parametersList) {
								parametersNumber++;
								CVSelector cvSelector = (new BatchExperimentResults.CVSelector())
										.dataSetNumber(dataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber).crossValidationNumber(crossValidationNumber);
								
								outN("  Avg. accuracy over folds for algorithm '%1(%2)': %3 # %4 # %5. Avg. main model decisions ratio: %6.",
										learningAlgorithms.get(learningAlgorithmNumber).getName(),
										parameters,
										batchExperimentResults.getAggregatedCVModelValidationResult(cvSelector).getOverallAccuracy(),
										batchExperimentResults.getAggregatedCVModelValidationResult(cvSelector).getMainModelAccuracy(),
										batchExperimentResults.getAggregatedCVModelValidationResult(cvSelector).getDefaultModelAccuracy(),
										batchExperimentResults.getAggregatedCVModelValidationResult(cvSelector).getMainModelDecisionsRatio());
							}
						}
						outN("  ----------");
						
						VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(); //release references to allow GC
					} //for crossValidationNumber
				} //if (doCrossValidations)
				//<<<<<

				e(t1, resolveText("Finishing calculations for data '%1'.", dataProvider.getDataName()));
				
				//>>>>> SUMMARIZE MULTIPLE CROSS-VALIDATIONS
				if (doCrossValidationsForProvider) {
					outN("==========");
					for (int learningAlgorithmNumber = 0; learningAlgorithmNumber < learningAlgorithms.size(); learningAlgorithmNumber++) {
						parametersList = processListOfParameters(parametersContainer.getParameters(learningAlgorithms.get(learningAlgorithmNumber).getName(), dataProvider.getDataName()));
						
						List<DataAlgorithmParametersSelector> bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>(); //initialize as an empty list
						double bestAccuracy = -1.0;
	
						int parametersNumber = -1;
						for (LearningAlgorithmDataParameters parameters : parametersList) {
							parametersNumber++;
							DataAlgorithmParametersSelector selector = (new DataAlgorithmParametersSelector())
									.dataSetNumber(dataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber);
							AverageEvaluations averageEvaluations = batchExperimentResults.getAverageDataAlgorithmParametersEvaluations(selector);
							outN("Avg. accuracy over cross-validations for algorithm '%1(%2)': %3 (stdDev: %4) # %5 (stdDev: %6) # %7 (stdDev: %8). Avg. main model decisions ratio: %9.",
									learningAlgorithms.get(learningAlgorithmNumber).getName(),
									parameters,
									averageEvaluations.getOverallAverageEvaluation().getAverage(),
									averageEvaluations.getOverallAverageEvaluation().getStdDev(),
									averageEvaluations.getMainModelAverageEvaluation().getAverage(),
									averageEvaluations.getMainModelAverageEvaluation().getStdDev(),
									averageEvaluations.getDefaultModelAverageEvaluation().getAverage(),
									averageEvaluations.getDefaultModelAverageEvaluation().getStdDev(),
									averageEvaluations.getMainModelDecisionsRatio());
							CalculationTimes totalFoldCalculationTimes = batchExperimentResults.getTotalFoldCalculationTimes(selector);
							outN("%% [Avg. fold calculation times]: training: %1 [ms], validation: %2 [ms]", totalFoldCalculationTimes.getAverageTrainingTime(), totalFoldCalculationTimes.getAverageValidationTime());
	
							AverageEvaluation averageEvaluation = useMainModelAccuracy ? averageEvaluations.getMainModelAverageEvaluation() : averageEvaluations.getOverallAverageEvaluation();
							if (averageEvaluation.getAverage() > bestAccuracy) { //better accuracy found
								bestAccuracy = averageEvaluation.getAverage();
								bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>();
								bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
							} else if (averageEvaluation.getAverage() == bestAccuracy) {
								bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
							}
						}
						
						//print the best parameters + accuracy for the current algorithm
						for (DataAlgorithmParametersSelector selector : bestAlgorithmParametersSelectors) {
							AverageEvaluations averageEvaluations = batchExperimentResults.getAverageDataAlgorithmParametersEvaluations(selector);
							outN("  Best avg. %1 accuracy over cross-validations for algorithm '%2(%3)': %4 (stdDev: %5) # %6 (stdDev: %7) # %8 (stdDev: %9). Avg. main model decisions ratio: %10.",
									useMainModelAccuracy ? "main model" : "overall",
									learningAlgorithms.get(learningAlgorithmNumber).getName(),
									parametersList.get(selector.parametersNumber),
									averageEvaluations.getOverallAverageEvaluation().getAverage(),
									averageEvaluations.getOverallAverageEvaluation().getStdDev(),
									averageEvaluations.getMainModelAverageEvaluation().getAverage(),
									averageEvaluations.getMainModelAverageEvaluation().getStdDev(),
									averageEvaluations.getDefaultModelAverageEvaluation().getAverage(),
									averageEvaluations.getDefaultModelAverageEvaluation().getStdDev(),
									averageEvaluations.getMainModelDecisionsRatio());
							
							CalculationTimes totalFoldCalculationTimes = batchExperimentResults.getTotalFoldCalculationTimes(selector);
							outN("  %% [Avg. fold calculation times]: training: %1 [ms], validation: %2 [ms]", totalFoldCalculationTimes.getAverageTrainingTime(), totalFoldCalculationTimes.getAverageValidationTime());
						}
					}
					outN("==========");
				} //if (doCrossValidations)
				//<<<<<
				outN();
			} //if (doCrossValidations || doFullDataReclassification)
			
			dataProvider.done(); //facilitate GC
			VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(); //release references to allow GC
		} //for dataProvider
		
		return batchExperimentResults;
	}
	
	public static void main(String[] args) {
		int k = 10; //number of folds
		
		//-----
		final String dataNameMonumentsNoMV = "zabytki";
		//-----
		final String dataNameMonumentsNoMV01 = "zabytki01";
		final String dataNameMonumentsNoMV01_K9_K10 = "zabytki01-K9-K10";
		//-----
		String dataNameChurn4000v8 = "bank-churn-4000-v8";
		//-----
		String dataNameChurn4000v8_0_05_mv2 = "bank-churn-4000-v8-0.05-mv2";
		String dataNameChurn4000v8_0_05_mv15 = "bank-churn-4000-v8-0.05-mv1.5";
		String dataNameChurn4000v8_0_10_mv2 = "bank-churn-4000-v8-0.10-mv2";
		String dataNameChurn4000v8_0_10_mv15 = "bank-churn-4000-v8-0.10-mv1.5";
		String dataNameChurn4000v8_0_15_mv2 = "bank-churn-4000-v8-0.15-mv2";
		String dataNameChurn4000v8_0_15_mv15 = "bank-churn-4000-v8-0.15-mv1.5";
		String dataNameChurn4000v8_0_20_mv2 = "bank-churn-4000-v8-0.20-mv2";
		String dataNameChurn4000v8_0_20_mv15 = "bank-churn-4000-v8-0.20-mv1.5";
		String dataNameChurn4000v8_0_25_mv2 = "bank-churn-4000-v8-0.25-mv2";
		String dataNameChurn4000v8_0_25_mv15 = "bank-churn-4000-v8-0.25-mv1.5";
		
//		long[] SKIP_DATA = new long[]{};
		
		//HINT: comment addition of data provider if given data set should not be used in this batch experiment OR give empty list array of seed
		List<DataProvider> dataProviders = new ArrayList<DataProvider>();
		//-----
//		dataProviders.add(new BasicDataProvider(
//				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
//				"data/csv/zabytki-data-noMV.csv",
//				false, ';',
//				dataNameMonumentsNoMV,
//				//SKIP_DATA,
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
		//-----
//		dataProviders.add(new BasicDataProvider(
//				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
//				"data/csv/zabytki-data-noMV-0-1.csv",
//				false, ';',
//				dataNameMonumentsNoMV01,
//				//SKIP_DATA,
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10.json",
//				"data/csv/zabytki-data-noMV-0-1.csv",
//				false, ';',
//				dataNameMonumentsNoMV01_K9_K10,
//				//SKIP_DATA,
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
		//-----
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata.json",
				"data/json-objects/bank-churn-4000-v8 data.json",
				dataNameChurn4000v8,
				//SKIP_DATA,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		//-----
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataNameChurn4000v8_0_05_mv2,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataNameChurn4000v8_0_05_mv15,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataNameChurn4000v8_0_10_mv2,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataNameChurn4000v8_0_10_mv15,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataNameChurn4000v8_0_15_mv2,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataNameChurn4000v8_0_15_mv15,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataNameChurn4000v8_0_20_mv2,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataNameChurn4000v8_0_20_mv15,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataNameChurn4000v8_0_25_mv2,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataNameChurn4000v8_0_25_mv15,
				//SKIP_DATA,
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		
		//HINT: comment algorithm addition if given algorithm should not be used in this batch experiment
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new VCDomLEMModeRuleClassifierLearner());
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new J48()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new NaiveBayes()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new SMO()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new RandomForest()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new MultilayerPerceptron()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OLM()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OSDL())); //does not work because of numerical attributes
		
		//HINT: there may be given lists of parameters for (algorithm-name, data-name) pairs for which there will be no calculations - they are just not used
		LearningAlgorithmDataParametersContainer parametersContainer = (new LearningAlgorithmDataParametersContainer())
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes")
						))
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes")
						))
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes")
						))
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"), //BEST w.r.t. overall accuracy when using default class
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"), //BEST w.r.t. main model accuracy
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_05_mv2,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_05_mv15,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_10_mv2,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_10_mv15,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_15_mv2,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_15_mv15,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_20_mv2,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_20_mv15,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_25_mv2,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameChurn4000v8_0_25_mv15,
						//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0") //BEST w.r.t. overall accuracy when using default class
						getVCDomLEMModeRuleClassifierLearnerChurn4000v8ParametersList())
				//-----
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(/*null, */new WEKAAlgorithmOptions("-D"))); //option -D means discretize numeric attributes
		
		//------------------------------------------------------------------------------------------------------------------------------
		
		parametersContainer.sortParametersLists(); //assure parameters for VCDomLEMModeRuleClassifierLearnerDataParameters algorithm are in ascending order w.r.t. consistency threshold
		
		BatchExperimentResults results = (new BatchExperiment(dataProviders, new RepeatableCrossValidationProvider(), new AcceptingDataProcessor(), learningAlgorithms, parametersContainer)).run();
		
		//------------------------------------------------------------------------------------------------------------------------------
		
		Function<String, Integer> d2i = (dataName) -> {
			return dataProviders.stream().map(provider -> provider.getDataName()).collect(Collectors.toList()).indexOf(dataName);
		}; //maps data name to data index at the dataProviders list
		Function<String, Integer> a2i = (algorithmName) -> {
			return learningAlgorithms.stream().map(algorithm -> algorithm.getName()).collect(Collectors.toList()).indexOf(algorithmName);
		}; //maps algorithm name to algorithm index at the learningAlgorithms list
		
		//------------------------------------------------------------------------------------------------------------------------------
		
		//get names of data sets for which there is a provider with non-empty list of seeds
		List<String> dataSetsNames = dataProviders.stream().filter(provider -> provider.getSeeds().length > 0).map(provider -> provider.getDataName()).collect(Collectors.toList());
		//get names of algorithms
		List<String> algorithmsNames = learningAlgorithms.stream().map(algorithm -> algorithm.getName()).collect(Collectors.toList());
		
		List<LearningAlgorithmDataParameters> parametersList;
		int parametersNumber;

		//print experiment summary:
		outN("####################");
		for (String dataSetName : dataSetsNames) {
			if (doFullDataReclassification) {
				outN(results.reportFullDataResults(dataSetName));
			} //if (doFullDataReclassification)
			
			if (doCrossValidations) {
				for (String algorithmName : algorithmsNames) {
					parametersList = processListOfParameters(parametersContainer.getParameters(algorithmName, dataSetName));
					parametersNumber = -1;
					List<DataAlgorithmParametersSelector> bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>(); //initialize as an empty list
					double bestAccuracy = -1.0;
					
					for (LearningAlgorithmDataParameters parameters : parametersList) { //check all parameters from the list of parameters for the current algorithm
						parametersNumber++;
						DataAlgorithmParametersSelector selector = (new DataAlgorithmParametersSelector())
								.dataSetNumber(d2i.apply(dataSetName)).learningAlgorithmNumber(a2i.apply(algorithmName)).parametersNumber(parametersNumber);
						AverageEvaluations averageEvaluations = results.getAverageDataAlgorithmParametersEvaluations(selector);
						outN("Avg. accuracy for ('%1', %2(%3)): %4 (stdDev: %5) # %6 (stdDev: %7) # %8 (stdDev: %9). Avg. main model decisions ratio: %10.",
								dataSetName, algorithmName, parameters,
								averageEvaluations.getOverallAverageEvaluation().getAverage(),
								averageEvaluations.getOverallAverageEvaluation().getStdDev(),
								averageEvaluations.getMainModelAverageEvaluation().getAverage(),
								averageEvaluations.getMainModelAverageEvaluation().getStdDev(),
								averageEvaluations.getDefaultModelAverageEvaluation().getAverage(),
								averageEvaluations.getDefaultModelAverageEvaluation().getStdDev(),
								averageEvaluations.getMainModelDecisionsRatio());
						
						CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
						outN("%% [Avg. fold calculation times]: training: %1, validation: %2", totalFoldCalculationTimes.getAverageTrainingTime(), totalFoldCalculationTimes.getAverageValidationTime());
	
						AverageEvaluation averageEvaluation = useMainModelAccuracy ? averageEvaluations.getMainModelAverageEvaluation() : averageEvaluations.getOverallAverageEvaluation();
						if (averageEvaluation.getAverage() > bestAccuracy) { //better accuracy found
							bestAccuracy = averageEvaluation.getAverage();
							bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>();
							bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
						} else if (averageEvaluation.getAverage() == bestAccuracy) {
							bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
						}
					}
					
					//print the best parameters + accuracy for the current algorithm
					for (DataAlgorithmParametersSelector selector : bestAlgorithmParametersSelectors) {
						AverageEvaluations averageEvaluations = results.getAverageDataAlgorithmParametersEvaluations(selector);
						outN("  Best avg. %1 accuracy for ('%2', %3(%4)): %5 (stdDev: %6) # %7 (stdDev: %8) # %9 (stdDev: %10). Avg. main model decisions ratio: %11.",
								useMainModelAccuracy ? "main model" : "overall",
								dataSetName, algorithmName, parametersList.get(selector.parametersNumber),
								averageEvaluations.getOverallAverageEvaluation().getAverage(),
								averageEvaluations.getOverallAverageEvaluation().getStdDev(),
								averageEvaluations.getMainModelAverageEvaluation().getAverage(),
								averageEvaluations.getMainModelAverageEvaluation().getStdDev(),
								averageEvaluations.getDefaultModelAverageEvaluation().getAverage(),
								averageEvaluations.getDefaultModelAverageEvaluation().getStdDev(),
								averageEvaluations.getMainModelDecisionsRatio());
						
						CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
						outN("  %% [Avg. fold calculation times]: training: %1, validation: %2", totalFoldCalculationTimes.getAverageTrainingTime(), totalFoldCalculationTimes.getAverageValidationTime());
					}
				} //for
			} //if (doCrossValidations)
			outN("####################");
		}
		
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

//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
				
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)

//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
	
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0225, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0275, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.03, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0",	new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
	
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0325, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.035, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0375, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("support >= 1"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
//		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")) //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02 & confidence > 0.6666"), "0", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")), //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		new VCDomLEMModeRuleClassifierLearnerDataParameters(0.04, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025 & confidence > 0.6666"), "0",	new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D")) //provide default class using trained NaiveBayes classifier with options "-D" (i.e., discretize numeric attributes)
		);
	}
	
}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.experiments.BatchExperimentResults.CVSelector;
import org.rulelearn.experiments.BatchExperimentResults.CalculationTimes;
import org.rulelearn.experiments.BatchExperimentResults.DataAlgorithmParametersSelector;
import org.rulelearn.experiments.BatchExperimentResults.FullDataModelValidationResult;
import org.rulelearn.experiments.BatchExperimentResults.FullDataResults;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;
import org.rulelearn.experiments.ModelValidationResult.MeansAndStandardDeviations;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

import keel.Algorithms.Monotonic_Classification.MoNGEL.MoNGEL;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.misc.OSDL;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OLM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Batch repeated cross-validation experiment over multiple data sets, with pre-processing of learning data, and different parameterized learning methods.
 * Each data set used in the experiment need to be an ordinal classification problem (with order among decisions).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperiment {
	
	public enum Churn4000v8DataSetVersion {
		NORMAL,
		OLM_OSDL, //tells if versions of data used only by OLM and OSDL should be used (if true, then make sure that OLM and OSDL are the only tested algorithms)
		MONGEL_NUM_OF_PRODUCTS_2X_GAIN, //tells if versions of data used only by MoNGEL should be used (if true, then make sure that MoNGEL is the only tested algorithm)
		MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER, //tells if versions of data used only by MoNGEL should be used (if true, then make sure that MoNGEL is the only tested algorithm)
		MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION, //tells if versions of data used only by MoNGEL should be used (if true, then make sure that MoNGEL is the only tested algorithm)
		MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER //gives the best results for MoNGEL!
	}

	List<DataProvider> dataProviders;
	CrossValidationProvider crossValidationProvider;
	DataProcessor trainDataPreprocessor;
	List<LearningAlgorithm> learningAlgorithms;
	LearningAlgorithmDataParametersContainer parametersContainer;
	
	//<BEGIN EXPERIMENT CONFIG>
	//TODO: configure?
	static boolean useMainModelAccuracy = false; //true = use main model accuracy; false = use overall accuracy
	static final boolean doFullDataReclassification = true;
	static final boolean doCrossValidations = true; //true = perform CVs; false = skip CVs
//	static final boolean generalizeConditions = true;
	static final boolean checkConsistencyOfTestDataDecisions = true;
	static final boolean printTrainedClassifiers = true; //concerns WEKA and KEEL classifiers + full data reclassification
	static final String decimalFormat = "%.5f"; //tells number of decimal places
	static final String percentDecimalFormat = "%.3f"; //tells number of decimal places in percentages
	
	static final boolean foldsInParallel = true; //false => folds will be done sequentially (useful only to measure more accurately avg. calculation times)
	
	//static final Churn4000v8DataSetVersion dataSetVersion = Churn4000v8DataSetVersion.MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER;
	static final Churn4000v8DataSetVersion dataSetVersion = Churn4000v8DataSetVersion.NORMAL;
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
	
	static String prepareText(String text, String prefix) {
		return text.replaceAll("%p", prefix).replaceAll("%n", System.lineSeparator());
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
				(InformationTableWithDecisionDistributions)informationTable : new InformationTableWithDecisionDistributions(informationTable, true, true));
		
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
		
		BatchExperimentResults results = (new BatchExperimentResults.Builder())
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
					Map<String, FullDataModelValidationResult> algorithmNameWithParameters2Evaluations = new LinkedHashMap<String, FullDataModelValidationResult>();
					Map<Double, Double> consistencyThreshold2QualityOfApproximation = new LinkedHashMap<Double, Double>();
					
					outN("--");
					
					//print full data set accuracies
					Data fullData = dataProvider.provideOriginalData(); //gets InformationTableWithDecisionDistributions, which involves time-consuming transformation from InformationTable read from 2 files
					outN("Quality of approximation for consistency threshold=%1: %2.", epsilonDRSAConsistencyThreshold, qualityOfDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), 0.0));
					consistencyThreshold2QualityOfApproximation.put(Double.valueOf(epsilonDRSAConsistencyThreshold), qualityOfDRSAApproximation);
					
					//if rule classifier is used for any data (thus in particular for the current data)
					if (learningAlgorithms.stream().filter(algorithm -> algorithm.getName().equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())).collect(Collectors.toList()).size() > 0) {
						
						parametersList = parametersContainer.getParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataProvider.getDataName()); //get list of parameters for rule classifier
						
						if (parametersList != null) { //VC-DRSA rule classifier has at least one parameter (although the algorithm itself may not be on the list of considered algorithms)
							for (LearningAlgorithmDataParameters parameters : parametersList) { //check quality of approximation for all considered consistency thresholds
								double consistencyThreshold = Double.valueOf(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.consistencyThresholdParameterName));
								if (!consistencyThreshold2QualityOfApproximation.containsKey(Double.valueOf(consistencyThreshold))) { //ensure that quality of approximation is calculated for each consistency threshold only once
									outN("Quality of approximation for consistency threshold=%1: %2.", consistencyThreshold, qualityOfVCDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), consistencyThreshold));
									consistencyThreshold2QualityOfApproximation.put(consistencyThreshold, qualityOfVCDRSAApproximation);
								}
							}
						}
					}
					
					outN("--");
					
					//calculate and process full data models for all (algorithm, parameters) pairs
					Data processedFullData = trainDataPreprocessor.process(fullData); //processedFullData should have the same informationTableTransformationTime!
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
							fullDataTrainingTime -= model.getModelLearningStatistics().getTotalStatisticsCountingTime();
							fullDataTrainingTime += model.getModelLearningStatistics().getTotalModelCalculationTimeSavedByUsingCache();
							if (algorithm.getName().equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())) {
								fullDataTrainingTime += processedFullData.getInformationTableTransformationTime(); //add time of data transformation (done out of time measurement zone marked by /**/), as VC-DRSA rule model needs this transformation!
								model.getModelLearningStatistics().totalDataTransformationTime = processedFullData.getInformationTableTransformationTime(); //set proper information table transformation time
							}
							
							/**/long validationStartTime = System.currentTimeMillis();
							//=====
							ModelValidationResult modelValidationResult = model.validate(fullData);
							//=====
							/**/long fullDataValidationTime = System.currentTimeMillis() - validationStartTime;
							fullDataValidationTime -= modelValidationResult.getClassificationStatistics().getTotalStatisticsCountingTime();
							fullDataValidationTime -= modelValidationResult.getModelDescription().getModelDescriptionCalculationTime();
	
							/**/BatchExperimentResults.DataAlgorithmParametersSelector selector = (new BatchExperimentResults.DataAlgorithmParametersSelector())
							/**/	.dataSetNumber(dataSetNumber).learningAlgorithmNumber(algorithmNumber).parametersNumber(parameterNumber);
							/**/CalculationTimes fullDataCalculationTimes = results.getFullDataCalculationTimes(selector);
							/**/fullDataCalculationTimes.increaseTotalTrainingTime(fullDataTrainingTime);
							/**/fullDataCalculationTimes.increaseTotalValidationTime(fullDataValidationTime);
							
							algorithmNameWithParameters2Evaluations.put(algorithm.getName()+"("+parameters+")", new FullDataModelValidationResult(selector, modelValidationResult));
							
							ClassificationStatistics classificationStatistics = modelValidationResult.getClassificationStatistics();
							
							//OUTPUT
							outN("Train data accuracy for '%1(%2)': "+System.lineSeparator()+
									"%3 (overall: %4, avg: %5) # %6 # %7 (%8|%9). Main model decisions ratio: %10."+System.lineSeparator()+
									"%% [Learning]: %11."+System.lineSeparator()+
									"%12"+System.lineSeparator()+
									"%% [Times]: training: %13 [ms], validation: %14 [ms].",
									algorithm.getName(),
									parameters,
									round(modelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
									round(classificationStatistics.getOverallAccuracy()), //test if the same as above
									round(classificationStatistics.getAvgAccuracy()), //test if the same as above
									round(classificationStatistics.getMainModelAccuracy()),
									round(classificationStatistics.getDefaultModelAccuracy()),
									round(classificationStatistics.getDefaultClassAccuracy()),
									round(classificationStatistics.getDefaultClassifierAccuracy()),
									round(classificationStatistics.getMainModelDecisionsRatio()),
									modelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> (new StringBuilder("%% ")).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())), //print validation summary in several lines
									fullDataTrainingTime,
									fullDataValidationTime);
							outN("  /");
							outN(" /");
							outN("/");
							outN("[Model]: "+model.getModelDescription().toString());
							outN("\\");
							outN(" \\");
							outN("  \\");
							model = null; //facilitate GC
						}
					}
					
					//save quality of approximation and results of all parameterized algorithms for full data
					FullDataResults fullDataResults = new FullDataResults(consistencyThreshold2QualityOfApproximation, algorithmNameWithParameters2Evaluations);
					results.storeFullDataResults(dataProvider.getDataName(), fullDataResults);
					
					VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(); //release references to allow GC
					
					outN("@@@@@ [BEGIN] Full train data reports:");
					//OUTPUT
					out(results.reportFullDataResults(dataProvider.getDataName()));
					outN("@@@@@ [END]");
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
								results.initializeFoldResults(initializingCVSelector, data.getInformationTable().getOrderedUniqueFullyDeterminedDecisions(), crossValidation.getNumberOfFolds());
							}
						}
						
						crossValidationFolds = crossValidation.getStratifiedFolds(data);
						
//						Object[][][][] foldResults = new Object[crossValidationFolds.size()][learningAlgorithms.size()][maxParametersCount][7];
						
						//run certain number of folds in parallel or sequentially
						Stream<CrossValidationFold> foldsStream = foldsInParallel ? crossValidationFolds.parallelStream() : crossValidationFolds.stream();
						foldsStream.forEach(fold -> {       //just for measuring time!
							String linePrefix = "      "+foldNumber2Spaces(fold.getIndex());
							String summaryLinePrefix = linePrefix + "%% ";
							String messageTemplate = prepareText("%pEnd of fold %1, algorithm %2(%3).%n"
									+ "%p%% [Accuracy]: %4 (overall: %5, avg: %6) # %7 # %8 (%9|%10). Main model decisions ratio: %11.%n"
									+ "%12%n"
									+ "%p%% [Duration]: %13 [ms].", linePrefix); //%p will be replaced by prefix, %n by new line
							
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
									foldTrainingTime -= model.getModelLearningStatistics().getTotalStatisticsCountingTime();
									foldTrainingTime += model.getModelLearningStatistics().getTotalModelCalculationTimeSavedByUsingCache();
									//here transformation of information table is done inside time measurement zone, so no correction is necessary
									
									/**/long validationStartTime = System.currentTimeMillis();
									//=====
									ModelValidationResult modelValidationResult = model.validate(fold.getTestData());
									//=====
									/**/long foldValidationTime = System.currentTimeMillis() - validationStartTime;
									foldValidationTime -= modelValidationResult.getClassificationStatistics().getTotalStatisticsCountingTime();
									foldValidationTime -= modelValidationResult.getModelDescription().getModelDescriptionCalculationTime();
									
									/**/BatchExperimentResults.DataAlgorithmParametersSelector selector = (new BatchExperimentResults.DataAlgorithmParametersSelector())
									/**/	.dataSetNumber(streamDataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber);
									/**/CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
									/**/totalFoldCalculationTimes.increaseTotalTrainingTime(foldTrainingTime);
									/**/totalFoldCalculationTimes.increaseTotalValidationTime(foldValidationTime);
									
									String classificationStatisticsAsText = modelValidationResult.getClassificationStatistics().toString();
									model = null; //facilitate GC
									
									BatchExperimentResults.CVSelector cvSelector = (new BatchExperimentResults.CVSelector())
											.dataSetNumber(streamDataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber).crossValidationNumber(streamCrossValidationNumber);
									results.storeFoldModelValidationResult(cvSelector, fold.getIndex(), modelValidationResult);
									
//									e(t5, resolveText("      %1End of fold %2, algorithm %3(%4)."+System.lineSeparator()+
//													  "      %5%% [Accuracy]: %6 # %7 # %8."+System.lineSeparator()+
//													  "%9",
//											foldNumber2Spaces(fold.getIndex()), fold.getIndex(), algorithm.getName(), parameters,
//											foldNumber2Spaces(fold.getIndex()),
//											round(modelValidationResult.getOverallAccuracy()), round(modelValidationResult.getMainModelAccuracy()), round(modelValidationResult.getDefaultModelAccuracy()),
//											Arrays.asList(classificationStatisticsAsText.split(System.lineSeparator())).stream()
//											.map(line -> (new StringBuilder("      ")).append(foldNumber2Spaces(fold.getIndex())).append("%% ").append(line).toString())
//											.collect(Collectors.joining(System.lineSeparator())) //print validation summary in several lines
//									));
									
									long duration = System.currentTimeMillis() - t5;
									//OUTPUT
									outN(resolveText(messageTemplate, //"End of fold" message
											fold.getIndex(), algorithm.getName(), parameters,
											round(modelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
											round(modelValidationResult.getClassificationStatistics().getOverallAccuracy()), //test if the same as above
											round(modelValidationResult.getClassificationStatistics().getAvgAccuracy()), //test if the same as above
											round(modelValidationResult.getClassificationStatistics().getMainModelAccuracy()),
											round(modelValidationResult.getClassificationStatistics().getDefaultModelAccuracy()),
											round(modelValidationResult.getClassificationStatistics().getDefaultClassAccuracy()),
											round(modelValidationResult.getClassificationStatistics().getDefaultClassifierAccuracy()),
											round(modelValidationResult.getClassificationStatistics().getMainModelDecisionsRatio()),
											Arrays.asList(classificationStatisticsAsText.split(System.lineSeparator())).stream()
											.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
											.collect(Collectors.joining(System.lineSeparator())), //print validation summary in several lines
											duration));
									
//									String template;
//									int foldIndex; //%1
//									LearningAlgorithm algorithm; //%2
//									LearningAlgorithmDataParameters parameters; //%3
//									double overallAccuracy; //%4
//									double mainModelAccuracy; //%5
//									double defaultModelAccuracy; //%6
//									ClassificationStatistics classificationStatistics; //%7
//									long duration; //%8
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
								ModelValidationResult aggregatedCVModelValidationResult = results.getAggregatedCVModelValidationResult(cvSelector);
								ClassificationStatistics classificationStatistics = aggregatedCVModelValidationResult.getClassificationStatistics();
								
								//OUTPUT
								outN("  Avg. accuracy over folds for algorithm '%1(%2)': "+System.lineSeparator()+
										"    %3 (overall: %4, avg: %5) # %6 # %7 (%8|%9). Avg. main model decisions ratio: %10.",
										learningAlgorithms.get(learningAlgorithmNumber).getName(),
										parameters,
										round(aggregatedCVModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
										round(classificationStatistics.getOverallAccuracy()), //test if the same as above
										round(classificationStatistics.getAvgAccuracy()), //test if the same as above
										round(classificationStatistics.getMainModelAccuracy()),
										round(classificationStatistics.getDefaultModelAccuracy()),
										round(classificationStatistics.getDefaultClassAccuracy()),
										round(classificationStatistics.getDefaultClassifierAccuracy()),
										round(classificationStatistics.getMainModelDecisionsRatio()) );
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
							ModelValidationResult aggregatedModelValidationResult = results.getAggregatedModelValidationResult(selector);
							ClassificationStatistics classificationStatistics = aggregatedModelValidationResult.getClassificationStatistics();
							MeansAndStandardDeviations meansAndStandardDeviations = classificationStatistics.getMeansAndStandardDeviations();
							CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
							
//							//+++++
//							String info;
//							String qualitiesOfApproximation = classificationStatistics.getQualitiesOfApproximation();
//							StringBuilder infoBuilder = (new StringBuilder(128)).append(aggregatedModelValidationResult.getModelDescription());
//							switch (classificationStatistics.getClassifierType()) {
//							case VCDRSA_RULES_CLASSIFIER:
//								infoBuilder.append("; ").append(String.format(Locale.US, "%s: %.2f", ModeRuleClassifier.avgNumberOfCoveringRulesIndicator, classificationStatistics.getAverageNumberOfCoveringRules()));
//								if (!qualitiesOfApproximation.equals("")) {
//									infoBuilder.append("; ").append(qualitiesOfApproximation);
//								}
//								infoBuilder.append(".");
//								break;
//							case WEKA_CLASSIFIER:
//								if (!qualitiesOfApproximation.equals("")) {
//									infoBuilder.append(" ").append(qualitiesOfApproximation).append(".");
//								}
//								break;
//							default:
//								throw new InvalidValueException("Incorrect classifier type.");
//							}
//							info = infoBuilder.toString();
//							//+++++
							
							String summaryLinePrefix = "  %% ";
							
							//OUTPUT
							outN("Avg. accuracy over CVs for algorithm '%1(%2)': "+System.lineSeparator()+
									"  %3 (stdDev: %4) (overall: %5 (stdDev: %6) | avg: %7) # %8 (stdDev: %9) # %10 (stdDev: %11) (%12 (stdDev: %13) | %14 (stdDev: %15)). Avg. main model decisions ratio: %16. "+System.lineSeparator()+
									"  %% [Learning]: %17"+System.lineSeparator()+
									"%18"+System.lineSeparator()+
									"  %% [Model]: %19."+System.lineSeparator()+
									"  %% [Avg. fold calculation times]: training: %20 [ms], validation: %21 [ms]",
									learningAlgorithms.get(learningAlgorithmNumber).getName(),
									parameters,
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getDeviationOfAccuracy()),
									round(meansAndStandardDeviations.getOverallAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getOverallAverageAccuracy().getStdDev()),
									round(aggregatedModelValidationResult.getClassificationStatistics().getAvgAccuracy()),
									round(meansAndStandardDeviations.getMainModelAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getMainModelAverageAccuracy().getStdDev()),
									round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getStdDev()),
									round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getStdDev()),
									round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getStdDev()),
									round(aggregatedModelValidationResult.getClassificationStatistics().getMainModelDecisionsRatio()),
									aggregatedModelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())),
//									info,
									aggregatedModelValidationResult.getModelDescription().toShortString(),
									round(totalFoldCalculationTimes.getAverageTrainingTime()),
									round(totalFoldCalculationTimes.getAverageValidationTime())
								);
	
							MeanAndStandardDeviation averageAccuracy = useMainModelAccuracy ?
									meansAndStandardDeviations.getMainModelAverageAccuracy() :
									meansAndStandardDeviations.getOverallAverageAccuracy();
							if (averageAccuracy.getMean() > bestAccuracy) { //better accuracy found
								bestAccuracy = averageAccuracy.getMean();
								bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>();
								bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
							} else if (averageAccuracy.getMean() == bestAccuracy) {
								bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
							}
						} //for
						
						//print the best parameters + accuracy for the current algorithm
						if (parametersList.size() > 1) {
							outN();
							
							for (DataAlgorithmParametersSelector selector : bestAlgorithmParametersSelectors) {
								ModelValidationResult aggregatedModelValidationResult = results.getAggregatedModelValidationResult(selector);
								ClassificationStatistics classificationStatistics = aggregatedModelValidationResult.getClassificationStatistics();
								MeansAndStandardDeviations meansAndStandardDeviations = classificationStatistics.getMeansAndStandardDeviations();
								CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
								
//								//+++++
//								String info;
//								String qualitiesOfApproximation = classificationStatistics.getQualitiesOfApproximation();
//								StringBuilder infoBuilder = (new StringBuilder(128)).append(aggregatedModelValidationResult.getModelDescription());
//								switch (classificationStatistics.getClassifierType()) {
//								case VCDRSA_RULES_CLASSIFIER:
//									infoBuilder.append("; ").append(String.format(Locale.US, "%s: %.2f", ModeRuleClassifier.avgNumberOfCoveringRulesIndicator, classificationStatistics.getAverageNumberOfCoveringRules()));
//									if (!qualitiesOfApproximation.equals("")) {
//										infoBuilder.append("; ").append(qualitiesOfApproximation);
//									}
//									infoBuilder.append(".");
//									break;
//								case WEKA_CLASSIFIER:
//									if (!qualitiesOfApproximation.equals("")) {
//										infoBuilder.append(" ").append(qualitiesOfApproximation).append(".");
//									}
//									break;
//								default:
//									throw new InvalidValueException("Incorrect classifier type.");
//								}
//								info = infoBuilder.toString();
//								//+++++
								
								String summaryLinePrefix = "    %% ";
								String accuracyType = useMainModelAccuracy ? "main model" : "overall";
								
								//OUTPUT
								outN("  Best avg. "+accuracyType+" accuracy over cross-validations for algorithm '%1(%2)': "+System.lineSeparator()+
									 "    %3 (stdDev: %4) (overall: %5 (stdDev: %6) | avg: %7) # %8 (stdDev: %9) # %10 (stdDev: %11) (%12 (stdDev: %13) | %14 (stdDev: %15)). Avg. main model decisions ratio: %16. "+System.lineSeparator()+
									 "    %% [Learning]: %17"+System.lineSeparator()+
									 "%18"+System.lineSeparator()+
									 "    %% [Model]: %19."+System.lineSeparator()+
									 "    %% [Avg. fold calculation times]: training: %20 [ms], validation: %21 [ms]",
										learningAlgorithms.get(learningAlgorithmNumber).getName(),
										parametersList.get(selector.parametersNumber),
										round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
										round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getDeviationOfAccuracy()),
										round(meansAndStandardDeviations.getOverallAverageAccuracy().getMean()),
										round(meansAndStandardDeviations.getOverallAverageAccuracy().getStdDev()),
										round(aggregatedModelValidationResult.getClassificationStatistics().getAvgAccuracy()),
										round(meansAndStandardDeviations.getMainModelAverageAccuracy().getMean()),
										round(meansAndStandardDeviations.getMainModelAverageAccuracy().getStdDev()),
										round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getMean()),
										round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getStdDev()),
										round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getMean()),
										round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getStdDev()),
										round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getMean()),
										round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getStdDev()),
										round(aggregatedModelValidationResult.getClassificationStatistics().getMainModelDecisionsRatio()),
										aggregatedModelValidationResult.getModelLearningStatistics().toString(),
										Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
										.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
										.collect(Collectors.joining(System.lineSeparator())),
//										info,
										aggregatedModelValidationResult.getModelDescription().toShortString(),
										round(totalFoldCalculationTimes.getAverageTrainingTime()),
										round(totalFoldCalculationTimes.getAverageValidationTime())
									);
							} //for
							outN("--");
						} else {
							outN("--");
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
		
		return results;
	}
	
	public static void main(String[] args) {
		int k = 10; //number of folds
		
		//-----
		final String dataNameMonumentsNoMV = "zabytki";
		//-----
		final String dataNameMonumentsNoMV_K9_K10 = "zabytki-K9-K10";
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
		
		//HINT: comment addition of data provider if given data set should not be used in this batch experiment OR give empty array of seeds
		//TODO: comment data sets not used in the experiment
		List<DataProvider> dataProviders = new ArrayList<DataProvider>();

//		dataProviders.add(new BasicDataProvider(
//				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
//				"data/csv/zabytki-data-noMV.csv",
//				false, ';',
//				dataNameMonumentsNoMV,
//				//SKIP_DATA,
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
//		/*-----*/
//		dataProviders.add(new BasicDataProvider(
//				"data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10.json",
//				"data/csv/zabytki-data-noMV.csv",
//				false, ';',
//				dataNameMonumentsNoMV_K9_K10,
//				//SKIP_DATA,
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
//		/*-----*/
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
//		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		long[] churn4000v8Seeds = new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L};
		//long[] churn4000v8Seeds = new long[]{0L, 5488762120989881L, 4329629961476882L}; //only first 3 CVs
		
		dataProviders.add(getDataProviderChurn4000v8(
				dataNameChurn4000v8,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		dataProviders.add(getDataProviderChurn4000v8_0_05_mv2(
				dataNameChurn4000v8_0_05_mv2,
				churn4000v8Seeds,
				k));
		
		dataProviders.add(getDataProviderChurn4000v8_0_05_mv15(
				dataNameChurn4000v8_0_05_mv15,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		dataProviders.add(getDataProviderChurn4000v8_0_10_mv2(
				dataNameChurn4000v8_0_10_mv2,
				churn4000v8Seeds,
				k));
		
		dataProviders.add(getDataProviderChurn4000v8_0_10_mv15(
				dataNameChurn4000v8_0_10_mv15,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		dataProviders.add(getDataProviderChurn4000v8_0_15_mv2(
				dataNameChurn4000v8_0_15_mv2,
				churn4000v8Seeds,
				k));
		
		dataProviders.add(getDataProviderChurn4000v8_0_15_mv15(
				dataNameChurn4000v8_0_15_mv15,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		dataProviders.add(getDataProviderChurn4000v8_0_20_mv2(
				dataNameChurn4000v8_0_20_mv2,
				churn4000v8Seeds,
				k));
		
		dataProviders.add(getDataProviderChurn4000v8_0_20_mv15(
				dataNameChurn4000v8_0_20_mv15,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		dataProviders.add(getDataProviderChurn4000v8_0_25_mv2(
				dataNameChurn4000v8_0_25_mv2,
				churn4000v8Seeds,
				k));
		
		dataProviders.add(getDataProviderChurn4000v8_0_25_mv15(
				dataNameChurn4000v8_0_25_mv15,
				churn4000v8Seeds,
				k));
		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		
		//TODO: comment algorithms that should not be used in this batch experiment
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new VCDomLEMModeRuleClassifierLearner());
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new J48()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new NaiveBayes()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new SMO()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new RandomForest()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new MultilayerPerceptron()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new JRip()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OLM()));
//		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OSDL())); //weka.core.UnsupportedAttributeTypeException: weka.classifiers.misc.OSDL: Cannot handle numeric attributes!
//		learningAlgorithms.add(new MoNGELClassifierLerner());
		
		//HINT: there may be given lists of parameters for (algorithm-name, data-name) pairs for which there will be no calculations - they are just not used
		LearningAlgorithmDataParametersContainer parametersContainer = new LearningAlgorithmDataParametersContainer();
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				//PARAMETERS FOR MONUMENTS DATA
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				//-----
		parametersContainer
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
								//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
						))
						
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV_K9_K10,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
								//new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
						))
				//-----
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
						))
				.putParameters(VCDomLEMModeRuleClassifierLearner.getAlgorithmName(), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", true),
								new VCDomLEMModeRuleClassifierLearnerDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes", new WEKAClassifierLearner(() -> new NaiveBayes()), new WEKAAlgorithmOptions("-D"), true)
						));
				//-----
		parametersContainer
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV,
						Arrays.asList(null, new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				//------
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV_K9_K10,
						Arrays.asList(null, new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				//------
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV01,
						Arrays.asList(null, new WEKAAlgorithmOptions("-D"))) //option -D means discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameMonumentsNoMV01_K9_K10,
						Arrays.asList(null, new WEKAAlgorithmOptions("-D"))); //option -D means discretize numeric attributes
				//-----
				
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				//PARAMETERS FOR CHURN4000v8 DATA
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		parametersContainer
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
				//-----
		parametersContainer
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
				//-----
		parametersContainer
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
				//-----
		if (learningAlgorithms.stream().filter(a -> a.getName() == KEELClassifierLerner.getAlgorithmName(MoNGEL.class)).collect(Collectors.toList()).size() > 0) { // MoNGEL is on the list of algorithms
			parametersContainer
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_05_mv2))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
				))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_05_mv15))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_10_mv2))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_10_mv15))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_15_mv2))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_15_mv15))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_20_mv2))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_20_mv15))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_25_mv2))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						))
				.putParameters(KEELClassifierLerner.getAlgorithmName(MoNGEL.class), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(
								new KEELAlgorithmDataParameters(new AttributeRanges(
										dataProviders.stream()
											.filter(p -> p.getDataName().equals(dataNameChurn4000v8_0_25_mv15))
											.limit(1)
											.collect(Collectors.toList())
											.get(0).previewOriginalData().getInformationTable() ))
						));
		} //if	
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
						ModelValidationResult aggregatedModelValidationResult = results.getAggregatedModelValidationResult(selector);
						ClassificationStatistics classificationStatistics = aggregatedModelValidationResult.getClassificationStatistics();
						MeansAndStandardDeviations meansAndStandardDeviations = classificationStatistics.getMeansAndStandardDeviations();
						CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
						
//						//+++++
//						String info;
//						String qualitiesOfApproximation = classificationStatistics.getQualitiesOfApproximation();
//						StringBuilder infoBuilder = (new StringBuilder(128)).append(aggregatedModelValidationResult.getModelDescription());
//						switch (classificationStatistics.getClassifierType()) {
//						case VCDRSA_RULES_CLASSIFIER:
//							infoBuilder.append("; ").append(String.format(Locale.US, "%s: %.2f", ModeRuleClassifier.avgNumberOfCoveringRulesIndicator, classificationStatistics.getAverageNumberOfCoveringRules()));
//							if (!qualitiesOfApproximation.equals("")) {
//								infoBuilder.append("; ").append(qualitiesOfApproximation);
//							}
//							infoBuilder.append(".");
//							break;
//						case WEKA_CLASSIFIER:
//							if (!qualitiesOfApproximation.equals("")) {
//								infoBuilder.append(" ").append(qualitiesOfApproximation).append(".");
//							}
//							break;
//						default:
//							throw new InvalidValueException("Incorrect classifier type.");
//						}
//						info = infoBuilder.toString();
//						//+++++
						
						String summaryLinePrefix = "  %% ";
						
						//OUTPUT
						outN("Avg. accuracy for ('%1', %2(%3)): "+System.lineSeparator()+
								"  %4 (stdDev: %5) (overall: %6 (stdDev: %7) | avg: %8) # %9 (stdDev: %10) # %11 (stdDev: %12) (%13 (stdDev: %14) | %15 (stdDev: %16)). Avg. main model decisions ratio: %17. "+System.lineSeparator()+
								"  %% [Learning]: %18"+System.lineSeparator()+
								"%19"+System.lineSeparator()+
								"  %% [Model]: %20."+System.lineSeparator()+
								"  %% [Avg. fold calculation times]: training: %21, validation: %22",
								dataSetName, algorithmName, parameters,
								round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()), //
								round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getDeviationOfAccuracy()), //
								round(meansAndStandardDeviations.getOverallAverageAccuracy().getMean()),
								round(meansAndStandardDeviations.getOverallAverageAccuracy().getStdDev()),
								round(aggregatedModelValidationResult.getClassificationStatistics().getAvgAccuracy()),
								round(meansAndStandardDeviations.getMainModelAverageAccuracy().getMean()),
								round(meansAndStandardDeviations.getMainModelAverageAccuracy().getStdDev()),
								round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getMean()),
								round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getStdDev()),
								round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getMean()), //
								round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getStdDev()), //
								round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getMean()), //
								round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getStdDev()), //
								round(aggregatedModelValidationResult.getClassificationStatistics().getMainModelDecisionsRatio()),
								aggregatedModelValidationResult.getModelLearningStatistics().toString(),
								Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
								.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
								.collect(Collectors.joining(System.lineSeparator())),
//								info,
								aggregatedModelValidationResult.getModelDescription().toShortString(),
								round(totalFoldCalculationTimes.getAverageTrainingTime()),
								round(totalFoldCalculationTimes.getAverageValidationTime())
							);
	
						MeanAndStandardDeviation averageAccuracy = useMainModelAccuracy ?
								meansAndStandardDeviations.getMainModelAverageAccuracy() :
								meansAndStandardDeviations.getOverallAverageAccuracy();
						if (averageAccuracy.getMean() > bestAccuracy) { //better accuracy found
							bestAccuracy = averageAccuracy.getMean();
							bestAlgorithmParametersSelectors = new ArrayList<DataAlgorithmParametersSelector>();
							bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
						} else if (averageAccuracy.getMean() == bestAccuracy) {
							bestAlgorithmParametersSelectors.add(new DataAlgorithmParametersSelector(selector));
						}
					} //for
					
					//print the best parameters + accuracy for the current algorithm
					if (parametersList.size() > 1) {
						outN();
						
						for (DataAlgorithmParametersSelector selector : bestAlgorithmParametersSelectors) {
							ModelValidationResult aggregatedModelValidationResult = results.getAggregatedModelValidationResult(selector);
							ClassificationStatistics classificationStatistics = aggregatedModelValidationResult.getClassificationStatistics();
							MeansAndStandardDeviations meansAndStandardDeviations = classificationStatistics.getMeansAndStandardDeviations();
							CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
							
//							//+++++
//							String info;
//							String qualitiesOfApproximation = classificationStatistics.getQualitiesOfApproximation();
//							StringBuilder infoBuilder = (new StringBuilder(128)).append(aggregatedModelValidationResult.getModelDescription());
//							switch (classificationStatistics.getClassifierType()) {
//							case VCDRSA_RULES_CLASSIFIER:
//								infoBuilder.append("; ").append(String.format(Locale.US, "%s: %.2f", ModeRuleClassifier.avgNumberOfCoveringRulesIndicator, classificationStatistics.getAverageNumberOfCoveringRules()));
//								if (!qualitiesOfApproximation.equals("")) {
//									infoBuilder.append("; ").append(qualitiesOfApproximation);
//								}
//								infoBuilder.append(".");
//								break;
//							case WEKA_CLASSIFIER:
//								if (!qualitiesOfApproximation.equals("")) {
//									infoBuilder.append(" ").append(qualitiesOfApproximation).append(".");
//								}
//								break;
//							default:
//								throw new InvalidValueException("Incorrect classifier type.");
//							}
//							info = infoBuilder.toString();
//							//+++++
							
							String summaryLinePrefix = "    %% ";
							String accuracyType = useMainModelAccuracy ? "main model" : "overall";
							
							//OUTPUT
							outN("  Best avg. "+accuracyType+" accuracy for ('%1', %2(%3)): "+System.lineSeparator()+
								 "    %4 (stdDev: %5) (overall: %6 (stdDev: %7) | avg: %8) # %9 (stdDev: %10) # %11 (stdDev: %12) (%13 (stdDev: %14) | %15 (stdDev: %16)). Avg. main model decisions ratio: %17. "+System.lineSeparator()+
								 "    %% [Learning]: %18"+System.lineSeparator()+
								 "%19"+System.lineSeparator()+
								 "    %% [Model]: %20."+System.lineSeparator()+
								 "    %% [Avg. fold calculation times]: training: %21, validation: %22",
									dataSetName, algorithmName, parametersList.get(selector.parametersNumber),
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()), //
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getDeviationOfAccuracy()), //
									round(meansAndStandardDeviations.getOverallAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getOverallAverageAccuracy().getStdDev()),
									round(aggregatedModelValidationResult.getClassificationStatistics().getAvgAccuracy()),
									round(meansAndStandardDeviations.getMainModelAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getMainModelAverageAccuracy().getStdDev()),
									round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getMean()),
									round(meansAndStandardDeviations.getDefaultModelAverageAccuracy().getStdDev()),
									round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getMean()), //
									round(meansAndStandardDeviations.getDefaultClassAverageAccuracy().getStdDev()), //
									round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getMean()), //
									round(meansAndStandardDeviations.getDefaultClassifierAverageAccuracy().getStdDev()), //
									round(aggregatedModelValidationResult.getClassificationStatistics().getMainModelDecisionsRatio()),
									aggregatedModelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())),
//									info,
									aggregatedModelValidationResult.getModelDescription().toShortString(),
									round(totalFoldCalculationTimes.getAverageTrainingTime()),
									round(totalFoldCalculationTimes.getAverageValidationTime())
								);
						} //for
						outN("--");
					} else {
						outN("--");
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
	
	/**
	 * Rounds a positive number and returns it as a string.
	 * 
	 * @param number number to round and convert to text
	 * @return rounded number as text
	 * 
	 * @see {@link #decimalFormat} and {@link #percentDecimalFormat}
	 */
	public static String round(double number) {
		if (number > 1.0) {
			return String.format(Locale.US, percentDecimalFormat, number);
		} else if (number == 1.0) {
			return "1.0";
		} else if (number == 0.0) {
			return "0.0";
		} else {
			return String.format(Locale.US, decimalFormat, number);
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata.json",
				"data/json-objects/bank-churn-4000-v8 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata.json",
				"data/csv/OLM/bank-churn-4000-v8-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.00_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.00_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.00_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.00_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_05_mv2(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.05-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_05_mv15(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.05-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.05_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_10_mv2(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.10-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_10_mv15(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.10-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.10_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_15_mv2(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.15-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_15_mv15(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.15-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.15_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_20_mv2(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.20-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_20_mv15(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.20-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.20_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_25_mv2(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv2.json",
				"data/csv/OLM/bank-churn-4000-v8_0.25-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv2.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
	private static DataProvider getDataProviderChurn4000v8_0_25_mv15(String dataSetName, long[] seeds, int k) {
		switch (dataSetVersion) {
		case NORMAL:
			return new BasicDataProvider(
				"data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataSetName, seeds, k);
		case OLM_OSDL:
			return new BasicDataProvider(
				"data/json-metadata/OLM/bank-churn-4000-v8-processed metadata_mv1.5.json",
				"data/csv/OLM/bank-churn-4000-v8_0.25-processed data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_2X_GAIN:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-2xgain metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-2xgain data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		case MONGEL_NUM_OF_PRODUCTS_NONE_ENUMERATION_AND_IS_ACTIVE_MEMBER_INTEGER:
			return new BasicDataProvider(
				"data/json-metadata/MoNGEL/bank-churn-4000-v8-processed_numOfProducts-none-enumeration_isActiveMember-integer metadata_mv1.5.json",
				"data/csv/MoNGEL/bank-churn-4000-v8_0.25_numOfProducts-none data.csv",
				true, ';',
				dataSetName, seeds, k);
		default:
			throw new InvalidValueException("Not supported data set version.");
		}
	}
	
}

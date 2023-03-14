/**
 * 
 */
package org.rulelearn.experiments;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
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
import org.rulelearn.data.Decision;
import org.rulelearn.data.DecisionDistribution;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.BalancingDataProcessor.BalancingStrategy;
import org.rulelearn.experiments.BatchExperimentResults.CVSelector;
import org.rulelearn.experiments.BatchExperimentResults.CalculationTimes;
import org.rulelearn.experiments.BatchExperimentResults.DataAlgorithmParametersSelector;
import org.rulelearn.experiments.BatchExperimentResults.FullDataModelValidationResult;
import org.rulelearn.experiments.BatchExperimentResults.FullDataResults;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;
import org.rulelearn.experiments.ModelValidationResult.MeansAndStandardDeviations;
import org.rulelearn.experiments.helpers.ResultsTable;
import org.rulelearn.experiments.setup.BatchExperimentSetup;
import org.rulelearn.experiments.setup.BatchExperimentSetupChurn10000v8OLM_OSDL;
import org.rulelearn.experiments.setup.BatchExperimentSetupChurn10000v8Original;
import org.rulelearn.experiments.setup.BatchExperimentSetupChurn4000v8MoNGEL;
import org.rulelearn.experiments.setup.BatchExperimentSetupChurn4000v8OLM_OSDL;
import org.rulelearn.experiments.setup.BatchExperimentSetupChurn4000v8Original;
import org.rulelearn.experiments.setup.BatchExperimentSetupMonumentsMoNGEL;
import org.rulelearn.experiments.setup.BatchExperimentSetupMonumentsOLM_OSDL;
import org.rulelearn.experiments.setup.BatchExperimentSetupMonumentsOriginal;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;
import org.rulelearn.types.IntegerField;
import org.rulelearn.types.UnknownSimpleField;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Batch repeated cross-validation experiment over multiple data sets, with pre-processing of learning data, and different parameterized learning methods.
 * Each data set used in the experiment need to be an ordinal classification problem (with order among decisions).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BatchExperiment {
	
	List<DataProvider> dataProviders;
	CrossValidationProvider crossValidationProvider;
	DataProcessorProvider trainDataPreprocessorProvider;
	List<LearningAlgorithm> learningAlgorithms;
	LearningAlgorithmDataParametersContainer parametersContainer;
	
	//<BEGIN EXPERIMENT CONFIG>
	//TODO: configure?
	static boolean useMainModelAccuracy = false; //true = use main model accuracy; false = use overall accuracy
	
	static final boolean doFullDataReclassification = true;
	static final boolean doCrossValidations = true; //true = perform CVs; false = skip CVs
	static final boolean checkConsistencyOfTestDataDecisions = true;
	static final boolean printTrainedClassifiers = true; //concerns WEKA and KEEL classifiers + full data reclassification
	
	static final String decimalFormat = "%.5f"; //tells number of decimal places
	static final String percentDecimalFormat = "%.3f"; //tells number of decimal places in percentages
	
	static final boolean foldsInParallel = true; //false => folds will be done sequentially (useful only to measure more accurately avg. calculation times)
	//<END EXPERIMENT CONFIG>
	
	/**
	 * Constructs this experiment.
	 * 
	 * @param dataProviders
	 * @param crossValidationProvider
	 * @param trainDataPreprocessorProvider
	 * @param learningAlgorithms
	 * @param parametersContainer
	 */
	public BatchExperiment(List<DataProvider> dataProviders, CrossValidationProvider crossValidationProvider, DataProcessorProvider trainDataPreprocessorProvider, List<LearningAlgorithm> learningAlgorithms,
			LearningAlgorithmDataParametersContainer parametersContainer) {
		this.dataProviders = dataProviders;
		this.crossValidationProvider = crossValidationProvider;
		this.trainDataPreprocessorProvider = trainDataPreprocessorProvider;
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
	
	static String getTruePositiveRates(OrdinalMisclassificationMatrix misclassificationMatrix) {
		Decision[] decisions = misclassificationMatrix.getOrderOfDecisions();
		StringBuilder truePositivesSB = new StringBuilder(64);
		int counter = 0; //tells how many decisions have been processed
		
		for (Decision decision : decisions) {
			truePositivesSB.append(((SimpleDecision)decision).getEvaluation()).append(": ").append(round(misclassificationMatrix.getTruePositiveRate(decision)))
				.append(" (stdDev: ").append(round(misclassificationMatrix.getDeviationOfTruePositiveRate(decision))).append(")");
			counter++;
			
			if (counter < decisions.length) {
				truePositivesSB.append(" # ");
			}
		}
		
		return truePositivesSB.toString();
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
		outN(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING BATCH EXPERIMENT RUN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
		outN(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training data preprocessor: " + trainDataPreprocessorProvider.toString());
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
					
					//TODO: choose another version of provide method for full data?
					DataProcessor fullDataPreprocessor = trainDataPreprocessorProvider.provide(fullData.getGroupName()); //get preprocessor used only for full data
					//DataProcessor fullDataPreprocessor = trainDataPreprocessorProvider.provide(); //get preprocessor used only for full data
					System.out.println("Using full train data preprocesssor: "+fullDataPreprocessor.toString());
					
					//calculate and process full data models for all (algorithm, parameters) pairs
					Data processedFullData = fullDataPreprocessor.process(fullData); //processedFullData will have the same informationTableTransformationTime only if AcceptingDataProcessor is used
					
					//----- PRINT NUMBER OF OBJECTS IN EACH CLASS
					DecisionDistribution decisionDistribution = new DecisionDistribution(processedFullData.getInformationTable());
					for (Decision decision : decisionDistribution.getDecisions()) {
						System.out.println("Class "+((SimpleDecision)decision).getEvaluation()+": "+decisionDistribution.getCount(decision)+" objects.");
					}
					outN("--");
//					InformationTableWriter informationTableWriter = new InformationTableWriter(true);
//					try (FileWriter fileWriter = new FileWriter("./meta.json")) {
//						informationTableWriter.writeAttributes(processedFullData.getInformationTable(), fileWriter);
//					}
//					catch (IOException exception) {
//						exception.printStackTrace();
//					}
//					try (FileWriter fileWriter = new FileWriter("./data.json")) {
//						informationTableWriter.writeObjects(processedFullData.getInformationTable(), fileWriter);
//					}
//					catch (IOException exception) {
//						exception.printStackTrace();
//					}
//					//----- PRINT SIGNATURE AND HASH -----
//					System.out.print("++ Signature of processed full data: ");
//					System.out.println(signature(processedFullData.getInformationTable()));
//					System.out.print("++ Hash of processed full data: ");
//					System.out.println(processedFullData.getInformationTable().getHash());
//					outN("--");
//					//-----
					
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
							if (algorithm.getName().equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName()) && trainDataPreprocessorProvider instanceof AcceptingDataProcessorProvider) { //processedFullData is the same as fullData
								fullDataTrainingTime += fullData.getInformationTableTransformationTime(); //add time of data transformation (done out of time measurement zone marked by /**/, when fullData were provided), as VC-DRSA rule model needs this transformation!
								model.getModelLearningStatistics().totalDataTransformationTime = fullData.getInformationTableTransformationTime(); //set proper information table transformation time
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
							outN("Train data result for '%1(%2)': "+System.lineSeparator()+
									"Accuracy: %3 (overall: %4, avg: %5) # %6 # %7 (%8|%9). Main model decisions ratio: %10."+System.lineSeparator()+
									"True positive rates: %11 # Gmean: %12."+System.lineSeparator()+
									"%% [Learning]: %13."+System.lineSeparator()+
									"%14"+System.lineSeparator()+
									"%% [Times]: training: %15 [ms], validation: %16 [ms].",
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
									getTruePositiveRates(modelValidationResult.getOrdinalMisclassificationMatrix()),
									round(modelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
									modelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> (new StringBuilder("%% ")).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())), //print validation summary in several lines
									fullDataTrainingTime,
									fullDataValidationTime);
							outN("  /");
							outN(" /");
							outN("/");
							String modelDescription = model.getModelDescription().toString();
							model.getModelDescription().compress(); //free some memory occupied by model description
							if (modelDescription.endsWith(System.lineSeparator())) {
								out("[Model]: "+modelDescription);
							} else {
								outN("[Model]: "+modelDescription);
							}
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
						String[] preprocesssorsLog = new String[crossValidationFolds.size()];
						
						foldsStream.forEach(fold -> {       //just for measuring time!
							String linePrefix = "      "+foldNumber2Spaces(fold.getIndex());
							String summaryLinePrefix = linePrefix + "%% ";
							//OUTPUT
							String messageTemplate = prepareText("%pEnd of fold %1, algorithm %2(%3).%n"
									+ "%p%% [Accuracy]: %4 (overall: %5, avg: %6) # %7 # %8 (%9|%10). Main model decisions ratio: %11.%n"
									+ "%p%% [TP rates]: %12 # Gmean: %13.%n"
									+ "%14%n"
									+ "%p%% [Duration]: %15 [ms].", linePrefix); //%p will be replaced by prefix, %n by new line
							
							//long t3;
							//t3 = b("    Starting calculations for fold "+fold.getIndex()+".");
							//t3 = b(null);
							b(null);
							
							DataProcessor foldTrainDataPreprocessor = trainDataPreprocessorProvider.provide(
									fold.getTrainData().getGroupName(),
									dataProvider.getSeeds()[streamCrossValidationNumber],
									fold.getIndex());//get preprocessor used only for fold training data, on subsequent data sets
							preprocesssorsLog[fold.getIndex()] = "  Fold "+fold.getIndex()+": used CV fold train data preprocesssor: "+foldTrainDataPreprocessor.toString();
							
							Data processedTrainData = foldTrainDataPreprocessor.process(fold.getTrainData()); //e.g.: over-sampling, under-sampling, bootstrapping
							
//							//----- PRINT SIGNATURE AND HASH -----
//							System.out.println("++ /Fold "+fold.getIndex()+"/ Signature of processed fold train data: "
//									+ signature(processedTrainData.getInformationTable())
//									+ " ++ Hash of processed fold train data: "
//									+ processedTrainData.getInformationTable().getHash());
//							outN("--");
//							//-----
							
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
											getTruePositiveRates(modelValidationResult.getOrdinalMisclassificationMatrix()),
											round(modelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
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
						for (String s: preprocesssorsLog) {
							System.out.println(s);
						}
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
								outN("  Avg. result over folds for algorithm '%1(%2)': "+System.lineSeparator()+
										"    Accuracy: %3 (overall: %4, avg: %5) # %6 # %7 (%8|%9). Avg. main model decisions ratio: %10."+System.lineSeparator()+
										"    True positive rates: %11 # Gmean: %12.",
										learningAlgorithms.get(learningAlgorithmNumber).getName(),
										parameters,
										round(aggregatedCVModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()),
										round(classificationStatistics.getOverallAccuracy()), //test if the same as above
										round(classificationStatistics.getAvgAccuracy()), //test if the same as above
										round(classificationStatistics.getMainModelAccuracy()),
										round(classificationStatistics.getDefaultModelAccuracy()),
										round(classificationStatistics.getDefaultClassAccuracy()),
										round(classificationStatistics.getDefaultClassifierAccuracy()),
										round(classificationStatistics.getMainModelDecisionsRatio()),
										getTruePositiveRates(aggregatedCVModelValidationResult.getOrdinalMisclassificationMatrix()),
										round(aggregatedCVModelValidationResult.getOrdinalMisclassificationMatrix().getGmean())
								);
							}
						}
						outN("  ----------");
						
						VCDomLEMModeRuleClassifierLearnerCache.getInstance().clear(); //release references to allow GC
						NumberOfConsistentObjectsCache.getInstance().clear(); //release references to allow GC
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
							
							String summaryLinePrefix = "  %% ";
							
							//OUTPUT
							outN("Avg. result over CVs for algorithm '%1(%2)': "+System.lineSeparator()+
									"  Accuracy: %3 (stdDev: %4) (overall: %5 (stdDev: %6) | avg: %7) # %8 (stdDev: %9) # %10 (stdDev: %11) (%12 (stdDev: %13) | %14 (stdDev: %15)). Avg. main model decisions ratio: %16. "+System.lineSeparator()+
									"  True positive rates: %17 # Gmean: %18."+System.lineSeparator()+
									"  %% [Learning]: %19"+System.lineSeparator()+
									"%20"+System.lineSeparator()+
									"  %% [Model]: %21."+System.lineSeparator()+
									"  %% [Avg. fold calculation times]: training: %22 [ms], validation: %23 [ms]",
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
									getTruePositiveRates(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix()),
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
									aggregatedModelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())),
									aggregatedModelValidationResult.getModelDescription().toShortString(),
									round(totalFoldCalculationTimes.getAverageTrainingTime()),
									round(totalFoldCalculationTimes.getAverageValidationTime())
								);
	
							MeanAndStandardDeviation averageAccuracy = useMainModelAccuracy ?
									meansAndStandardDeviations.getMainModelAverageAccuracy() :
									meansAndStandardDeviations.getOverallAverageAccuracy(); //TODO: generalize comparison to other quality measures!
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
								
								String summaryLinePrefix = "    %% ";
								String accuracyType = useMainModelAccuracy ? "main model" : "overall"; //TODO: generalize comparison to other quality measures!
								
								//OUTPUT
								outN("  Best avg. "+accuracyType+" result over cross-validations for algorithm '%1(%2)': "+System.lineSeparator()+
									 "    Accuracy: %3 (stdDev: %4) (overall: %5 (stdDev: %6) | avg: %7) # %8 (stdDev: %9) # %10 (stdDev: %11) (%12 (stdDev: %13) | %14 (stdDev: %15)). Avg. main model decisions ratio: %16. "+System.lineSeparator()+
									 "    True positive rates: %17 # Gmean: %18."+System.lineSeparator()+
									 "    %% [Learning]: %19"+System.lineSeparator()+
									 "%20"+System.lineSeparator()+
									 "    %% [Model]: %21."+System.lineSeparator()+
									 "    %% [Avg. fold calculation times]: training: %22 [ms], validation: %23 [ms]",
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
										getTruePositiveRates(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix()),
										round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
										aggregatedModelValidationResult.getModelLearningStatistics().toString(),
										Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
										.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
										.collect(Collectors.joining(System.lineSeparator())),
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
			NumberOfConsistentObjectsCache.getInstance().clear(); //release references to allow GC
		} //for dataProvider
		
		return results;
	}
	
	public static void main(String[] args) {
		//<BEGIN EXPERIMENT CONFIG>
		int k = 10; //number of folds
//		int k = 4; //number of folds
		
//		long[] SKIP_DATA = new long[]{};
		
		long[] monumentsSeeds = new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302L, 4914830721005112L};
//		long[] monumentsSeeds = new long[]{0L, 8897335920153900L, 5347765673520470L}; //only first 3 CVs
//		long[] monumentsSeeds = SKIP_DATA;
		
//		/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
		long[] churn4000v8Seeds = new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L};
//		long[] churn4000v8Seeds = new long[]{0L, 5488762120989881L, 4329629961476882L}; //only first 3 CVs
//		long[] churn4000v8Seeds = new long[]{0L, 5488762120989881L}; //only first 2 CVs
//		long[] churn4000v8Seeds = SKIP_DATA;
		
		long[] churn10000v8Seeds = churn4000v8Seeds;
		
		//TODO: configure which setup should be used in this batch experiment
		BatchExperimentSetup[] batchExperimentSetups = {
				new BatchExperimentSetupMonumentsOriginal(monumentsSeeds, k, new AcceptingDataProcessorProvider()),
				new BatchExperimentSetupMonumentsOLM_OSDL(monumentsSeeds, k, new AcceptingDataProcessorProvider()),
				new BatchExperimentSetupMonumentsMoNGEL(monumentsSeeds, k, new AcceptingDataProcessorProvider()),
				
//				new BatchExperimentSetupMonumentsOriginalWithID(monumentsSeeds, k, new AcceptingDataProcessorProvider()), //setup just for testing purposes
				
				//setups just for testing purposes:
//				new BatchExperimentSetupMonumentsOriginal(monumentsSeeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDERSAMPLING, 9240360408272270L)),
//				new BatchExperimentSetupMonumentsOriginal(monumentsSeeds, k, new BalancingDataProcessorProvider(BalancingStrategy.OVERSAMPLING, 3637508937195708L)),
//				new BatchExperimentSetupMonumentsOriginal(monumentsSeeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDER_AND_OVERSAMPLING, 7449350427617649L)),
				
//				new BatchExperimentSetupMonumentsOriginalWithID(monumentsSeeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDERSAMPLING, 9240360408272270L)), //setup just for testing purposes
				
				new BatchExperimentSetupChurn4000v8Original(churn4000v8Seeds, k, new AcceptingDataProcessorProvider()),
				new BatchExperimentSetupChurn4000v8OLM_OSDL(churn4000v8Seeds, k, new AcceptingDataProcessorProvider()),
				new BatchExperimentSetupChurn4000v8MoNGEL(churn4000v8Seeds, k, new AcceptingDataProcessorProvider()),
				
				new BatchExperimentSetupChurn10000v8Original(churn10000v8Seeds, k, new AcceptingDataProcessorProvider()),
				new BatchExperimentSetupChurn10000v8OLM_OSDL(churn10000v8Seeds, k, new AcceptingDataProcessorProvider()),
//				
				new BatchExperimentSetupChurn10000v8Original(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDERSAMPLING, 9240360408272270L)),
				new BatchExperimentSetupChurn10000v8OLM_OSDL(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDERSAMPLING, 9240360408272270L)),
//				new BatchExperimentSetupChurn10000v8Original(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.OVERSAMPLING, 3637508937195708L)),
//				new BatchExperimentSetupChurn10000v8OLM_OSDL(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.OVERSAMPLING, 3637508937195708L)),
//				new BatchExperimentSetupChurn10000v8Original(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDER_AND_OVERSAMPLING, 7449350427617649L)),
//				new BatchExperimentSetupChurn10000v8OLM_OSDL(churn10000v8Seeds, k, new BalancingDataProcessorProvider(BalancingStrategy.UNDER_AND_OVERSAMPLING, 7449350427617649L))
		};
		//<END EXPERIMENT CONFIG>
		
		for (BatchExperimentSetup batchExperimentSetup : batchExperimentSetups) {
			List<DataProvider> dataProviders = batchExperimentSetup.getDataProviders();
			List<LearningAlgorithm> learningAlgorithms = batchExperimentSetup.getLearningAlgorithms();
			LearningAlgorithmDataParametersContainer parametersContainer = batchExperimentSetup.getLearningAlgorithmDataParametersContainer();
			DataProcessorProvider dataProcessorProvider = batchExperimentSetup.getDataProcessorProvider();
			
			BatchExperimentResults results = (new BatchExperiment(
					dataProviders,
					new RepeatableCrossValidationProvider(),
					dataProcessorProvider,
					learningAlgorithms,
					parametersContainer)
			).run();
			
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
			
			//$$$$$
			ResultsTable<String> accuracies = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> tPRsAndGmean = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> fullDataModelCharacteristics = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			//---
			ResultsTable<String> avgAccuracies = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> stdDevs = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> avgTPRsAndGmean = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> avgTrainingTimes = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> avgValidationTimes = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> avgDataModelCharacteristics = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			ResultsTable<String> avgTestDataQualities = new ResultsTable<>(dataSetsNames.size(), algorithmsNames.size());
			
			if (doFullDataReclassification) {
				accuracies.setTopLeftCell("% missing");
				accuracies.setColumnHeaders(algorithmsNames);
				tPRsAndGmean.setTopLeftCell("% missing");
				tPRsAndGmean.setColumnHeaders(algorithmsNames);
				fullDataModelCharacteristics.setTopLeftCell("% missing");
				fullDataModelCharacteristics.setColumnHeaders(algorithmsNames);
			}
			if (doCrossValidations) { //there are going to be average results => initialize tables
				avgAccuracies.setTopLeftCell("% missing");
				avgAccuracies.setColumnHeaders(algorithmsNames);
				stdDevs.setTopLeftCell("% missing");
				stdDevs.setColumnHeaders(algorithmsNames);
				avgTPRsAndGmean.setTopLeftCell("% missing");
				avgTPRsAndGmean.setColumnHeaders(algorithmsNames);
				avgTrainingTimes.setTopLeftCell("% missing");
				avgTrainingTimes.setColumnHeaders(algorithmsNames);
				avgValidationTimes.setTopLeftCell("% missing");
				avgValidationTimes.setColumnHeaders(algorithmsNames);
				avgDataModelCharacteristics.setTopLeftCell("% missing");
				avgDataModelCharacteristics.setColumnHeaders(algorithmsNames);
				avgTestDataQualities.setTopLeftCell("% missing");
				avgTestDataQualities.setColumnHeaders(algorithmsNames);
			}
			//$$$$$
			
			for (String dataSetName : dataSetsNames) {
				if (doFullDataReclassification) {
					outN(results.reportFullDataResults(dataSetName));
					//$$$$$
					accuracies.newRow(dataSetName);
					tPRsAndGmean.newRow(dataSetName);
					fullDataModelCharacteristics.newRow(dataSetName);
					
					for (String algorithmName : algorithmsNames) {
						List<LearningAlgorithmDataParameters> parameters;
						String parametersTxt;
						if ((parameters = parametersContainer.getParameters(algorithmName, dataSetName)) != null) {
							//TODO: get results for the best parameters if CV is done, not for the first
							parametersTxt = parameters.get(0).toString(); //get results for the first parameters
						} else {
							parametersTxt = "null";
						}
						ModelValidationResult modelValidationResult = results.dataName2FullDataResults.get(dataSetName).algorithmNameWithParameters2Results.get(algorithmName+"("+parametersTxt+")").getModelValidationResult();
						OrdinalMisclassificationMatrix fullDataOrdinalMisclassificationMatrix = modelValidationResult.getOrdinalMisclassificationMatrix();
						
						accuracies.addRowValue(round(fullDataOrdinalMisclassificationMatrix.getAccuracy()));
						tPRsAndGmean.addRowValue(String.format(Locale.US, "%s # Gmean: %s.", getTruePositiveRates(fullDataOrdinalMisclassificationMatrix), round(fullDataOrdinalMisclassificationMatrix.getGmean())));
						
						String rowValue = modelValidationResult.getModelDescription().toCompressedShortString();
						if (algorithmName.equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())) {
							rowValue += ", r/o: " + round(modelValidationResult.getClassificationStatistics().getAverageNumberOfCoveringRules());
						}
						fullDataModelCharacteristics.addRowValue(rowValue);
						
					}
					//$$$$$
				} //if (doFullDataReclassification)
				
				if (doCrossValidations) {
					//$$$$$
					avgAccuracies.newRow(dataSetName);
					stdDevs.newRow(dataSetName);
					avgTPRsAndGmean.newRow(dataSetName);
					avgTrainingTimes.newRow(dataSetName);
					avgValidationTimes.newRow(dataSetName);
					avgDataModelCharacteristics.newRow(dataSetName);
					avgTestDataQualities.newRow(dataSetName);
					//$$$$$
					
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
							
							String summaryLinePrefix = "  %% ";
							
							//OUTPUT
							outN("Avg. result for ('%1', %2(%3)): "+System.lineSeparator()+
									"  Accuracy: %4 (stdDev: %5) (overall: %6 (stdDev: %7) | avg: %8) # %9 (stdDev: %10) # %11 (stdDev: %12) (%13 (stdDev: %14) | %15 (stdDev: %16)). Avg. main model decisions ratio: %17. "+System.lineSeparator()+
									"  True positive rates: %18 # Gmean: %19."+System.lineSeparator()+
									"  %% [Learning]: %20"+System.lineSeparator()+
									"%21"+System.lineSeparator()+
									"  %% [Model]: %22."+System.lineSeparator()+
									"  %% [Avg. fold calculation times]: training: %23, validation: %24",
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
									getTruePositiveRates(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix()),
									round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
									aggregatedModelValidationResult.getModelLearningStatistics().toString(),
									Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
									.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
									.collect(Collectors.joining(System.lineSeparator())),
									aggregatedModelValidationResult.getModelDescription().toShortString(),
									round(totalFoldCalculationTimes.getAverageTrainingTime()),
									round(totalFoldCalculationTimes.getAverageValidationTime())
								);
		
							MeanAndStandardDeviation averageAccuracy = useMainModelAccuracy ?
									meansAndStandardDeviations.getMainModelAverageAccuracy() :
									meansAndStandardDeviations.getOverallAverageAccuracy(); //TODO: generalize comparison to other quality measures!
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
								
								String summaryLinePrefix = "    %% ";
								String accuracyType = useMainModelAccuracy ? "main model" : "overall"; //TODO: generalize comparison to other quality measures!
								
								//OUTPUT
								outN("  Best avg. "+accuracyType+" result for ('%1', %2(%3)): "+System.lineSeparator()+
									 "    Accuracy: %4 (stdDev: %5) (overall: %6 (stdDev: %7) | avg: %8) # %9 (stdDev: %10) # %11 (stdDev: %12) (%13 (stdDev: %14) | %15 (stdDev: %16)). Avg. main model decisions ratio: %17. "+System.lineSeparator()+
									 "    True positive rates: %18 # Gmean: %19."+System.lineSeparator()+
									 "    %% [Learning]: %20"+System.lineSeparator()+
									 "%21"+System.lineSeparator()+
									 "    %% [Model]: %22."+System.lineSeparator()+
									 "    %% [Avg. fold calculation times]: training: %23, validation: %24",
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
										getTruePositiveRates(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix()),
										round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getGmean()),
										aggregatedModelValidationResult.getModelLearningStatistics().toString(),
										Arrays.asList(classificationStatistics.toString().split(System.lineSeparator())).stream()
										.map(line -> new StringBuilder(128).append(summaryLinePrefix).append(line).toString())
										.collect(Collectors.joining(System.lineSeparator())), //[Testing]
										aggregatedModelValidationResult.getModelDescription().toShortString(), //[Model]
										round(totalFoldCalculationTimes.getAverageTrainingTime()),
										round(totalFoldCalculationTimes.getAverageValidationTime())
									);
							} //for
							outN("--");
						} else {
							outN("--");
						}
						
						//$$$$$
						{ //update results tables concerning avg. results
							DataAlgorithmParametersSelector selector = bestAlgorithmParametersSelectors.get(0); //get first selector concerning eq-equo best parameters //TODO: generalize?
							ModelValidationResult aggregatedModelValidationResult = results.getAggregatedModelValidationResult(selector);
							CalculationTimes totalFoldCalculationTimes = results.getTotalFoldCalculationTimes(selector);
							
							if (doCrossValidations) {
								avgAccuracies.addRowValue(round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getAccuracy()));
								stdDevs.addRowValue(round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getDeviationOfAccuracy()));
								avgTPRsAndGmean.addRowValue(String.format(Locale.US, "%s # Gmean: %s",
										getTruePositiveRates(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix()),
										round(aggregatedModelValidationResult.getOrdinalMisclassificationMatrix().getGmean()) ));
								avgTrainingTimes.addRowValue(round(totalFoldCalculationTimes.getAverageTrainingTime()));
								avgValidationTimes.addRowValue(round(totalFoldCalculationTimes.getAverageValidationTime()));
								//calculate row value:
								String rowValue = aggregatedModelValidationResult.getModelDescription().toCompressedShortString();
								if (algorithmName.equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())) {
									rowValue += ", r/o: " + round(aggregatedModelValidationResult.getClassificationStatistics().getAverageNumberOfCoveringRules());
								}
								avgDataModelCharacteristics.addRowValue(rowValue);
								//
								avgTestDataQualities.addRowValue(aggregatedModelValidationResult.getClassificationStatistics().getCompressedQualitiesOfApproximation());
							}
						}
						//$$$$$
					} //for algorithmName
				} //if (doCrossValidations)
				outN("####################");
			} //for dataSetName
			
			outN(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINISHING BATCH EXPERIMENT RUN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
			outN(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training data preprocessor: " + batchExperimentSetup.getDataProcessorProvider());

			//$$$$$
			if (doFullDataReclassification) {
				outN("Full data accuracy");
				outN(accuracies.toString("\t"));
				outN("--");
				outN("Full data TPR & Gmean");
				outN(tPRsAndGmean.toString("\t"));
				outN("--");
				outN("Full data model characteristics");
				outN(fullDataModelCharacteristics.toString("\t"));
				outN("--");
			}
			if (doCrossValidations) {
				outN("Avg accuracy");
				outN(avgAccuracies.toString("\t"));
				outN("--");
				outN("Standard deviations");
				outN(stdDevs.toString("\t"));
				outN("--");
				outN("Avg TPR & Gmean");
				outN(avgTPRsAndGmean.toString("\t"));
				outN("--");
				outN("Avg. fold training time [ms]");
				outN(avgTrainingTimes.toString("\t"));
				outN("--");
				outN("Avg. fold validation time [ms]");
				outN(avgValidationTimes.toString("\t"));
				outN("--");
				outN("Avg. data model characteristics (across cross-validation folds)");
				outN(avgDataModelCharacteristics.toString("\t"));
				outN("--");
				outN("Avg. quality for test data");
				outN(avgTestDataQualities.toString("\t"));
				outN("--");
			}
			//$$$$$
			
		} //for batchExperimentSetup
		
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
	
	/**
	 * Gets signature of given information table taking into account IDs of subsequent objects in that information table.
	 * Assumes that first column is of type {@link IntegerField}.
	 * 
	 * @param informationTable information table to be summarized using a short signature
	 * @return signature of given information table
	 */
	private static String signature(InformationTable informationTable) {
		if (informationTable == null) {
			return null;
		}
		
		//Index2IdMapper mapper = informationTable.getIndex2IdMapper();
		
		MessageDigest md = null;
		byte[] allBytes = new byte[4 * informationTable.getNumberOfObjects()];
		
		int index = 0;
		int value;
		for (int i = 0; i < informationTable.getNumberOfObjects(); i++) {
			if (!(informationTable.getField(i, 0) instanceof UnknownSimpleField)) {
				value = ((IntegerField)informationTable.getField(i, 0)).getValue();
			} else {
				value = i;
			}
			byte[] bytes = intToBytes(value);
			System.arraycopy(bytes, 0, allBytes, index, bytes.length);
			index += 4;
		}

		try {
			md = MessageDigest.getInstance("SHA-256");
		} catch (NoSuchAlgorithmException e) {
			return null;
		}
		
		md.update(allBytes);
		byte[] hashBytes = md.digest();
		
		StringBuilder builder = new StringBuilder(hashBytes.length * 2);
		for (byte b : hashBytes) {
			builder.append(String.format("%02X", b));
		}
		
		return builder.toString();
	}
	
	/**
	 * Converts given index to array of bytes.
	 * 
	 * @param index index to be converted
	 * @return array of bytes encoding given index
	 */
	private static byte[] intToBytes(final int index) {
	    return new byte[] {
	        (byte)((index >> 24) & 0xff),
	        (byte)((index >> 16) & 0xff),
	        (byte)((index >> 8) & 0xff),
	        (byte)((index >> 0) & 0xff),
	    };
	}
	
}

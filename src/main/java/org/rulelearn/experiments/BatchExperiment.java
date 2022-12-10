/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.experiments.BatchExperimentResults.AverageEvaluation;
import org.rulelearn.experiments.BatchExperimentResults.CVSelector;
import org.rulelearn.experiments.BatchExperimentResults.DataAlgorithmParametersSelector;
import org.rulelearn.experiments.BatchExperimentResults.FullDataResults;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
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
	
	public BatchExperimentResults run() {
		
		//calculate maximum number of cross validations among all data sets
		int maxCrossValidationsCount = -1;
		for (DataProvider dataProvider : dataProviders) {
			if (dataProvider.getSeeds().length > maxCrossValidationsCount) {
				maxCrossValidationsCount = dataProvider.getSeeds().length;
			}
		}
		outN("Maximum number of cross-validations: %1.", maxCrossValidationsCount); //!
		
		//calculate maximum number of parameters for an algorithm
		int maxParametersCount = -1;
		int parametersCount;
		for (DataProvider dataProvider : dataProviders) {
			for (LearningAlgorithm algorithm : learningAlgorithms) {
				Optional<List<LearningAlgorithmDataParameters>> optional = Optional.ofNullable(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
				parametersCount = optional.map(list -> list.size()).orElse(0);
				if (parametersCount > maxParametersCount) {
					maxParametersCount = parametersCount;
				}
			}
		}
		outN("Maximum number of algorithm vs data parameters, over all (data, algorithm) pairs: %1.", maxParametersCount); //!
		outN(); //!
		
		BatchExperimentResults batchExperimentResults = (new BatchExperimentResults.Builder())
				.dataSetsCount(dataProviders.size()).learningAlgorithmsCount(learningAlgorithms.size()).maxParametersCount(maxParametersCount).maxCrossValidationsCount(maxCrossValidationsCount).build();
		
		List<CrossValidationFold> crossValidationFolds;
		Data data;
		int dataSetNumber;
		int crossValidationsCount;
		CrossValidation crossValidation;
		BatchExperimentResults.CVSelector initializingCVSelector;
		
		long t1, t2;
		boolean calculationsForProvider;
		List<LearningAlgorithmDataParameters> parametersList;
		
		dataSetNumber = -1;
		for (DataProvider dataProvider : dataProviders) {
			dataSetNumber++;
			t1 = b(resolveText("Starting calculations for data %1.", dataProvider.getDataName()));
			final int streamDataSetNumber = dataSetNumber;
			
			calculationsForProvider = dataProvider.getSeeds().length > 0;
			
			if (calculationsForProvider) {
				double epsilonDRSAConsistencyThreshold = 0.0;
				double qualityOfDRSAApproximation;
				double qualityOfVCDRSAApproximation;
				Map<String, Double> algorithmNameWithParameters2Accuracy = new LinkedHashMap<String, Double>();
				Map<Double, Double> consistencyThreshold2QualityOfApproximation = new LinkedHashMap<Double, Double>();
				
				//print full data set accuracies
				Data fullData = dataProvider.provideOriginalData();
				outN("Quality of approximation for consistency threshold=%1: %2.", epsilonDRSAConsistencyThreshold, qualityOfDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), 0.0));
				consistencyThreshold2QualityOfApproximation.put(Double.valueOf(epsilonDRSAConsistencyThreshold), qualityOfDRSAApproximation);
				
				parametersList = parametersContainer.getParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataProvider.getDataName()); //get list of parameters for rule classifier
				if (parametersList != null) { //rule classifier is there (i.e., has at least one parameters) :)
					for (LearningAlgorithmDataParameters parameters : parametersList) { //check quality of approximation for all considered consistency thresholds
						double consistencyThreshold = Double.valueOf(parameters.getParameter(VCDomLEMRulesModeClassifierDataParameters.consistencyThresholdParameterName));
						if (!consistencyThreshold2QualityOfApproximation.containsKey(Double.valueOf(consistencyThreshold))) { //ensure that quality of approximation is calculated for each consistency threshold only once
							outN("Quality of approximation for consistency threshold=%1: %2.", consistencyThreshold, qualityOfVCDRSAApproximation = calculateQualityOfApproximation(fullData.getInformationTable(), consistencyThreshold));
							consistencyThreshold2QualityOfApproximation.put(consistencyThreshold, qualityOfVCDRSAApproximation);
						}
					}
				}
				
				//print full data models for all algorithm/parameters pairs
				Data processedFullData = trainDataPreprocessor.process(fullData);
				for (LearningAlgorithm algorithm : learningAlgorithms) {
					parametersList = processListOfParameters(parametersContainer.getParameters(algorithm.getName(), dataProvider.getDataName()));
					for (LearningAlgorithmDataParameters parameters : parametersList) { //check all parameters from the list of parameters for current algorithm
						ClassificationModel model = algorithm.learn(processedFullData, parameters); //can change result of processedFullData.getInformationTable()
						OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = model.validate(fullData);
						algorithmNameWithParameters2Accuracy.put(algorithm.getName()+"("+parameters+")", ordinalMisclassificationMatrix.getAccuracy());
						outN("Train data accuracy for parameterized algorithm '%1(%2)': %3.",
								algorithm.getName(),
								parameters,
								ordinalMisclassificationMatrix.getAccuracy());
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
				
				FullDataResults fullDataResults = new FullDataResults(consistencyThreshold2QualityOfApproximation, algorithmNameWithParameters2Accuracy);
				batchExperimentResults.storeFullDataResults(dataProvider.getDataName(), fullDataResults); //save quality of approximation and results of all parameterized algorithms for full data
			}
			
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
							ClassificationModel model = algorithm.learn(processedTrainData, parameters); //can change result of processedTrainData.getInformationTable()
							OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = model.validate(fold.getTestData());
							model = null; //facilitate GC
							
							BatchExperimentResults.CVSelector cvSelector = (new BatchExperimentResults.CVSelector())
									.dataSetNumber(streamDataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber).crossValidationNumber(streamCrossValidationNumber);
							batchExperimentResults.storeFoldMisclassificationMatrix(cvSelector, fold.getIndex(), ordinalMisclassificationMatrix);
							
							e(t5, resolveText("      %1Finishing calculations for fold %2, algorithm %3(%4).", foldNumber2Spaces(fold.getIndex()), fold.getIndex(), algorithm.getName(), parameters));
						}
						
						//e(t4, resolveText("      %1Finishing calculations for fold %2, algorithm %3.", foldNumber2Spaces(fold.getIndex()), fold.getIndex(), algorithm.getName()));
					}
					
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
						
						outN("  Avg. accuracy over folds for algorithm '%1(%2)': %3.",
								learningAlgorithms.get(learningAlgorithmNumber).getName(),
								parameters,
								batchExperimentResults.getAverageCVMisclassificationMatrix(cvSelector).getAccuracy());
					}
				}
				outN("  ----------");
			} //for crossValidationNumber
			
			e(t1, resolveText("Finishing calculations for data '%1'.", dataProvider.getDataName()));
//			e(t1, "Finishing calculations for data '"+dataProvider.getDataName()+"'.");
			
			if (calculationsForProvider) {
				outN("==========");
				for (int learningAlgorithmNumber = 0; learningAlgorithmNumber < learningAlgorithms.size(); learningAlgorithmNumber++) {
					parametersList = processListOfParameters(parametersContainer.getParameters(learningAlgorithms.get(learningAlgorithmNumber).getName(), dataProvider.getDataName()));
					int parametersNumber = -1;
					for (LearningAlgorithmDataParameters parameters : parametersList) {
						parametersNumber++;
						DataAlgorithmParametersSelector selector = (new DataAlgorithmParametersSelector())
								.dataSetNumber(dataSetNumber).learningAlgorithmNumber(learningAlgorithmNumber).parametersNumber(parametersNumber);
						AverageEvaluation averageEvaluation = batchExperimentResults.getAverageDataAlgorithmParametersAccuracy(selector);
						outN("Avg. accuracy over cross-validations for algorithm '%1(%2)': %3 (stdDev: %4).",
								learningAlgorithms.get(learningAlgorithmNumber).getName(),
								parameters,
								averageEvaluation.getAverage(),
								averageEvaluation.getStdDev());
					}
				}
				outN("==========");
			}
			outN();
			
			dataProvider.done(); //facilitate GC
		} //for dataProvider
		
		return batchExperimentResults;
	}
	
	public static void main(String[] args) {
		int k = 10; //number of folds
//		final String dataNameMonumentsNoMV = "zabytki";
		//-----
//		final String dataNameMonumentsNoMV01 = "zabytki01";
//		final String dataNameMonumentsNoMV01_K9_K10 = "zabytki01-K9-K10";
		//-----
		String dataNameChurn4000v8 = "bank-churn-4000-v8";
//		String dataNameChurn4000v8Copy1 = "bank-churn-4000-v8Copy1";
//		String dataNameChurn4000v8Copy2 = "bank-churn-4000-v8Copy2";
//		String dataNameChurn4000v8Copy3 = "bank-churn-4000-v8Copy3";
//		String dataNameChurn4000v8Copy4 = "bank-churn-4000-v8Copy4";
//		String dataNameChurn4000v8Copy5 = "bank-churn-4000-v8Copy5";
//		String dataNameChurn4000v8Copy6 = "bank-churn-4000-v8Copy6";
//		String dataNameChurn4000v8Copy7 = "bank-churn-4000-v8Copy7";
//		String dataNameChurn4000v8Copy8 = "bank-churn-4000-v8Copy8";
//		String dataNameChurn4000v8Copy9 = "bank-churn-4000-v8Copy9";
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
		
		List<DataProvider> dataProviders = new ArrayList<DataProvider>();
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
//				"src/main/resources/data/csv/zabytki-data-noMV.csv",
//				false, ';',
//				dataNameMonumentsNoMV,
//				//new long[]{},
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal.json",
//				"src/main/resources/data/csv/zabytki-data-noMV-0-1.csv",
//				false, ';',
//				dataNameMonumentsNoMV01,
//				//new long[]{},
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/zabytki-metadata-Y1-K-numeric-ordinal-K9-K10.json",
//				"src/main/resources/data/csv/zabytki-data-noMV-0-1.csv",
//				false, ';',
//				dataNameMonumentsNoMV01_K9_K10,
//				//new long[]{},
//				//new long[]{0L, 8897335920153900L, 5347765673520470L},
//				new long[]{0L, 8897335920153900L, 5347765673520470L, 3684779165093844L, 5095550231390613L, 1503924106488124L, 5782954920893053L, 3231154532347289L, 9843288945267302l, 4914830721005112L},
//				k));
		//-----
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
				dataNameChurn4000v8,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy1,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy2,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy3,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy4,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy5,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy6,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy7,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy8,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
//		dataProviders.add(new BasicDataProvider(
//				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata.json",
//				"src/main/resources/data/json-objects/bank-churn-4000-v8 data.json",
//				dataNameChurn4000v8Copy9,
//				new long[]{},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L},
//				//new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
//				k));
		//-----
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataNameChurn4000v8_0_05_mv2,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.05 data.json",
				dataNameChurn4000v8_0_05_mv15,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataNameChurn4000v8_0_10_mv2,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.10 data.json",
				dataNameChurn4000v8_0_10_mv15,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataNameChurn4000v8_0_15_mv2,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.15 data.json",
				dataNameChurn4000v8_0_15_mv15,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataNameChurn4000v8_0_20_mv2,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.20 data.json",
				dataNameChurn4000v8_0_20_mv15,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv2.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataNameChurn4000v8_0_25_mv2,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		dataProviders.add(new BasicDataProvider(
				"src/main/resources/data/json-metadata/bank-churn-4000-v8 metadata_mv1.5.json",
				"src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.json",
				dataNameChurn4000v8_0_25_mv15,
				//new long[]{},
				//new long[]{0L, 5488762120989881L, 4329629961476882L},
				new long[]{0L, 5488762120989881L, 4329629961476882L, 9522694898378332L, 6380856248140969L, 6557502705862619L, 2859990958560648L, 3853558955285837L, 6493344966644321L, 8051004458813256L},
				k));
		
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new VCDomLEMRulesModeClassifier());
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new J48()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new NaiveBayes()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new SMO()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new RandomForest()));
		
		LearningAlgorithmDataParametersContainer parametersContainer = (new LearningAlgorithmDataParametersContainer())
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameMonumentsNoMV,
//						Arrays.asList(
//								new VCDomLEMRulesModeClassifierDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), "yes")
//						))
//						//new VCDRSARuleModelBuilderDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
//				//-----
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameMonumentsNoMV01,
//						//new VCDRSARuleModelBuilderDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
//						Arrays.asList(
//								new VCDomLEMRulesModeClassifierDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes")
//						))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameMonumentsNoMV01_K9_K10,
//						//new VCDRSARuleModelBuilderDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("confidence>0.5"), DefaultClassificationResultChoiceMethod.MODE))
//						Arrays.asList(
//								new VCDomLEMRulesModeClassifierDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.018, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.036, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.054, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.072, CompositeRuleCharacteristicsFilter.of("s>0"), "yes"),
//								new VCDomLEMRulesModeClassifierDataParameters(0.09, CompositeRuleCharacteristicsFilter.of("s>0"), "yes")
//						));
				//-----
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0025"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence > 0.5"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175 & confidence >= 0.8"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0")
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy1,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.0, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.005"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.6"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy2,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.001, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.001, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.001, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0075"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.6666666666"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy3,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0025, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.0025, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.75"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy4,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.8"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy5,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.9"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy6,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						//new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"))
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01 & confidence >= 0.95"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy7,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.015, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy8,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.0175, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"))
//				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8Copy9,
//						//new VCDomLEMRulesModeClassifierDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
//						//new VCDomLEMRulesModeClassifierDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
//						//new VCDRSARuleModelBuilderDataParameters(0.02, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
//						new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"))
				//-----
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_05_mv2,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_05_mv15,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_10_mv2,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_10_mv15,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_15_mv2,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_15_mv15,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_20_mv2,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_20_mv15,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_25_mv2,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				.putParameters(VCDomLEMRulesModeClassifier.getAlgorithmName(), dataNameChurn4000v8_0_25_mv15,
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0"), "0"))
						//new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"))
						//new VCDRSARuleModelBuilderDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), DefaultClassificationResultChoiceMethod.MODE));
						Arrays.asList(
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.01"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0125"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.005, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),

								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.015"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0175"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.0075, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.02"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.0225"), "0"),
								new VCDomLEMRulesModeClassifierDataParameters(0.01, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0"),
								
								new VCDomLEMRulesModeClassifierDataParameters(0.0125, CompositeRuleCharacteristicsFilter.of("s > 0 & coverage-factor >= 0.025"), "0")
						))
				//-----
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv2,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_05_mv15,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv2,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_10_mv15,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv2,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_15_mv15,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv2,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_20_mv15,
						Arrays.asList(new WEKAAlgorithmOptions("-D")))
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv2,
						Arrays.asList(new WEKAAlgorithmOptions("-D"))) //discretize numeric attributes
				.putParameters(WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class), dataNameChurn4000v8_0_25_mv15,
						Arrays.asList(new WEKAAlgorithmOptions("-D")));
		
		BatchExperimentResults results = (new BatchExperiment(dataProviders, new RepeatableCrossValidationProvider(), new AcceptingDataProcessor(), learningAlgorithms, parametersContainer)).run();
		
		Function<String, Integer> d2i = (dataName) -> {
			return dataProviders.stream().map(provider -> provider.getDataName()).collect(Collectors.toList()).indexOf(dataName);
		}; //maps data name to data index at the dataProviders list
		Function<String, Integer> a2i = (algorithmName) -> {
			return learningAlgorithms.stream().map(algorithm -> algorithm.getName()).collect(Collectors.toList()).indexOf(algorithmName);
		}; //maps algorithm name to algorithm index at the learningAlgorithms list
		
		//------------------------------------------------------------------------------------------------------------------------------
		
		//print experiment summary:
		outN("####################");
		String[] dataSetsNames = {
//				dataNameMonumentsNoMV,
//				dataNameMonumentsNoMV01, dataNameMonumentsNoMV01_K9_K10,
				dataNameChurn4000v8,
//				dataNameChurn4000v8Copy1,
//				dataNameChurn4000v8Copy2,
//				dataNameChurn4000v8Copy3,
//				dataNameChurn4000v8Copy4,
//				dataNameChurn4000v8Copy5,
//				dataNameChurn4000v8Copy6,
//				dataNameChurn4000v8Copy7,
//				dataNameChurn4000v8Copy8,
//				dataNameChurn4000v8Copy9,
				dataNameChurn4000v8_0_05_mv2, dataNameChurn4000v8_0_05_mv15,
				dataNameChurn4000v8_0_10_mv2, dataNameChurn4000v8_0_10_mv15,
				dataNameChurn4000v8_0_15_mv2, dataNameChurn4000v8_0_15_mv15,
				dataNameChurn4000v8_0_20_mv2, dataNameChurn4000v8_0_20_mv15,
				dataNameChurn4000v8_0_25_mv2, dataNameChurn4000v8_0_25_mv15
		};
		
		String[] algorithmsNames = {VCDomLEMRulesModeClassifier.getAlgorithmName(), WEKAClassifierLearner.getAlgorithmName(J48.class), WEKAClassifierLearner.getAlgorithmName(NaiveBayes.class),
				WEKAClassifierLearner.getAlgorithmName(SMO.class), WEKAClassifierLearner.getAlgorithmName(RandomForest.class)};
		//String[] algorithmsNames = {VCDomLEMRulesModeClassifier.getAlgorithmName()};
		
		List<LearningAlgorithmDataParameters> parametersList;
		int parametersNumber;
		
		for (String dataSetName : dataSetsNames) {
			outN(results.reportFullDataResults(dataSetName));
			for (String algorithmName : algorithmsNames) {
				parametersList = processListOfParameters(parametersContainer.getParameters(algorithmName, dataSetName));
				parametersNumber = -1;
				for (LearningAlgorithmDataParameters parameters : parametersList) { //check all parameters from the list of parameters for the current algorithm
					parametersNumber++;
					
					AverageEvaluation averageEvaluation = results.getAverageDataAlgorithmParametersAccuracy(
							new DataAlgorithmParametersSelector().dataSetNumber(d2i.apply(dataSetName)).learningAlgorithmNumber(a2i.apply(algorithmName)).parametersNumber(parametersNumber));
					outN("Avg. accuracy for ('%1', %2(%3)): %4 (stdDev: %5).", dataSetName, algorithmName, parameters,
							averageEvaluation.getAverage(),
							averageEvaluation.getStdDev());
				}
			}
			outN("####################");
		}
		
	}
	
}

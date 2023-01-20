/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Locale;

import org.rulelearn.classification.SimpleClassificationResult;
import org.rulelearn.classification.SimpleOptimizingCountingRuleClassifier;
import org.rulelearn.classification.SimpleOptimizingRuleClassifier;
import org.rulelearn.classification.SimpleOptimizingCountingRuleClassifier.ResolutionStrategy;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.ModelValidationResult.DefaultClassificationType;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;
import org.rulelearn.experiments.ModelValidationResult.ClassifierType;
import org.rulelearn.rules.RuleSetWithComputableCharacteristics;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

/**
 * Classifies test data using decision rules and {@link SimpleOptimizingRuleClassifier}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ModeRuleClassifier implements ClassificationModel {
	
	static final String avgNumberOfCoveringRulesIndicator = "avg. number of cov. rules";
	
	public static class ModelDescriptionBuilder extends ClassificationModel.ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array is not an instance of {@link ModelDescription[]}.
		 */
		@Override
		ModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) {
			ModelDescription[] modelDescriptions = new ModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (ModelDescription)genericModelDescription;
			}
			return new ModelDescription(aggregationMode, modelDescriptions);
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		long totalRulesCount = 0L; //sum of number of rules
		long sumRuleLength = 0L; //sum of lengths of rules
		long sumRuleSupport = 0L; //sum of supports of rules
		double sumRuleConfidence = 0.0; //sum of confidences of rules
		
		int aggregationCount = 0; //tells how many ModelDescription objects have been used to build this object
		AggregationMode aggregationMode = AggregationMode.NONE;
		
		//TODO: add more fields to address situation when aggregationMode == AggregationMode.MEAN_AND_DEVIATION
		
		public ModelDescription(long totalRulesCount, long sumRuleLength, long sumRuleSupport, double sumRuleConfidence) {
			this.totalRulesCount = totalRulesCount;
			this.sumRuleLength = sumRuleLength;
			this.sumRuleSupport = sumRuleSupport;
			this.sumRuleConfidence = sumRuleConfidence;
			
			aggregationCount = 1;
			aggregationMode = AggregationMode.NONE;
		}
		
		public ModelDescription(AggregationMode aggregationMode, ModelDescription... modelDescriptions) {
			if (aggregationMode == null || aggregationMode == AggregationMode.NONE) {
				throw new InvalidValueException("Incorrect aggregation mode.");
			}
			
			//calculate sums
			for (ModelDescription modelDescription : modelDescriptions) {
				totalRulesCount += modelDescription.totalRulesCount;
				sumRuleLength += modelDescription.sumRuleLength;
				sumRuleSupport += modelDescription.sumRuleSupport;
				sumRuleConfidence += modelDescription.sumRuleConfidence;
				
				aggregationCount += modelDescription.aggregationCount;
			}
			
			this.aggregationMode = aggregationMode;
			
			if (this.aggregationMode == AggregationMode.MEAN_AND_DEVIATION) {
				//TODO: calculate means and standard deviations
			}
		}
		
		@Override
		public String toString() { //TODO: if aggregationMode == AggregationMode.MEAN_AND_DEVIATION, then print also standard deviations calculated in constructor
			StringBuilder sb = new StringBuilder(100);
			
			if (aggregationCount == 1) {
				sb.append("number of rules: ").append(totalRulesCount);
			} else {
				//sb.append("avg. number of rules: ").append((double)totalRulesCount / aggregationCount);
				sb.append(String.format(Locale.US, "avg. number of rules: %.2f", aggregationCount > 0 ? (double)totalRulesCount / aggregationCount : 0.0));
			}
			
			sb.append(String.format(Locale.US, ", average length: %.2f", (double)sumRuleLength / totalRulesCount));
			sb.append(String.format(Locale.US, ", average support: %.2f", (double)sumRuleSupport / totalRulesCount));
			sb.append(String.format(Locale.US, ", average confidence: %.2f", (double)sumRuleConfidence / totalRulesCount));
			
			return sb.toString();
		}
		
		@Override
		public String toShortString() {
			return toString();
		}

		@Override
		public ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new ModelDescriptionBuilder();
		}
		
	}
	
	//******************** BEGIN class members ********************
	
	RuleSetWithComputableCharacteristics ruleSet;
	SimpleClassificationResult defaultClassificationResult;
	SimpleOptimizingCountingRuleClassifier simpleOptimizingCountingRuleClassifier;
	ClassificationModel defaultClassificationModel = null; //classification model (classifier) used when no rule matches classified object (if the model is != null)
	
	ModelDescription modelDescription = null;
	
	ModelLearningStatistics modelLearningStatistics;
	
	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult,
			ModelLearningStatistics modelLearningStatistics) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.modelLearningStatistics = modelLearningStatistics;
	}
	
	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult, ClassificationModel defaultClassificationModel,
			ModelLearningStatistics modelLearningStatistics) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.defaultClassificationModel = defaultClassificationModel;
		this.modelLearningStatistics = modelLearningStatistics;
	}
	
	private SimpleDecision[] blendDecisions(SimpleDecision[] to, SimpleDecision[] from) { //returns blended array - what is not in "to", will be taken from "from"
		SimpleDecision[] result = new SimpleDecision[to.length];
		
		if (defaultClassificationModel != null) {
			for (int i = 0; i < to.length; i++) {
				if (to[i] != null) {
					result[i] = to[i];
				} else {
					result[i] = from[i]; //take undefined decision from array "from"
				}
			}
		} else {
			result = from.clone();
		}
		
		return result;
	}
	
	/**
	 * Validates this classifier on test data with known decisions.
	 * 
	 * @throws UnsupportedOperationException if given test data do not contain decisions for subsequent objects
	 */
	@Override
	public ModelValidationResult validate(Data testData) {
		
		InformationTable testInformationTable = testData.getInformationTable();
		
		if (testInformationTable.getDecisions(true) == null) {
			throw new UnsupportedOperationException("Cannot validate data without decisions.");
		}

		int testDataSize = testInformationTable.getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testInformationTable.getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testInformationTable.getDecisions(true);
		SimpleDecision[] defaultClassAssignedDecisions = new SimpleDecision[testDataSize]; //will contain decisions assigned using default decision class
		SimpleDecision[] assignedDecisions = new SimpleDecision[testDataSize]; //will contain assigned decisions
		
		ResolutionStrategy resolutionStrategy;
		boolean strategySucceeded;
		ClassificationStatistics classificationStatistics = new ClassificationStatistics(
				defaultClassificationModel != null ? DefaultClassificationType.USING_DEFAULT_CLASSIFIER : DefaultClassificationType.USING_DEFAULT_CLASS,
				ClassifierType.VCDRSA_RULES_CLASSIFIER);
		long totalCoveringRulesCount = 0;
	
		for (int testObjectIndex = 0; testObjectIndex < testDataSize; testObjectIndex++) {
			IntList indicesOfCoveringRules = new IntArrayList();
			assignedDecisions[testObjectIndex] = simpleOptimizingCountingRuleClassifier.classify(testObjectIndex, testInformationTable, indicesOfCoveringRules).getSuggestedDecision();
			totalCoveringRulesCount += indicesOfCoveringRules.size();
			
			resolutionStrategy = simpleOptimizingCountingRuleClassifier.getLatestResolutionStrategy();
			strategySucceeded = assignedDecisions[testObjectIndex].equals(originalDecisions[testObjectIndex]);
			
			switch (resolutionStrategy) {
			case MODE:
				if (strategySucceeded) {
					classificationStatistics.resolvingConflictCorrectCount++;
				} else {
					classificationStatistics.resolvingConflictIncorrectCount++;
				}
				break;
			case DEFAULT:
				if (strategySucceeded) {
					classificationStatistics.defaultClassCorrectCount++;
				} else {
					classificationStatistics.defaultClassIncorrectCount++;
				}
				
				if (defaultClassificationModel != null) { //SUPPORT FOR DEFAULT MODEL (fired when no rule matches classified object)
					defaultClassAssignedDecisions[testObjectIndex] = assignedDecisions[testObjectIndex]; //remember decision assigned using default decision class
					assignedDecisions[testObjectIndex] = defaultClassificationModel.classify(testObjectIndex, testData); //override rule classifier's default decision with default model's decision
					strategySucceeded = assignedDecisions[testObjectIndex].equals(originalDecisions[testObjectIndex]);
					
					if (strategySucceeded) {
						classificationStatistics.defaultClassifierCorrectCount++;
					} else {
						classificationStatistics.defaultClassifierIncorrectCount++;
					}
				}
				break;
			default:
				if (strategySucceeded) {
					classificationStatistics.preciseCorrectCount++;
				} else {
					classificationStatistics.preciseIncorrectCount++;
				}
				break;
			}
		} //for
		
		classificationStatistics.totalNumberOfCoveringRules = totalCoveringRulesCount;
		classificationStatistics.totalNumberOfClassifiedObjects = testDataSize;
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		if (BatchExperiment.checkConsistencyOfTestDataDecisions) {
			long start = System.currentTimeMillis();
			
			//synchronizes defaultClassAssignedDecisions
			SimpleDecision[] blendedDecisions = blendDecisions(defaultClassAssignedDecisions, assignedDecisions);
			
			classificationStatistics.totalNumberOfPreConsistentTestObjects =
					ClassificationModel.getNumberOfConsistentObjects(testInformationTable, 0.0);
			
			classificationStatistics.totalNumberOfPostConsistentTestObjectsIfDecisionsAssignedByMainModelAndDefaultClass =
					ClassificationModel.getNumberOfConsistentObjects(testInformationTable, blendDecisions(defaultClassAssignedDecisions, assignedDecisions), 0.0);
			
			classificationStatistics.totalNumberOfPostConsistentTestObjectsIfDecisionsAssignedByMainAndDefaultModel =
					ClassificationModel.getNumberOfConsistentObjects(testInformationTable, assignedDecisions, 0.0);
			
			classificationStatistics.totalNumberOfPreAndPostConsistentTestObjectsIfDecisionsAssignedByMainModelAndDefaultClass =
					ClassificationModel.getNumberOfPreAndPostConsistentObjects(testInformationTable, blendedDecisions, 0.0);
			
			classificationStatistics.totalNumberOfPreAndPostConsistentTestObjectsIfDecisionsAssignedByMainAndDefaultModel =
					ClassificationModel.getNumberOfPreAndPostConsistentObjects(testInformationTable, assignedDecisions, 0.0);
			
			classificationStatistics.avgQualityOfClassification = (double)classificationStatistics.totalNumberOfPreConsistentTestObjects / classificationStatistics.totalNumberOfClassifiedObjects;
			
			classificationStatistics.avgAccuracy = classificationStatistics.getOverallAccuracy();
			
			classificationStatistics.totalStatisticsCountingTime = System.currentTimeMillis() - start;
		}
		
		return new ModelValidationResult(ordinalMisclassificationMatrix, classificationStatistics, modelLearningStatistics, getModelDescription());
	}
	
	@Override
	public ModelDescription getModelDescription() {
		if (modelDescription == null) {
			long size = ruleSet.size();
			long sumLength = 0L;
			long sumSupport = 0L;
			double sumConfidence = 0.0;
			
			for (int i = 0; i < size; i++) {
				sumLength += ruleSet.getRuleCharacteristics(i).getNumberOfConditions();
				sumSupport += ruleSet.getRuleCharacteristics(i).getSupport();
				sumConfidence += ruleSet.getRuleCharacteristics(i).getConfidence();
			}
			
			modelDescription = new ModelDescription(size, sumLength, sumSupport, sumConfidence);
		}
		
		return modelDescription;
	}

	@Override
	public SimpleDecision classify(int i, Data testData) {
		return simpleOptimizingCountingRuleClassifier.classify(i, testData.getInformationTable()).getSuggestedDecision();
	}

	@Override
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}

}

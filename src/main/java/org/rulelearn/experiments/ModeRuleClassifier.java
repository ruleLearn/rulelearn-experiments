/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Locale;

import org.rulelearn.classification.SimpleClassificationResult;
import org.rulelearn.classification.SimpleOptimizingCountingRuleClassifier;
import org.rulelearn.classification.SimpleOptimizingRuleClassifier;
import org.rulelearn.classification.SimpleOptimizingCountingRuleClassifier.ResolutionStrategy;
import org.rulelearn.data.Decision;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.SimpleDecision;
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
	
	static final String avgNumberOfRulesIndicator = "avg. number of cov. rules";
	
	public static class ValidationSummary extends ClassificationModel.ValidationSummary {
		double preciseClassificationPercentage; //w.r.t. the size of validation set
		double correctPreciseClassificationPercentage; //w.r.t. the size of validation set
		
		double modeClassificationPercentage; //w.r.t. the size of validation set
		double correctModeClassificationPercentage; //w.r.t. the size of validation set
		
		boolean defaultClassifierUsed;
		
		double defaultClassClassificationPercentage; //w.r.t. the size of validation set
		double correctDefaultClassClassificationPercentage; //w.r.t. the size of validation set
		
		double defaultClassifierClassificationPercentage; //w.r.t. the size of validation set
		double correctDefaultClassifierClassificationPercentage; //w.r.t. the size of validation set
		
		double accuracy;
		double accuracyWhenClassifiedByRules;
		
		double accuracyWhenClassifiedByRulesPrecise;
		double accuracyWhenClassifiedByRulesMode;
		
		double accuracyWhenClassifiedByDefaultClass;
		double accuracyWhenClassifiedByDefaultClassifier;
		
		double avgNumberOfCoveringRules;
		//TODO: define more statistics (like avg. confidence?)
		
		double originalDecisionsQualityOfApproximation; //not used if -1.0
		double assignedDefaultClassDecisionsQualityOfApproximation; //not used if -1.0
		double assignedDecisionsQualityOfApproximation; //not used if -1.0
		
		public ValidationSummary(double preciseClassificationPercentage, double correctPreciseClassificationPercentage,
				double modeClassificationPercentage, double correctModeClassificationPercentage,
				boolean defaultClassifierUsed,
				double defaultClassClassificationPercentage, double correctDefaultClassClassificationPercentage,
				double defaultClassifierClassificationPercentage, double correctDefaultClassifierClassificationPercentage,
				double accuracy, double accuracyWhenClassifiedByRules, double accuracyWhenClassifiedByRulesPrecise, double accuracyWhenClassifiedByRulesMode,
				double accuracyWhenClassifiedByDefaultClass, double accuracyWhenClassifiedByDefaultClassifier,
				double avgNumberOfCoveringRules,
				double originalDecisionsQualityOfApproximation, double assignedDefaultClassDecisionsQualityOfApproximation, double assignedDecisionsQualityOfApproximation) {
			this.preciseClassificationPercentage = preciseClassificationPercentage;
			this.correctPreciseClassificationPercentage = correctPreciseClassificationPercentage;
			this.modeClassificationPercentage = modeClassificationPercentage;
			this.correctModeClassificationPercentage = correctModeClassificationPercentage;
			this.defaultClassifierUsed = defaultClassifierUsed;
			this.defaultClassClassificationPercentage = defaultClassClassificationPercentage;
			this.correctDefaultClassClassificationPercentage = correctDefaultClassClassificationPercentage;
			this.defaultClassifierClassificationPercentage = defaultClassifierClassificationPercentage;
			this.correctDefaultClassifierClassificationPercentage = correctDefaultClassifierClassificationPercentage;
			this.accuracy = accuracy;
			this.accuracyWhenClassifiedByRules = accuracyWhenClassifiedByRules;
			this.accuracyWhenClassifiedByRulesPrecise = accuracyWhenClassifiedByRulesPrecise;
			this.accuracyWhenClassifiedByRulesMode = accuracyWhenClassifiedByRulesMode;
			this.accuracyWhenClassifiedByDefaultClass = accuracyWhenClassifiedByDefaultClass;
			this.accuracyWhenClassifiedByDefaultClassifier = accuracyWhenClassifiedByDefaultClassifier;
			this.avgNumberOfCoveringRules = avgNumberOfCoveringRules;
			this.originalDecisionsQualityOfApproximation = originalDecisionsQualityOfApproximation;
			this.assignedDefaultClassDecisionsQualityOfApproximation = assignedDefaultClassDecisionsQualityOfApproximation;
			this.assignedDecisionsQualityOfApproximation = assignedDecisionsQualityOfApproximation;
		}
		
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder(120);
			
			sb.append("[Summary]: ");
			sb.append(String.format(Locale.US, "precise: %.2f%% (%.2f%% hit)", preciseClassificationPercentage, correctPreciseClassificationPercentage));
			sb.append(String.format(Locale.US, ", mode: %.2f%% (%.2f%% hit)", modeClassificationPercentage, correctModeClassificationPercentage));
			sb.append(String.format(Locale.US, ", default class: %.2f%% (%.2f%% hit)", defaultClassClassificationPercentage, correctDefaultClassClassificationPercentage));
			sb.append(String.format(Locale.US, ", default classifier: %.2f%% (%.2f%% hit)", defaultClassifierClassificationPercentage, correctDefaultClassifierClassificationPercentage));
			sb.append(String.format(Locale.US, "; by rules: %.2f%% r.hit", accuracyWhenClassifiedByRules)); //accuracy among objects covered by 1+ rule
			sb.append(accuracyWhenClassifiedByRules > accuracy ? " [UP]" : " [!UP]");
			sb.append(String.format(Locale.US, " (precise: %.2f%% r.hit", accuracyWhenClassifiedByRulesPrecise)); //accuracy among objects covered by 1+ rule(s) of the same type (at least or at most)
			sb.append(String.format(Locale.US, ", mode: %.2f%% r.hit),", accuracyWhenClassifiedByRulesMode)); //accuracy among objects covered by 1+ rule(s) of different types (at least and at most)
			sb.append(String.format(Locale.US, "%n[Summary]: "));
			sb.append(String.format(Locale.US, "by default class: %.2f%% r.hit", accuracyWhenClassifiedByDefaultClass)); //accuracy among objects not covered by any rule
			sb.append(String.format(Locale.US, ", by default classifier: %.2f%% r.hit", accuracyWhenClassifiedByDefaultClassifier)); //accuracy among objects not covered by any rule
			sb.append(String.format(Locale.US, "; "+avgNumberOfRulesIndicator+": %.2f", avgNumberOfCoveringRules));
			if (originalDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, "; original quality: %.4f", originalDecisionsQualityOfApproximation));
			}
			if (assignedDefaultClassDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, "; assigned default class quality: %.4f", assignedDefaultClassDecisionsQualityOfApproximation));
			}
			if (assignedDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, ", assigned quality: %.4f", assignedDecisionsQualityOfApproximation));
			}
			sb.append(".");
			//TODO: show more statistics (like avg. confidence?)
			
			return sb.toString();
		}
	}
	
	public static class ModelDescriptionBuilder extends ClassificationModel.ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array is not an instance of {@link ModelDescription[]}.
		 */
		@Override
		ModelDescription build(ClassificationModel.ModelDescription... genericModelDescriptions) {
			ModelDescription[] modelDescriptions = new ModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (ModelDescription)genericModelDescription;
			}
			return new ModelDescription(modelDescriptions);
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		long totalRulesCount = 0L; //sumRulesCount
		long sumRuleLength = 0L; //sum of lengths of rules
		long sumRuleSupport = 0L;  //sum of supports of rules
		
		int aggregationCount = 0; //tells how many ModelDescription objects have been used to build this object
		
		public ModelDescription(long totalRulesCount, long sumRuleLength, long sumRuleSupport) {
			this.totalRulesCount = totalRulesCount;
			this.sumRuleLength = sumRuleLength;
			this.sumRuleSupport = sumRuleSupport;
			
			aggregationCount = 1;
		}
		
		public ModelDescription(ModelDescription... modelDescriptions) {
			for (ModelDescription modelDescription : modelDescriptions) {
				totalRulesCount += modelDescription.totalRulesCount;
				sumRuleLength += modelDescription.sumRuleLength;
				sumRuleSupport += modelDescription.sumRuleSupport;
				aggregationCount += modelDescription.aggregationCount;
			}
		}		
		
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder(100);
			if (aggregationCount == 1) {
				sb.append("Number of rules: ").append(totalRulesCount);
			} else {
				sb.append("Avg. number of rules: ").append((double)totalRulesCount / aggregationCount);
			}
			sb.append(String.format(Locale.US, ", average length: %.2f", (double)sumRuleLength / totalRulesCount));
			sb.append(String.format(Locale.US, ", average support: %.2f", (double)sumRuleSupport / totalRulesCount));
			return sb.toString();
		}

		@Override
		public ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new ModelDescriptionBuilder();
		}
		
	}
	
	RuleSetWithComputableCharacteristics ruleSet;
	SimpleClassificationResult defaultClassificationResult;
	SimpleOptimizingCountingRuleClassifier simpleOptimizingCountingRuleClassifier;
	ClassificationModel defaultClassificationModel = null; //classification model (classifier) used when no rule matches classified object (if the model is != null)
	
	ValidationSummary validationSummary = null;
	String modelLearnerDescription;
	ModelDescription modelDescription = null;
	
	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult, String modelLearnerDescription) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.modelLearnerDescription = modelLearnerDescription;
	}
	
	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult,
			ClassificationModel defaultClassificationModel, String modelLearnerDescription) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.defaultClassificationModel = defaultClassificationModel;
		this.modelLearnerDescription = modelLearnerDescription;
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
		
		final class ResolutionCounters {
			long preciseCorrect = 0;
			long preciseIncorrect = 0;
			
			long modeCorrect = 0;
			long modeIncorrect = 0;
			
			long defaultCorrect = 0;
			long defaultIncorrect = 0;
			
			long defaultClassCorrect = 0;
			long defaultClassIncorrect = 0;
			
			long defaultClassifierCorrect = 0;
			long defaultClassifierIncorrect = 0;
		}
		
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
		ResolutionCounters resolutionCounters = new ResolutionCounters();
		long totalCoveringRulesCount = 0;
	
		for (int testObjectIndex = 0; testObjectIndex < testDataSize; testObjectIndex++) {
			IntList indicesOfCoveringRules = new IntArrayList();
			assignedDecisions[testObjectIndex] = simpleOptimizingCountingRuleClassifier.classify(testObjectIndex, testInformationTable, indicesOfCoveringRules).getSuggestedDecision();
			totalCoveringRulesCount += indicesOfCoveringRules.size();
			
			resolutionStrategy = simpleOptimizingCountingRuleClassifier.getLatestResolutionStrategy();
			
//			//SUPPORT FOR DEFAULT MODEL (fired when no rule matches classified object)
//			if (resolutionStrategy == ResolutionStrategy.DEFAULT && defaultClassificationModel != null) {
//				assignedDecisions[testObjectIndex] = defaultClassificationModel.classify(testObjectIndex, testData); //override rule classifier's default decision with default model's decision
//			}
			
			strategySucceeded = assignedDecisions[testObjectIndex].equals(originalDecisions[testObjectIndex]);
			
			switch (resolutionStrategy) {
			case MODE:
				if (strategySucceeded) {
					resolutionCounters.modeCorrect++;
				} else {
					resolutionCounters.modeIncorrect++;
				}
				break;
			case DEFAULT:
				if (strategySucceeded) {
					resolutionCounters.defaultClassCorrect++;
				} else {
					resolutionCounters.defaultClassIncorrect++;
				}
				
				if (defaultClassificationModel != null) { //SUPPORT FOR DEFAULT MODEL (fired when no rule matches classified object)
					defaultClassAssignedDecisions[testObjectIndex] = assignedDecisions[testObjectIndex]; //remember decision assigned using default decision class
					assignedDecisions[testObjectIndex] = defaultClassificationModel.classify(testObjectIndex, testData); //override rule classifier's default decision with default model's decision
					strategySucceeded = assignedDecisions[testObjectIndex].equals(originalDecisions[testObjectIndex]);
					
					if (strategySucceeded) {
						resolutionCounters.defaultClassifierCorrect++;
					} else {
						resolutionCounters.defaultClassifierIncorrect++;
					}
				}
				break;
			default:
				if (strategySucceeded) {
					resolutionCounters.preciseCorrect++;
				} else {
					resolutionCounters.preciseIncorrect++;
				}
				break;
			}
		} //for
		
		resolutionCounters.defaultCorrect = (defaultClassificationModel != null) ? resolutionCounters.defaultClassifierCorrect : resolutionCounters.defaultClassCorrect;
		resolutionCounters.defaultIncorrect = (defaultClassificationModel != null) ? resolutionCounters.defaultClassifierIncorrect : resolutionCounters.defaultClassIncorrect;
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		double accuracyWhenClassifiedByRules = 100 * ( (double)(resolutionCounters.preciseCorrect + resolutionCounters.modeCorrect) /
				(double)((long)testDataSize - (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect)) ); //divide by the number of objects not-classified to the default class
		
//		double accuracyWhenClassifiedByDefault = 100 * ((double)resolutionCounters.defaultCorrect / (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect));
		
		double originalDecisionsQualityOfApproximation = -1.0;
		long originalDecisionsConsistentObjectsCount = -1L;
		
		double assignedDefaultClassDecisionsQualityOfApproximation = -1.0;
		long assignedDefaultClassDecisionsConsistentObjectsCount = -1L;
		
		double assignedDecisionsQualityOfApproximation = -1.0;
		long assignedDecisionsConsistentObjectsCount = -1L;
		
		if (BatchExperiment.checkConsistencyOfAssignedDecisions) {
			originalDecisionsQualityOfApproximation = getQualityOfApproximation(testInformationTable, 0.0);
			originalDecisionsConsistentObjectsCount = Math.round(originalDecisionsQualityOfApproximation * testDataSize); //go back to integer number
			
			//synchronizes defaultClassAssignedDecisions
			SimpleDecision[] blendedDecisions = blendDecisions(defaultClassAssignedDecisions, assignedDecisions);
			assignedDefaultClassDecisionsQualityOfApproximation = getQualityOfApproximationForDecisions(testInformationTable, blendedDecisions, 0.0);
			assignedDefaultClassDecisionsConsistentObjectsCount = Math.round(assignedDefaultClassDecisionsQualityOfApproximation * testDataSize); //go back to integer number
			
			assignedDecisionsQualityOfApproximation = getQualityOfApproximationForDecisions(testInformationTable, assignedDecisions, 0.0);
			assignedDecisionsConsistentObjectsCount = Math.round(assignedDecisionsQualityOfApproximation * testDataSize); //go back to integer number
		}
		
		this.validationSummary = new ValidationSummary(
				100 * ((double)(resolutionCounters.preciseCorrect + resolutionCounters.preciseIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.preciseCorrect / testDataSize),
				100 * ((double)(resolutionCounters.modeCorrect + resolutionCounters.modeIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.modeCorrect / testDataSize),
				defaultClassificationModel != null,
				100 * ((double)(resolutionCounters.defaultClassCorrect + resolutionCounters.defaultClassIncorrect) / testDataSize),
				100 * ((double)(resolutionCounters.defaultClassCorrect) / testDataSize),
				100 * ((double)(resolutionCounters.defaultClassifierCorrect + resolutionCounters.defaultClassifierIncorrect) / testDataSize),
				100 * ((double)(resolutionCounters.defaultClassifierCorrect) / testDataSize),
				ordinalMisclassificationMatrix.getAccuracy(),
				accuracyWhenClassifiedByRules,
				100 * ( (double)resolutionCounters.preciseCorrect /
						(resolutionCounters.preciseCorrect + resolutionCounters.preciseIncorrect)), //accuracy when classified by precise rules
				100 * ( (double)resolutionCounters.modeCorrect /
						(resolutionCounters.modeCorrect + resolutionCounters.modeIncorrect)),  //accuracy when classified by mode
				(resolutionCounters.defaultClassCorrect + resolutionCounters.defaultClassIncorrect) > 0 ?
						100 * ((double)resolutionCounters.defaultClassCorrect / (resolutionCounters.defaultClassCorrect + resolutionCounters.defaultClassIncorrect)) : 0.0,
				(resolutionCounters.defaultClassifierCorrect + resolutionCounters.defaultClassifierIncorrect) > 0 ?
						100 * ((double)resolutionCounters.defaultClassifierCorrect / (resolutionCounters.defaultClassifierCorrect + resolutionCounters.defaultClassifierIncorrect)) : 0.0,
				(double)totalCoveringRulesCount / testDataSize,
				originalDecisionsQualityOfApproximation, assignedDefaultClassDecisionsQualityOfApproximation, assignedDecisionsQualityOfApproximation);

		
		return new ModelValidationResult(ordinalMisclassificationMatrix,
				resolutionCounters.preciseCorrect + resolutionCounters.modeCorrect,
				(long)testDataSize - (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect),
				resolutionCounters.defaultCorrect,
				resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect,
				getModelDescription(),
				totalCoveringRulesCount,
				testDataSize); //possible abstaining taken into account!
	}
	
	public ValidationSummary getValidationSummary() { //gets summary of last validate method invocation
		return validationSummary;
	}

	@Override
	public ModelDescription getModelDescription() {
		if (modelDescription == null) {
			long size = ruleSet.size();
			long sumLength = 0L;
			long sumSupport = 0L;
			for (int i = 0; i < size; i++) {
				sumLength += ruleSet.getRuleCharacteristics(i).getNumberOfConditions();
				sumSupport += ruleSet.getRuleCharacteristics(i).getSupport();
			}
			
			modelDescription = new ModelDescription(size, sumLength, sumSupport);
		}
		
		return modelDescription;
	}

	@Override
	public SimpleDecision classify(int i, Data testData) {
		return simpleOptimizingCountingRuleClassifier.classify(i, testData.getInformationTable()).getSuggestedDecision();
	}

	@Override
	public String getModelLearnerDescription() {
		return modelLearnerDescription;
	}

}

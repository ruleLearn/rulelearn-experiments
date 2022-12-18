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
	
	RuleSetWithComputableCharacteristics ruleSet;
	SimpleClassificationResult defaultClassificationResult;
	SimpleOptimizingCountingRuleClassifier simpleOptimizingCountingRuleClassifier;
	ClassificationModel defaultClassificationModel = null; //classification model (classifier) used when no rule matches classified object (if the model is != null)
	
	String modelLearnerDescription;
	String validationSummary;

	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult, String modelLearnerDescription) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.modelLearnerDescription = modelLearnerDescription;
		validationSummary = "";
	}
	
	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult,
			ClassificationModel defaultClassificationModel, String modelLearnerDescription) {
		this.ruleSet = ruleSet;
		this.defaultClassificationResult = defaultClassificationResult;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		this.defaultClassificationModel = defaultClassificationModel;
		this.modelLearnerDescription = modelLearnerDescription;
		validationSummary = "";
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
		}
		
		InformationTable testInformationTable = testData.getInformationTable();
		
		if (testInformationTable.getDecisions(true) == null) {
			throw new UnsupportedOperationException("Cannot validate data without decisions.");
		}

		int testDataSize = testInformationTable.getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testInformationTable.getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testInformationTable.getDecisions(true);
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
			
			//SUPPORT FOR DEFAULT MODEL (fired when no rule matches classified object)
			if (resolutionStrategy == ResolutionStrategy.DEFAULT && defaultClassificationModel != null) {
				assignedDecisions[testObjectIndex] = defaultClassificationModel.classify(testObjectIndex, testData); //override rule classifier's default decision with default model's decision
			}
			
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
					resolutionCounters.defaultCorrect++;
				} else {
					resolutionCounters.defaultIncorrect++;
				}
				break;
			default:
				if (strategySucceeded) {
					resolutionCounters.preciseCorrect++;
				} else {
					resolutionCounters.preciseIncorrect++;
				}
			}
		}
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		double accuracyWhenClassifiedByRules = 100 * ( (double)(resolutionCounters.preciseCorrect + resolutionCounters.modeCorrect) /
				(double)((long)testDataSize - (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect)) ); //divide by the number of objects not-classified to the default class
		
		double accuracyWhenClassifiedByDefault = 100 * ((double)resolutionCounters.defaultCorrect / (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect));
		
		StringBuilder sb = new StringBuilder(120);
		sb.append("[Classification]: ");
		sb.append(String.format(Locale.US, "precise: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.preciseCorrect + resolutionCounters.preciseIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.preciseCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", mode: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.modeCorrect + resolutionCounters.modeIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.modeCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", default: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.defaultCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", using rules: %.2f%% r.hit", //accuracy among objects covered by 1+ rule
				accuracyWhenClassifiedByRules));
		sb.append(accuracyWhenClassifiedByRules > ordinalMisclassificationMatrix.getAccuracy() ? " [UP]" : " [!UP]");
		sb.append(String.format(Locale.US, ", using %s: %.2f%% r.hit", //accuracy among objects not covered by any rule
				defaultClassificationModel != null ? "default classifier" : "default class",
				accuracyWhenClassifiedByDefault));
		sb.append(String.format(Locale.US, ", avg. number of cov. rules: %.2f.",
				(double)totalCoveringRulesCount / testDataSize));
		
		validationSummary = sb.toString();
		
		return new ModelValidationResult(ordinalMisclassificationMatrix,
				resolutionCounters.preciseCorrect + resolutionCounters.modeCorrect,
				(long)testDataSize - (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect),
				resolutionCounters.defaultCorrect,
				resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect); //possible abstaining taken into account!
	}
	
	public String getValidationSummary() {
		return validationSummary;
	}

	@Override
	public String getModelDescription() {
		int size = ruleSet.size();
		StringBuilder sb = new StringBuilder(100);
		sb.append("number of rules: ").append(size);
		long sumLength = 0L;
		long sumSupport = 0L;
		for (int i = 0; i < size; i++) {
			sumLength += ruleSet.getRuleCharacteristics(i).getNumberOfConditions();
			sumSupport += ruleSet.getRuleCharacteristics(i).getSupport();
		}
		sb.append(", average length: ").append(((double)sumLength) / size);
		sb.append(", average support: ").append(((double)sumSupport) / size);
		if (defaultClassificationModel != null) {
			sb.append(", default model learned using: ").append(defaultClassificationModel.getModelLearnerDescription());
		} else {
			sb.append(", default class: ").append(((SimpleDecision)defaultClassificationResult.getSuggestedDecision()).getEvaluation());
		}
		return sb.toString();
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

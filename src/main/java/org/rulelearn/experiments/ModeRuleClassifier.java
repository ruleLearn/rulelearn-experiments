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
	SimpleOptimizingCountingRuleClassifier simpleOptimizingCountingRuleClassifier;
	String validationSummary;

	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult) {
		this.ruleSet = ruleSet;
		simpleOptimizingCountingRuleClassifier = new SimpleOptimizingCountingRuleClassifier(ruleSet, defaultClassificationResult);
		validationSummary = "";
	}

	/**
	 * Validates this classifier on test data with known decisions.
	 * 
	 * @throws UnsupportedOperationException if given test data do not contain decisions for subsequent objects
	 */
	@Override
	public OrdinalMisclassificationMatrix validate(Data testData) {
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
		
		OrdinalMisclassificationMatrix result = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		double accuracyWhenClassifiedByRules = 100 * ( (double)(resolutionCounters.preciseCorrect + resolutionCounters.modeCorrect) /
				(double)(testDataSize - (resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect)) );
		
		StringBuilder sb = new StringBuilder(120);
		sb.append("[Classification summary]: ");
		sb.append(String.format(Locale.US, "precise: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.preciseCorrect + resolutionCounters.preciseIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.preciseCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", mode: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.modeCorrect + resolutionCounters.modeIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.modeCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", default: %.2f%%(%.2f%% hit)",
				100 * ((double)(resolutionCounters.defaultCorrect + resolutionCounters.defaultIncorrect) / testDataSize),
				100 * ((double)resolutionCounters.defaultCorrect / testDataSize) ));
		sb.append(String.format(Locale.US, ", using rules: %.2f%% hit", //accuracy among objects covered by 1+ rule
				accuracyWhenClassifiedByRules)); //divide by the number of objects not-classified to the default class
		sb.append(accuracyWhenClassifiedByRules > result.getAccuracy() ? " [UP]" : " [!UP]");
		sb.append(String.format(Locale.US, ", avg. number of covering rules: %.2f.",
				(double)totalCoveringRulesCount / testDataSize));
		
		validationSummary = sb.toString();
		
		return result;
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
		sb.append(", ");
		sb.append("average length: ").append(((double)sumLength) / size);
		sb.append(", ");
		sb.append("average support: ").append(((double)sumSupport) / size);
		return sb.toString();
	}

}

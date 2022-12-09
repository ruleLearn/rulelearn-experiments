/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.classification.SimpleClassificationResult;
import org.rulelearn.classification.SimpleOptimizingRuleClassifier;
import org.rulelearn.data.Decision;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.rules.RuleSetWithComputableCharacteristics;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Classifies test data using decision rules and {@link SimpleOptimizingRuleClassifier}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ModeRuleClassifier implements ClassificationModel {
	
	RuleSetWithComputableCharacteristics ruleSet;
	SimpleOptimizingRuleClassifier simpleOptimizingRuleClassifier;

	public ModeRuleClassifier(RuleSetWithComputableCharacteristics ruleSet, SimpleClassificationResult defaultClassificationResult) {
		this.ruleSet = ruleSet;
		simpleOptimizingRuleClassifier = new SimpleOptimizingRuleClassifier(ruleSet, defaultClassificationResult);
	}

	/**
	 * Validates this classifier on test data with known decisions.
	 * 
	 * @throws UnsupportedOperationException if given test data do not contain decisions for subsequent objects
	 */
	@Override
	public OrdinalMisclassificationMatrix validate(Data testData) {
		InformationTable testInformationTable = testData.getInformationTable();
		
		if (testInformationTable.getDecisions(true) == null) {
			throw new UnsupportedOperationException("Cannot validate data without decisions.");
		}

		int testDataSize = testInformationTable.getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testInformationTable.getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testInformationTable.getDecisions(true);
		SimpleDecision[] assignedDecisions = new SimpleDecision[testDataSize]; //will contain assigned decisions
	
		for (int testObjectIndex = 0; testObjectIndex < testDataSize; testObjectIndex++) {
			assignedDecisions[testObjectIndex] = simpleOptimizingRuleClassifier.classify(testObjectIndex, testInformationTable).getSuggestedDecision();
		}
		
		return new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
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

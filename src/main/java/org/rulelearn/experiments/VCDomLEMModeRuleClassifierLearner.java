/**
 * 
 */
package org.rulelearn.experiments;

import java.util.List;
import java.util.Random;

import org.rulelearn.classification.SimpleClassificationResult;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.core.ValueNotFoundException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;
import org.rulelearn.rules.RuleFilter;
import org.rulelearn.rules.RuleSetWithComputableCharacteristics;
import org.rulelearn.types.EvaluationField;
import org.rulelearn.wrappers.VCDomLEMWrapper;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class VCDomLEMModeRuleClassifierLearner extends AbstractLearningAlgorithm {
	
	/**
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public static enum DefaultClassificationResultChoiceMethod {
		
		/**
		 * Choose the most frequent class (mode) in the train data, using its seed.
		 */
		MODE,
		/**
		 * Choose median class in the train data.
		 */
		MEDIAN,
		/**
		 * Use decision class assumed a priori for the entire data.
		 */
		FIXED,
		/**
		 * Use decision class returned by some trained classifier.
		 */
		CLASSIFIER;
		
		/**
		 * @param defaultClassificationResultChoiceMethod
		 * @return constructed default classification result choice method
		 * @throws InvalidValueException if default classification result choice method cannot be parsed from given text
		 */
		public static DefaultClassificationResultChoiceMethod of(String defaultClassificationResultChoiceMethod) {
			if (defaultClassificationResultChoiceMethod.equalsIgnoreCase(MODE.toString())) {
				return MODE;
			}
			if (defaultClassificationResultChoiceMethod.equalsIgnoreCase(MEDIAN.toString())) {
				return MEDIAN;
			}
			if (defaultClassificationResultChoiceMethod.equalsIgnoreCase(FIXED.toString())) {
				return FIXED;
			}
			if (defaultClassificationResultChoiceMethod.equalsIgnoreCase(CLASSIFIER.toString())) {
				return CLASSIFIER;
			}
			
			throw new InvalidValueException("Invalid value of default classification result choice method.");
		}
		
		public String toString() {
			switch(this) {
			case MODE:
				return "mode";
			case MEDIAN:
				return "median";
			case FIXED:
				return "fixed";
			case CLASSIFIER:
				return "classifier";
			default:
				return "other";
			}
		}
	}
	
	/**
	 * 
	 * 
	 * @param trainData data for training this learning algorithm so it could learn a classification model from the data
	 * @param parameters parameters concerning application this learning algorithm to given train data
	 * 
	 * @throws ValueNotFoundException if {@link #getDefaultDecisionClassChoiceMethod default decision class choice method} is {@link DefaultClassificationResultChoiceMethod#FIXED}
	 *         and decision with label equal to {@link Data#getDefaultClassificationResultLabel() label of default classification result}
	 *         provided in given train data could not be found
	 */
	@Override
	public ModeRuleClassifier learn(Data trainData, LearningAlgorithmDataParameters parameters) {
		double consistencyThreshold = Double.valueOf(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.consistencyThresholdParameterName));
		RuleFilter ruleFilter = CompositeRuleCharacteristicsFilter.of(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.filterParameterName));
		DefaultClassificationResultChoiceMethod defaultDecisionClassChoiceMethod = DefaultClassificationResultChoiceMethod.of(
				parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.defaultClassificationResultChoiceMethodParameterName));
		
		//first try to get rules from cache
		RuleSetWithComputableCharacteristics ruleSetWithCharacteristics = VCDomLEMModeRuleClassifierLearnerCache.getInstance().getRules(trainData.getName(), consistencyThreshold);
		if (ruleSetWithCharacteristics == null) {
			ruleSetWithCharacteristics = (new VCDomLEMWrapper()).induceRulesWithCharacteristics(trainData.getInformationTable(), consistencyThreshold);
			//ruleSetWithCharacteristics.setLearningInformationTableHash(trainData.getInformationTable().getHash()); //save data hash along with rules - skipped to speed up computations
			VCDomLEMModeRuleClassifierLearnerCache.getInstance().putRules(trainData.getName(), consistencyThreshold, ruleSetWithCharacteristics); //store rules in cache for later use!
		}
		
		ruleSetWithCharacteristics = ruleSetWithCharacteristics.filter(ruleFilter);
		
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions;
		SimpleClassificationResult defaultClassificationResult = null;
		
		String modelLearnerDescription = (new StringBuilder(getName())).append("(").append(parameters).append(")").toString();
		
		switch (defaultDecisionClassChoiceMethod) {
		case MODE:
			if (!(trainData.getInformationTable() instanceof InformationTableWithDecisionDistributions)) {
				trainData.extendInformationTableWithDecisionDistributions(); //save extended data in fold, so next algorithms can use it
			}
			informationTableWithDecisionDistributions = (InformationTableWithDecisionDistributions)trainData.getInformationTable();
			
			List<Decision> modes = informationTableWithDecisionDistributions.getDecisionDistribution().getMode();
			if (modes.size() > 1) { //if modes.size() > 1, then choose mode randomly, using a seed
				Random random = new Random();
				random.setSeed(trainData.getSeed());
				int modeIndex = random.nextInt(modes.size());
				defaultClassificationResult = new SimpleClassificationResult((SimpleDecision)modes.get(modeIndex));
			} else {
				defaultClassificationResult = new SimpleClassificationResult((SimpleDecision)modes.get(0));
			}
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearnerDescription);
		case MEDIAN:
			if (!(trainData.getInformationTable() instanceof InformationTableWithDecisionDistributions)) {
				trainData.extendInformationTableWithDecisionDistributions(); //save extended data in fold, so next algorithms can use it
			}
			informationTableWithDecisionDistributions = (InformationTableWithDecisionDistributions)trainData.getInformationTable();
			
			Decision median = informationTableWithDecisionDistributions.getDecisionDistribution().getMedian(trainData.getInformationTable().getOrderedUniqueFullyDeterminedDecisions());
			defaultClassificationResult = new SimpleClassificationResult((SimpleDecision)median);
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearnerDescription);
		case FIXED:
			defaultClassificationResult = calculateDefaultClassificationResult(trainData, parameters);
			
			if (defaultClassificationResult == null) {
				throw new ValueNotFoundException("Could not find decision with requested label.");
			}
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearnerDescription);
		case CLASSIFIER:
			defaultClassificationResult = calculateDefaultClassificationResult(trainData, parameters); //just to make rule classifier happy ;) - this result will be always overriden by default model
			
			if (defaultClassificationResult == null) {
				throw new ValueNotFoundException("Could not find decision with requested label.");
			}
			
			VCDomLEMModeRuleClassifierLearnerDataParameters ruleParameters = ((VCDomLEMModeRuleClassifierLearnerDataParameters)parameters);
			ClassificationModel defaultClassificationModel = ruleParameters.getDefaultClassificationResultAlgorithm().learn(trainData,
					ruleParameters.getDefaultClassificationResultAlgorithmParameters());
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, defaultClassificationModel, modelLearnerDescription);
		default:
			throw new UnsupportedOperationException("Not supported default decision class choice method.");
		}
		
	}
	
	SimpleClassificationResult calculateDefaultClassificationResult(Data trainData, LearningAlgorithmDataParameters parameters) {
		SimpleClassificationResult defaultClassificationResult = null;
		
		Decision[] uniqueDecisions = trainData.getInformationTable().getUniqueDecisions();
		SimpleDecision testedDecision;
		
		int decisionAttributeIndex = ((SimpleDecision)uniqueDecisions[0]).getAttributeIndex(); //take decision attribute number from the first decision
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)trainData.getInformationTable().getAttribute(decisionAttributeIndex);
		
		EvaluationField testedEvaluation = decisionAttribute.getValueType().getDefaultFactory().create(
				parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.defaultClassificationResultLabelParameterName),
				decisionAttribute);
		testedDecision = new SimpleDecision(testedEvaluation, decisionAttributeIndex);
		
		for (Decision uniqueDecision : uniqueDecisions) {
			if (testedDecision.equals(uniqueDecision)) {
				defaultClassificationResult = new SimpleClassificationResult(testedDecision);
				break; //decision found
			}
		}
		
		return defaultClassificationResult;
	}

	@Override
	public String getName() {
		return getAlgorithmName();
	}
	
	public static String getAlgorithmName() {
		return VCDomLEMModeRuleClassifierLearner.class.getSimpleName();
	}

}

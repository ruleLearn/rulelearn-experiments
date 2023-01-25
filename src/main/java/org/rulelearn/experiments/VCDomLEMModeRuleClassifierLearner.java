/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.rulelearn.approximations.Union;
import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
import org.rulelearn.classification.SimpleClassificationResult;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.core.ValueNotFoundException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearnerCache.RuleSetWithComputableCharacteristicsPlusCalculationTime;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;
import org.rulelearn.rules.ApproximatedSetProvider;
import org.rulelearn.rules.ApproximatedSetRuleDecisionsProvider;
import org.rulelearn.rules.AttributeOrderRuleConditionsPruner;
import org.rulelearn.rules.CertainRuleInducerComponents;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;
import org.rulelearn.rules.DummyRuleConditionsGeneralizer;
import org.rulelearn.rules.EvaluationAndCoverageStoppingConditionChecker;
import org.rulelearn.rules.OptimizingRuleConditionsGeneralizer;
import org.rulelearn.rules.RuleFilter;
import org.rulelearn.rules.RuleInducerComponents;
import org.rulelearn.rules.RuleInductionStoppingConditionChecker;
import org.rulelearn.rules.RuleSetWithComputableCharacteristics;
import org.rulelearn.rules.UnionProvider;
import org.rulelearn.rules.UnionWithSingleLimitingDecisionRuleDecisionsProvider;
import org.rulelearn.rules.VCDomLEM;
import org.rulelearn.types.EvaluationField;

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
	
	private class DataTransformationTime {
		long duration = 0L;

		public DataTransformationTime() {
			this.duration = 0L;
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

		DataTransformationTime dataTransformationTime = new DataTransformationTime();
		
		//first try to get rules from cache
		boolean useConditionGeneralization = Boolean.valueOf(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.useConditionGeneralizationParameterName));
		RuleSetWithComputableCharacteristicsPlusCalculationTime ruleSetWithCharacteristicsPlusCalculationTime = VCDomLEMModeRuleClassifierLearnerCache.getInstance().getRules(
				trainData.getName(), consistencyThreshold, useConditionGeneralization);
		RuleSetWithComputableCharacteristics ruleSetWithCharacteristics;
		long timeSavedByRetreivingRulesFromCache;
		
		if (ruleSetWithCharacteristicsPlusCalculationTime == null) {
			timeSavedByRetreivingRulesFromCache = 0L;
			
			//***********************************
			long start = System.currentTimeMillis();
			ruleSetWithCharacteristics = learnRules(trainData.getInformationTable(), consistencyThreshold, useConditionGeneralization, dataTransformationTime);
			long ruleSetCalculationTime = System.currentTimeMillis() - start;
			//***********************************
			
			//ruleSetWithCharacteristics.setLearningInformationTableHash(trainData.getInformationTable().getHash()); //save data hash along with rules - skipped to speed up computations
			VCDomLEMModeRuleClassifierLearnerCache.getInstance().putRules(trainData.getName(), consistencyThreshold, useConditionGeneralization, ruleSetWithCharacteristics, ruleSetCalculationTime, dataTransformationTime.duration); //store rules in cache for later use!
		} else {
			//remember saved time used earlier to calculate rules
			dataTransformationTime.duration = ruleSetWithCharacteristicsPlusCalculationTime.getInformationTableTransformationTime();
			timeSavedByRetreivingRulesFromCache = ruleSetWithCharacteristicsPlusCalculationTime.getCalculationTime();
			ruleSetWithCharacteristics = ruleSetWithCharacteristicsPlusCalculationTime.getRuleSet();
		}
		
//		boolean useFilter = Boolean.valueOf(parameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.useFilterParameterName));
//		if (useFilter) {
//			ruleSetWithCharacteristics = ruleSetWithCharacteristics.filter(ruleFilter);
//		} else {
//			ruleSetWithCharacteristics = ruleSetWithCharacteristics.filter(CompositeRuleCharacteristicsFilter.of("s > 0"));
//		}
		ruleSetWithCharacteristics = ruleSetWithCharacteristics.filter(ruleFilter);
		
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions;
		SimpleClassificationResult defaultClassificationResult = null;
		
		//calculate ModelLearningStatistics
		//+++++
		long start = System.currentTimeMillis();
		int numberOfLearningObjects = trainData.getInformationTable().getNumberOfObjects();
		
		Integer numberOfConsistentLearningObjectsObj = NumberOfConsistentObjectsCache.getInstance().getNumberOfConsistentObjects(trainData.getName(), 0.0);
		int numberOfConsistentLearningObjects;
		if (numberOfConsistentLearningObjectsObj != null) { //number of objects already in cache
			numberOfConsistentLearningObjects = numberOfConsistentLearningObjectsObj.intValue();
		} else { //number of objects not yet in cache
			numberOfConsistentLearningObjects = ClassificationModel.getNumberOfConsistentObjects(trainData.getInformationTable(), 0.0);
			NumberOfConsistentObjectsCache.getInstance().putNumberOfConsistentObjects(trainData.getName(), 0.0, numberOfConsistentLearningObjects); //store calculated number of objects in cache
		}
		
		//consistencyThreshold - defined above
		
		Integer numberOfConsistentLearningObjectsForConsistencyThresholdObj = NumberOfConsistentObjectsCache.getInstance().getNumberOfConsistentObjects(trainData.getName(), consistencyThreshold);
		int numberOfConsistentLearningObjectsForConsistencyThreshold;
		if (numberOfConsistentLearningObjectsForConsistencyThresholdObj != null) { //number of objects already in cache
			numberOfConsistentLearningObjectsForConsistencyThreshold = numberOfConsistentLearningObjectsForConsistencyThresholdObj.intValue();
		} else {
			numberOfConsistentLearningObjectsForConsistencyThreshold = ClassificationModel.getNumberOfConsistentObjects(trainData.getInformationTable(), consistencyThreshold);
			NumberOfConsistentObjectsCache.getInstance().putNumberOfConsistentObjects(trainData.getName(), consistencyThreshold, numberOfConsistentLearningObjectsForConsistencyThreshold); //store calculated number of objects in cache
		}
		
		String modelLearnerDescription = (new StringBuilder(getName())).append("(").append(parameters).append(")").toString();
		long statisticsCountingTime = System.currentTimeMillis() - start;
		
//		//account for calculation of InformationTableWithDecisionDistributions from InformationTable
//		statisticsCountingTime += dataTransformationTime.duration;
		
		ModelLearningStatistics modelLearningStatistics = new ModelLearningStatistics(
				numberOfLearningObjects, numberOfConsistentLearningObjects, consistencyThreshold, numberOfConsistentLearningObjectsForConsistencyThreshold,
				modelLearnerDescription, dataTransformationTime.duration, timeSavedByRetreivingRulesFromCache, statisticsCountingTime); //separately account for data transformation time
		//+++++
		
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
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearningStatistics);
		case MEDIAN:
			if (!(trainData.getInformationTable() instanceof InformationTableWithDecisionDistributions)) {
				trainData.extendInformationTableWithDecisionDistributions(); //save extended data in fold, so next algorithms can use it
			}
			informationTableWithDecisionDistributions = (InformationTableWithDecisionDistributions)trainData.getInformationTable();
			
			Decision median = informationTableWithDecisionDistributions.getDecisionDistribution().getMedian(trainData.getInformationTable().getOrderedUniqueFullyDeterminedDecisions());
			defaultClassificationResult = new SimpleClassificationResult((SimpleDecision)median);
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearningStatistics);
		case FIXED:
			defaultClassificationResult = calculateDefaultClassificationResult(trainData, parameters);
			
			if (defaultClassificationResult == null) {
				throw new ValueNotFoundException("Could not find decision with requested label.");
			}
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, modelLearningStatistics);
		case CLASSIFIER:
			defaultClassificationResult = calculateDefaultClassificationResult(trainData, parameters); //just to make rule classifier happy ;) - this result will be always overriden by default model
			
			if (defaultClassificationResult == null) {
				throw new ValueNotFoundException("Could not find decision with requested label.");
			}
			
			VCDomLEMModeRuleClassifierLearnerDataParameters ruleParameters = ((VCDomLEMModeRuleClassifierLearnerDataParameters)parameters);
			ClassificationModel defaultClassificationModel = ruleParameters.getDefaultClassificationResultAlgorithm().learn(trainData,
					ruleParameters.getDefaultClassificationResultAlgorithmParameters());
			
			return new ModeRuleClassifier(ruleSetWithCharacteristics, defaultClassificationResult, defaultClassificationModel, modelLearningStatistics);
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
	
	RuleSetWithComputableCharacteristics learnRules(InformationTable informationTable, double consistencyThreshold, boolean useConditionGeneralization,
			DataTransformationTime dataTransformationTime) { //dataProcessingTime just 1 element - data processing time in ms
		//the code below is copied from method VCDomLEMWrapper.induceRulesWithCharacteristics(InformationTable informationTable, double consistencyThreshold,
		//with adjusted rule conditions generalizer and skipped calculation of all rule characteristics
		final RuleInductionStoppingConditionChecker stoppingConditionChecker = 
				new EvaluationAndCoverageStoppingConditionChecker(EpsilonConsistencyMeasure.getInstance(), EpsilonConsistencyMeasure.getInstance(),
						EpsilonConsistencyMeasure.getInstance(), consistencyThreshold);
		
//			RuleInducerComponents ruleInducerComponents = new CertainRuleInducerComponents.Builder().
//					ruleInductionStoppingConditionChecker(stoppingConditionChecker).
//					ruleConditionsPruner(new DummyRuleConditionsPruner()).
//					ruleConditionsGeneralizer(new DummyRuleConditionsGeneralizer()). //ADDED-RULE-CONDITIONS-GENERALIZER
//					ruleConditionsSetPruner(new DummyRuleConditionsSetPruner()).
//					ruleMinimalityChecker(new DummyRuleMinimalityChecker()).
//					allowedNegativeObjectsType(AllowedNegativeObjectsType.ANY_REGION).
//					build(); //TODO: remove!
		
		RuleInducerComponents ruleInducerComponents = new CertainRuleInducerComponents.Builder().
				ruleInductionStoppingConditionChecker(stoppingConditionChecker).
				ruleConditionsPruner(new AttributeOrderRuleConditionsPruner(stoppingConditionChecker)).
				ruleConditionsGeneralizer(useConditionGeneralization ? new OptimizingRuleConditionsGeneralizer(stoppingConditionChecker) : new DummyRuleConditionsGeneralizer()). //THE CHANGE HERE!
				build();
		
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions;
		if (informationTable instanceof InformationTableWithDecisionDistributions) {
			informationTableWithDecisionDistributions = (InformationTableWithDecisionDistributions)informationTable;
			dataTransformationTime.duration = 0L;
		} else {
			long start = System.currentTimeMillis();
			informationTableWithDecisionDistributions = new InformationTableWithDecisionDistributions(informationTable, true, true);
			dataTransformationTime.duration = System.currentTimeMillis() - start; //record time of getting from InformationTable to InformationTableWithDecisionDistributions
		}
		
		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, 
								   new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
		ApproximatedSetProvider unionAtLeastProvider = new UnionProvider(Union.UnionType.AT_LEAST, unions);
		ApproximatedSetProvider unionAtMostProvider = new UnionProvider(Union.UnionType.AT_MOST, unions);
		ApproximatedSetRuleDecisionsProvider unionRuleDecisionsProvider = new UnionWithSingleLimitingDecisionRuleDecisionsProvider();
		
		List<VCDomLEM> vcDomLEMs = new ArrayList<VCDomLEM>(2);
		vcDomLEMs.add(new VCDomLEM(ruleInducerComponents, unionAtLeastProvider, unionRuleDecisionsProvider));
		vcDomLEMs.add(new VCDomLEM(ruleInducerComponents, unionAtMostProvider, unionRuleDecisionsProvider));
		
		//calculate rules (and their characteristics) in parallel
		List<RuleSetWithComputableCharacteristics> ruleSets = vcDomLEMs.parallelStream().map(vcDomLem -> vcDomLem.generateRules()).collect(Collectors.toList());
		//ruleSets.parallelStream().forEach(ruleSet -> ruleSet.calculateAllCharacteristics()); //THE CHANGE HERE!
		
//			System.out.println(ruleSets.get(0).serialize()); //TODO: remove
//			System.out.println();
//			System.out.println(ruleSets.get(1).serialize());
//			
		return RuleSetWithComputableCharacteristics.join(ruleSets.get(0), ruleSets.get(1));
	}

}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Map;

import org.rulelearn.rules.RuleSetWithComputableCharacteristics;

/**
 * Caches rules generated by {@link VCDomLEMModeRuleClassifierLearner} for many data sets and single consistency threshold for each of these data sets (so they can be used several times).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class VCDomLEMModeRuleClassifierLearnerCache {
	
	/**
	 * Stores rules together with their calculation time.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public static class RuleSetWithComputableCharacteristicsPlusCalculationTime {
		RuleSetWithComputableCharacteristics ruleSet;
		long calculationTime; //total time, including information table transformation time and calculation of rules on transformed information table [ms]
		long informationTableTransformationTime; //sole information table transformation time [ms]
		
		public RuleSetWithComputableCharacteristicsPlusCalculationTime(RuleSetWithComputableCharacteristics ruleSet, long calculationTime, long informationTableTransformationTime) {
			this.ruleSet = ruleSet;
			this.calculationTime = calculationTime;
			this.informationTableTransformationTime = informationTableTransformationTime;
		}

		public RuleSetWithComputableCharacteristics getRuleSet() {
			return ruleSet;
		}

		public long getCalculationTime() {
			return calculationTime;
		}

		public long getInformationTableTransformationTime() {
			return informationTableTransformationTime;
		}
		
	}
	
	/**
	 * The only instance of this class (singleton).
	 */
	static private VCDomLEMModeRuleClassifierLearnerCache instance = null;
	
	/**
	 * Singleton providing method.
	 * 
	 * @return the only instance of this class (singleton)
	 */
	static VCDomLEMModeRuleClassifierLearnerCache getInstance() {
		if (instance == null) {
			instance = new VCDomLEMModeRuleClassifierLearnerCache();
		}
		return instance;
	}
	
	Map<String, Map<Double, Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>>> dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap = new HashMap<String, Map<Double, Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>>>();
	
	public RuleSetWithComputableCharacteristicsPlusCalculationTime getRules(String dataSetName, double consistencyThreshold, boolean useConditionGeneralization) { //can return null
		RuleSetWithComputableCharacteristicsPlusCalculationTime result = null;
		
		if (dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.containsKey(dataSetName)) {
			if (dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).containsKey(consistencyThreshold)) {
				if (dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).get(consistencyThreshold).containsKey(useConditionGeneralization)) {
					result = dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).get(consistencyThreshold).get(useConditionGeneralization);
				}
			}
		}
		
		return result;
	}
	
	public void putRules(String dataSetName, double consistencyThreshold, boolean useConditionGeneralization, RuleSetWithComputableCharacteristics rules,
			long calculationTime, long informationTableTransformationTime) {
		if (dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.containsKey(dataSetName)) {
			if (dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).containsKey(consistencyThreshold)) {
				dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).get(consistencyThreshold).put(useConditionGeneralization,
						new RuleSetWithComputableCharacteristicsPlusCalculationTime(rules, calculationTime, informationTableTransformationTime));
			} else { //no mapping for given consistency threshold
				Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime> useConditionGeneralizationVSRulesMap = new HashMap<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>();
				useConditionGeneralizationVSRulesMap.put(useConditionGeneralization, new RuleSetWithComputableCharacteristicsPlusCalculationTime(rules, calculationTime, informationTableTransformationTime));
				dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.get(dataSetName).put(consistencyThreshold, useConditionGeneralizationVSRulesMap);
			}
		} else { //no mapping for given data set name
			Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime> useConditionGeneralizationVSRulesMap = new HashMap<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>();
			useConditionGeneralizationVSRulesMap.put(useConditionGeneralization, new RuleSetWithComputableCharacteristicsPlusCalculationTime(rules, calculationTime, informationTableTransformationTime));
			Map<Double, Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>> consistencyThreshold2UseConditionGeneralizationVSRulesMap = new HashMap<Double, Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>>();
			consistencyThreshold2UseConditionGeneralizationVSRulesMap.put(consistencyThreshold, useConditionGeneralizationVSRulesMap);
			dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.put(dataSetName, consistencyThreshold2UseConditionGeneralizationVSRulesMap);
		}
	}
	
	public void clear() {
		dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap = new HashMap<String, Map<Double, Map<Boolean, RuleSetWithComputableCharacteristicsPlusCalculationTime>>>(); //clear map to free memory
	}
	
	public void clear(String dataSetName) {
		dataSetName2ConsistencyThreshold2UseConditionGeneralizationVSRulesMap.remove(dataSetName); //clears cache for the given data set name (leaving other mappings, i.e., for other fold train data)
	}

}

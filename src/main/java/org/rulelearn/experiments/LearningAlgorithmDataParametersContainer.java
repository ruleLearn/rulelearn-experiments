/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class LearningAlgorithmDataParametersContainer {
	Map<String, Map<String, List<LearningAlgorithmDataParameters>>> map = new HashMap<String, Map<String, List<LearningAlgorithmDataParameters>>>();
	
	public LearningAlgorithmDataParametersContainer putParameters(String algorithmName, String dataName, List<LearningAlgorithmDataParameters> learningAlgorithmDataParametersList) {
		if (map.containsKey(algorithmName)) {
			map.get(algorithmName).put(dataName, learningAlgorithmDataParametersList);
		} else {
			Map<String, List<LearningAlgorithmDataParameters>> value = new HashMap<String, List<LearningAlgorithmDataParameters>>();
			value.put(dataName, learningAlgorithmDataParametersList);
			map.put(algorithmName, value);
		}
		return this;
	}
	
	public LearningAlgorithmDataParametersContainer putParameters(LearningAlgorithm algorithm, Data data, List<LearningAlgorithmDataParameters> learningAlgorithmDataParametersList) {
		return putParameters(algorithm.getName(), data.getName(), learningAlgorithmDataParametersList);
	}
	
	//returns null if algorithmName not found, or if for algorithmName dataName not found
	public List<LearningAlgorithmDataParameters> getParameters(String algorithmName, String dataName) {
		if (map.containsKey(algorithmName)) {
			return map.get(algorithmName).get(dataName);
		} else {
			return null;
		}
	}
	
	//returns null if algorithm not found, or if for algorithm data not found
	public List<LearningAlgorithmDataParameters> getParameters(LearningAlgorithm algorithm, Data data) {
		return getParameters(algorithm.getName(), data.getName());
	}
	
	//concerns only parameters of VCDomLEMModeRuleClassifierLearner algorithm, for any data set
	public void sortParametersLists() {
		map.forEach((algorithmName, dataName2ParametersList) -> {
			if (algorithmName.equals(VCDomLEMModeRuleClassifierLearner.getAlgorithmName())) {
				dataName2ParametersList.forEach((dataName, parametersList) -> {
					parametersList.sort((firstParameters, secondParameters) -> {
						double threshold1 = Double.valueOf(firstParameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.consistencyThresholdParameterName));
						double threshold2 = Double.valueOf(secondParameters.getParameter(VCDomLEMModeRuleClassifierLearnerDataParameters.consistencyThresholdParameterName));
						if (threshold1 < threshold2) {
							return -1;
						} else if (threshold1 == threshold2) {
							return 0;
						} else {
							return 1;
						}
					});
				});
			}
		});
	}
	
}

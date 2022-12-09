/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class LearningAlgorithmDataParametersContainer {
	Map<String, Map<String, LearningAlgorithmDataParameters>> map = new HashMap<String, Map<String, LearningAlgorithmDataParameters>>();
	
	public LearningAlgorithmDataParametersContainer putParameters(String algorithmName, String dataName, LearningAlgorithmDataParameters learningAlgorithmDataParameters) {
		if (map.containsKey(algorithmName)) {
			map.get(algorithmName).put(dataName, learningAlgorithmDataParameters);
		} else {
			Map<String, LearningAlgorithmDataParameters> value = new HashMap<String, LearningAlgorithmDataParameters>();
			value.put(dataName, learningAlgorithmDataParameters);
			map.put(algorithmName, value);
		}
		return this;
	}
	
	public LearningAlgorithmDataParametersContainer putParameters(LearningAlgorithm algorithm, Data data, LearningAlgorithmDataParameters learningAlgorithmDataParameters) {
		return (putParameters(algorithm.getName(), data.getName(), learningAlgorithmDataParameters));
	}
	
	//returns null if algorithmName not found, or if for algorithmName dataName not found
	public LearningAlgorithmDataParameters getParameters(String algorithmName, String dataName) {
		if (map.containsKey(algorithmName)) {
			return map.get(algorithmName).get(dataName);
		} else {
			return null;
		}
	}
	
	//returns null if algorithm not found, or if for algorithm data not found
	public LearningAlgorithmDataParameters getParameters(LearningAlgorithm algorithm, Data data) {
		return getParameters(algorithm.getName(), data.getName());
	}
}

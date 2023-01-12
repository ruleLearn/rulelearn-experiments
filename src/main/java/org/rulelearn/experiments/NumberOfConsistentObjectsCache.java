/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Map;

/**
 * Caches classification quality many (data set name, consistency threshold) pairs.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class NumberOfConsistentObjectsCache {
	
	static private NumberOfConsistentObjectsCache instance = null;
	
	static NumberOfConsistentObjectsCache getInstance() {
		if (instance == null) {
			instance = new NumberOfConsistentObjectsCache();
		}
		return instance;
	}
	
	Map<String, Map<Double, Integer>> dataSetName2ConsistencyThreshold2QualityOfClassification = new HashMap<String, Map<Double, Integer>>();
	
	public Integer getNumberOfConsistentObjects(String dataSetName, double consistencyThreshold) { //can return null
		Integer result = null;
		
		if (dataSetName2ConsistencyThreshold2QualityOfClassification.containsKey(dataSetName)) {
			Double key = Double.valueOf(consistencyThreshold);
			if (dataSetName2ConsistencyThreshold2QualityOfClassification.get(dataSetName).containsKey(key)) {
				result = dataSetName2ConsistencyThreshold2QualityOfClassification.get(dataSetName).get(key);
			}
		}
		
		return result;
	}
	
	private HashMap<Double, Integer> put(HashMap<Double, Integer> hashMap, Double key, Integer value) {
		hashMap.put(key, value);
		return hashMap;
	}
	
	public void putNumberOfConsistentObjects(String dataSetName, double consistencyThreshold, int numberOfConsistentObjects) {
		if (dataSetName2ConsistencyThreshold2QualityOfClassification.containsKey(dataSetName)) {
			dataSetName2ConsistencyThreshold2QualityOfClassification.get(dataSetName).put(consistencyThreshold, Integer.valueOf(numberOfConsistentObjects));
		} else {
			dataSetName2ConsistencyThreshold2QualityOfClassification.put(dataSetName, put(new HashMap<Double, Integer>(), consistencyThreshold, Integer.valueOf(numberOfConsistentObjects)));
		}
	}
	
	public void clear() {
		dataSetName2ConsistencyThreshold2QualityOfClassification = new HashMap<String, Map<Double, Integer>>(); //clear map to free memory
	}
	
	public void clear(String dataSetName) {
		dataSetName2ConsistencyThreshold2QualityOfClassification.remove(dataSetName); //clears cache for the given data set name (leaving other mappings, e.g., for other fold train data)
	}

}

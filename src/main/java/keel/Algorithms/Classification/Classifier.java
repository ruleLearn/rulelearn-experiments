/**
 * 
 */
package keel.Algorithms.Classification;

import keel.Dataset.InstanceSet;

/**
 * Generic KEEL classification algorithm (classifier).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface Classifier {
	
	/**
	 * Builds classifier for given train data.
	 * 
	 * @param trainData
	 */
	void buildClassifier(InstanceSet trainData);
	
	/**
	 * Classifies all instances from the given test data set. Assumes that the data have one nominal output attribute.
	 * 
	 * @param testData test data set containing instances to be classified
	 * @return array with indices of nominal decisions assigned to subsequent instances
	 */
	int[] classify(InstanceSet testData);
	
}

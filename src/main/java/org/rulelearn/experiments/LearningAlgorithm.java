/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Learns model from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface LearningAlgorithm {
	
	ClassificationModel learn(Data data, LearningAlgorithmDataParameters parameters);
	String getName();
	
	int hashCode();
	boolean equals(Object other);
}

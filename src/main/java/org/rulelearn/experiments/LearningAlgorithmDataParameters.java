/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Supplies parameters concerning a learning algorithm and particular full data set.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface LearningAlgorithmDataParameters {
	
	String getParameter(String parameterName);
	
	/**
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public interface Builder {
		Builder parameters(String parameters);
		LearningAlgorithmDataParameters build(); //uses textual representation of parameters
	}
	
}

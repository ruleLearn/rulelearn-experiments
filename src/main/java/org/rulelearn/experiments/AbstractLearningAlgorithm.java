/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Objects;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class AbstractLearningAlgorithm implements LearningAlgorithm {

	@Override
	public int hashCode() {
		return Objects.hash(this.getClass(), getName());
	}
	
	@Override
	public boolean equals(Object other) {
		return (other instanceof AbstractLearningAlgorithm) && getName().equals(((AbstractLearningAlgorithm)other).getName());
	}

}

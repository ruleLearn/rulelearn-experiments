/**
 * 
 */
package org.rulelearn.experiments;

/**
 * Provides a fresh instance of {@link CrossValidation}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface CrossValidationProvider {
	
	CrossValidation provide();
}

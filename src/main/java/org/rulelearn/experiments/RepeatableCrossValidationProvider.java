/**
 * 
 */
package org.rulelearn.experiments;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class RepeatableCrossValidationProvider implements CrossValidationProvider {

	@Override
	public RepeatableCrossValidation provide() {
		return new RepeatableCrossValidation();
	}

}

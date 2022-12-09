/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.experiments.VCDomLEMRulesModeClassifier.DefaultClassificationResultChoiceMethod;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class VCDomLEMRulesModeClassifierDataParameters implements LearningAlgorithmDataParameters {
	
	public static final String consistencyThresholdParameterName = "consistencyThreshold";
	public static final String filterParameterName = "filter";
	public static final String defaultClassificationResultLabelParameterName = "defaultClassificationResultLabel";
	public static final String defaultClassificationResultChoiceMethodParameterName = "defaultClassificationResultChoiceMethod";
	
	Map<String, String> parameters;

	public VCDomLEMRulesModeClassifierDataParameters(double consistencyThreshold, CompositeRuleCharacteristicsFilter filter, String defaultClassificationResultLabel) {
		parameters = new HashMap<String, String>();
		parameters.put(consistencyThresholdParameterName, String.valueOf(consistencyThreshold));
		parameters.put(filterParameterName, filter.toString());
		parameters.put(defaultClassificationResultLabelParameterName, defaultClassificationResultLabel); //default label for each test subset
		parameters.put(defaultClassificationResultChoiceMethodParameterName, DefaultClassificationResultChoiceMethod.FIXED.toString());
	}
	
	/**
	 * @param consistencyThreshold
	 * @param filter
	 * @param defaultClassificationResultChoiceMethod
	 * 
	 * @throws InvalidValueException if given default classification result choice method is neither {@link DefaultClassificationResultChoiceMethod#MODE} nor {@link DefaultClassificationResultChoiceMethod#MEDIAN}
	 */
	public VCDomLEMRulesModeClassifierDataParameters(double consistencyThreshold, CompositeRuleCharacteristicsFilter filter,
			DefaultClassificationResultChoiceMethod defaultClassificationResultChoiceMethod) {
		parameters = new HashMap<String, String>();
		parameters.put(consistencyThresholdParameterName, String.valueOf(consistencyThreshold));
		parameters.put(filterParameterName, filter.toString());
		//parameters.put(defaultClassificationResultLabelParameterName, null);
		if (defaultClassificationResultChoiceMethod == DefaultClassificationResultChoiceMethod.MODE || defaultClassificationResultChoiceMethod == DefaultClassificationResultChoiceMethod.MEDIAN) {
			parameters.put(defaultClassificationResultChoiceMethodParameterName, defaultClassificationResultChoiceMethod.toString());
		} else {
			throw new InvalidValueException("Invalid value of default classification result choice method.");
		}
	}

	@Override
	public String getParameter(String parameterName) {
		return parameters.get(parameterName);
	}
	
	@Override
	public String toString() {
		return String.format(Locale.US, "%s=%s, %s=%s, %s=%s%s", 
				consistencyThresholdParameterName, parameters.get(consistencyThresholdParameterName),
				filterParameterName, parameters.get(filterParameterName),
				defaultClassificationResultChoiceMethodParameterName, parameters.get(defaultClassificationResultChoiceMethodParameterName),
				DefaultClassificationResultChoiceMethod.of(parameters.get(defaultClassificationResultChoiceMethodParameterName)) == DefaultClassificationResultChoiceMethod.FIXED ?
						"("+parameters.get(defaultClassificationResultLabelParameterName)+")" : "");
	}

}

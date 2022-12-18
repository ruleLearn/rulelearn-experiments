/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearner.DefaultClassificationResultChoiceMethod;
import org.rulelearn.rules.CompositeRuleCharacteristicsFilter;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class VCDomLEMModeRuleClassifierLearnerDataParameters implements LearningAlgorithmDataParameters {
	
	public static final String consistencyThresholdParameterName = "consistencyThreshold";
	public static final String filterParameterName = "filter";
	public static final String defaultClassificationResultLabelParameterName = "defaultClassificationResultLabel";
	public static final String defaultClassificationResultChoiceMethodParameterName = "defaultClassificationResultChoiceMethod";
	
//	public static final String defaultClassificationResultPredictorParameterName = "defaultClassificationResultPredictor";
//	public static final String defaultClassificationResultPredictorOptionsParameterName = "defaultClassificationResultPredictorOptions";
	
	Map<String, String> parameters;
	
	LearningAlgorithm defaultClassificationResultAlgorithm = null;
	LearningAlgorithmDataParameters defaultClassificationResultAlgorithmParameters = null;

	public VCDomLEMModeRuleClassifierLearnerDataParameters(double consistencyThreshold, CompositeRuleCharacteristicsFilter filter,
			String defaultClassificationResultLabel) {
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
	public VCDomLEMModeRuleClassifierLearnerDataParameters(double consistencyThreshold, CompositeRuleCharacteristicsFilter filter,
			DefaultClassificationResultChoiceMethod defaultClassificationResultChoiceMethod) {
		parameters = new HashMap<String, String>();
		parameters.put(consistencyThresholdParameterName, String.valueOf(consistencyThreshold));
		parameters.put(filterParameterName, filter.toString());

		if (defaultClassificationResultChoiceMethod == DefaultClassificationResultChoiceMethod.MODE || defaultClassificationResultChoiceMethod == DefaultClassificationResultChoiceMethod.MEDIAN) {
			parameters.put(defaultClassificationResultChoiceMethodParameterName, defaultClassificationResultChoiceMethod.toString());
		} else {
			throw new InvalidValueException("Invalid value of default classification result choice method.");
		}
	}
	
	/**
	 * @param consistencyThreshold
	 * @param filter
	 * @param defaultClassificationResultPredictor {@link LearningAlgorithm algorithm} to be trained along with the main algorithm, used as a default class provider (when no rule covers classified object)
	 * @param predictorParameters parameters of the default classification result predictor
	 */
	public VCDomLEMModeRuleClassifierLearnerDataParameters(double consistencyThreshold, CompositeRuleCharacteristicsFilter filter,
			String defaultClassificationResultLabel, //will not be used!
			LearningAlgorithm defaultClassificationResultPredictor,
			LearningAlgorithmDataParameters predictorParameters) {
		parameters = new HashMap<String, String>();
		parameters.put(consistencyThresholdParameterName, String.valueOf(consistencyThreshold));
		parameters.put(filterParameterName, filter.toString());
		parameters.put(defaultClassificationResultLabelParameterName, defaultClassificationResultLabel); //default label for main model

		parameters.put(defaultClassificationResultChoiceMethodParameterName, DefaultClassificationResultChoiceMethod.CLASSIFIER.toString());
		
		this.defaultClassificationResultAlgorithm = defaultClassificationResultPredictor;
		this.defaultClassificationResultAlgorithmParameters = predictorParameters;
		
//		parameters.put(defaultClassificationResultPredictorParameterName, defaultClassificationResultPredictor.getName());
//		parameters.put(defaultClassificationResultPredictorOptionsParameterName, predictorOptions.toString());
	}

	@Override
	public String getParameter(String parameterName) {
		return parameters.get(parameterName);
	}
	
	@Override
	public String toString() {
		return String.format(Locale.US, "%s=%s, %s=%s, %s=%s%s%s", 
				consistencyThresholdParameterName, parameters.get(consistencyThresholdParameterName),
				filterParameterName, parameters.get(filterParameterName),
				defaultClassificationResultChoiceMethodParameterName, parameters.get(defaultClassificationResultChoiceMethodParameterName),
				DefaultClassificationResultChoiceMethod.of(parameters.get(defaultClassificationResultChoiceMethodParameterName)) == DefaultClassificationResultChoiceMethod.FIXED ?
						String.format(Locale.US, "(%s)", parameters.get(defaultClassificationResultLabelParameterName)) : "",
				DefaultClassificationResultChoiceMethod.of(parameters.get(defaultClassificationResultChoiceMethodParameterName)) == DefaultClassificationResultChoiceMethod.CLASSIFIER ?
						String.format(Locale.US, "(%s > %s(%s))",
								parameters.get(defaultClassificationResultLabelParameterName),
								defaultClassificationResultAlgorithm.getName(), defaultClassificationResultAlgorithmParameters) : "");
	}
	
	public LearningAlgorithm getDefaultClassificationResultAlgorithm() {
		return defaultClassificationResultAlgorithm;
	}

	public LearningAlgorithmDataParameters getDefaultClassificationResultAlgorithmParameters() {
		return defaultClassificationResultAlgorithmParameters;
	}

	public static class Builder implements LearningAlgorithmDataParameters.Builder {
		String parameters = null;
		
		public Builder() {}
		
		/**
		 * Stores textual representation of algorithm parameters. Expects "options=&lt;options&gt;", e.g., "consistencyThreshold=0.04, filter=S>0&coverage-factor>=0.025, defaultClassificationResultChoiceMethod=fixed(0)".
		 * 
		 * @param parameters textual representation of algorithm parameters
		 * @return this builder
		 */
		@Override
		public Builder parameters(String parameters) {
			this.parameters = parameters;
			return this;
		}
		
		@Override
		public VCDomLEMModeRuleClassifierLearnerDataParameters build() { //parses parameters
			//VCDomLEMRulesModeClassifierDataParameters result = null; //default result //TODO: uncomment
			
			if (parameters != null && parameters.length() > 0) {
				String[] values = parameters.split(",");
				if (values.length >= 3) {
					//TODO: implement
				}
			}
			
			//return result; //TODO: uncomment
			throw new UnsupportedOperationException("VCDomLEMRulesModeClassifierDataParameters.build method not implemented yet."); //for now
		}
		
	}
	
}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAAlgorithmOptions implements LearningAlgorithmDataParameters {
	
	public static final String optionsParameterName = "options";
	
	Map<String, String> parameters;
	
	public WEKAAlgorithmOptions(String options) {
		parameters = new HashMap<String, String>();
		parameters.put(optionsParameterName, options);
	}

	@Override
	public String getParameter(String parameterName) {
		return parameters.get(parameterName);
	}
	
	@Override
	public String toString() {
		return String.format(Locale.US, "%s=%s", optionsParameterName, parameters.get(optionsParameterName));
	}
	
	public static class Builder implements LearningAlgorithmDataParameters.Builder {
		String parameters = null;
		
		public Builder() {}
		
		/**
		 * Stores textual representation of algorithm parameters. Expects "options=&lt;options&gt;", e.g., "options=-D" for Naive Bayes algorithm.
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
		public WEKAAlgorithmOptions build() { //parses parameters
			WEKAAlgorithmOptions result = null; //default result
			
			if (parameters != null && parameters.length() > 0) {
				String[] values = parameters.split("=");
				if (values.length >= 2 && values[0].trim().toLowerCase().equals(optionsParameterName)) { //== 2
					result = new WEKAAlgorithmOptions(values[1].trim());
				}
			}
			
			return result;
		}
		
	}
	
}

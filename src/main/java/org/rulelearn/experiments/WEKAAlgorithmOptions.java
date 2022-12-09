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
	
}

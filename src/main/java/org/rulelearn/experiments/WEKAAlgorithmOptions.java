/**
 * 
 */
package org.rulelearn.experiments;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import weka.filters.Filter;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAAlgorithmOptions implements LearningAlgorithmDataParameters {
	
	public static final String optionsParameterName = "options";
	
	Map<String, String> parameters;
	Supplier<Filter[]> filtersProvider = null; //not used if null
	
	public WEKAAlgorithmOptions(String options) {
		parameters = new HashMap<String, String>();
		parameters.put(optionsParameterName, options);
	}
	
	public WEKAAlgorithmOptions(String options, Supplier<Filter[]> filtersProvider) {
		parameters = new HashMap<String, String>();
		parameters.put(optionsParameterName, options);
		
		this.filtersProvider = filtersProvider;
	}

	@Override
	public String getParameter(String parameterName) {
		return parameters.get(parameterName);
	}
	
	public Supplier<Filter[]> getFiltersProvider() {
		return filtersProvider;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(64);
		if (filtersProvider != null) {
			Filter[] filters = filtersProvider.get();
			if (filters != null) {
				if (filters.length > 0) {
					sb.append("; ");
				}
				int i = 0;
				for (Filter filter : filters) {
					sb.append(filter.getClass().getSimpleName());
					if (i < filters.length - 1) {
						sb.append("|");
					}
					i++;
				}
			}
		}
		return String.format(Locale.US, "%s=%s%s", optionsParameterName, parameters.get(optionsParameterName), sb.toString());
	}
	
	public static class Builder implements LearningAlgorithmDataParameters.Builder {
		String parameters = null;
		
		//TODO: extend for filters
		
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

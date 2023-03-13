package org.rulelearn.experiments;

/**
 * Provides a fresh instance of {@link AcceptingDataProcessor}.
 *
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class AcceptingDataProcessorProvider implements DataProcessorProvider {

	@Override
	public DataProcessor provide() {
		return new AcceptingDataProcessor();
	}
	
	@Override
	public String toString() {
		return AcceptingDataProcessor.serialize();
	}

	@Override
	public DataProcessor provide(String dataGroupName) {
		return new AcceptingDataProcessor();
	}

	@Override
	public DataProcessor provide(String dataGroupName, long crossValidationSelector, int foldSelector) {
		return new AcceptingDataProcessor();
	}

}

package org.rulelearn.experiments;

/**
 * Accepts data without any processing (returns the data passed as the input).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class AcceptingDataProcessor implements DataProcessor {

	@Override
	public Data process(Data data) {
		return data;
	}
	
	@Override
	public String toString() {
		return serialize();
	}

	@Override
	public Long getSeed() {
		return null; //seed not used
	}
	
	public static String serialize() {
		return AcceptingDataProcessor.class.getSimpleName();
	}

}

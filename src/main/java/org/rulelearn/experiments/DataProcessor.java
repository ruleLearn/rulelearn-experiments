package org.rulelearn.experiments;

/**
 * Processes data, using a seed.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface DataProcessor {
	
	Data process(Data data);
	Long getSeed();
}

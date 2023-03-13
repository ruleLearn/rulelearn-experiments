package org.rulelearn.experiments;

/**
 * Provides a fresh instance of configured {@link DataProcessor}, with a proper seed.
 *
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface DataProcessorProvider {

    public DataProcessor provide();
    public DataProcessor provide(String dataGroupName);
    public DataProcessor provide(String dataGroupName, long crossValidationSelector, int foldSelector);
    @Override
    public String toString(); 

}

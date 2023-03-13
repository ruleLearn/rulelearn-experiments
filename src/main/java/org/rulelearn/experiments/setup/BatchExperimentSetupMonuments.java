package org.rulelearn.experiments.setup;

import java.util.ArrayList;
import java.util.List;

import org.rulelearn.experiments.DataProcessorProvider;
import org.rulelearn.experiments.DataProvider;

/**
 * Batch experiment setup for monuments ('zabytki' in Polish) data set.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class BatchExperimentSetupMonuments extends BatchExperimentSetup {

	final protected String dataNameMonumentsNoMV = "zabytki";
	final protected String dataNameMonumentsNoMV_K9_K10 = "zabytki-K9-K10";
	//-----
	final protected String dataNameMonumentsNoMV01 = "zabytki01";
	final protected String dataNameMonumentsNoMV01_K9_K10 = "zabytki01-K9-K10";
	
	final protected String dataGroupMonuments = "zabytki";
	
	public BatchExperimentSetupMonuments(long[] seeds, int k, DataProcessorProvider dataProcessorProvider) {
		super(seeds, k, dataProcessorProvider);
	}
	
	@Override
	public List<DataProvider> getDataProviders() {
		if (dataProviders == null) {
			dataProviders = new ArrayList<DataProvider>();
			
			dataProviders.add(getDataProviderMonuments(dataNameMonumentsNoMV, dataGroupMonuments, seeds, k));
			dataProviders.add(getDataProviderMonuments_K9_K10(dataNameMonumentsNoMV_K9_K10, dataGroupMonuments, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			dataProviders.add(getDataProviderMonuments01(dataNameMonumentsNoMV01, dataGroupMonuments, seeds, k));
			dataProviders.add(getDataProviderMonuments01_K9_K10(dataNameMonumentsNoMV01_K9_K10, dataGroupMonuments, seeds, k));
		}
		
		return dataProviders;
	}
	
	abstract protected DataProvider getDataProviderMonuments(String dataSetName, String dataSetGroup, long[] seeds, int k);
	abstract protected DataProvider getDataProviderMonuments_K9_K10(String dataSetName, String dataSetGroup, long[] seeds, int k);
	abstract protected DataProvider getDataProviderMonuments01(String dataSetName, String dataSetGroup, long[] seeds, int k);
	abstract protected DataProvider getDataProviderMonuments01_K9_K10(String dataSetName, String dataSetGroup, long[] seeds, int k);
	
}

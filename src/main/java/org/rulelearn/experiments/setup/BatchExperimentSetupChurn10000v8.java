package org.rulelearn.experiments.setup;

import java.util.ArrayList;
import java.util.List;

import org.rulelearn.experiments.DataProcessor;
import org.rulelearn.experiments.DataProvider;

/**
 * Batch experiment setup for churn10000v8 data set.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class BatchExperimentSetupChurn10000v8 extends BatchExperimentSetup {
	
	final protected String dataNameChurn10000v8 = "bank-churn-10000-v8";
	//-----
//	final protected String dataNameChurn10000v8_0_05_mv2 = "bank-churn-10000-v8-0.05-mv2";
	final protected String dataNameChurn10000v8_0_05_mv15 = "bank-churn-10000-v8-0.05-mv1.5";
//	final protected String dataNameChurn10000v8_0_10_mv2 = "bank-churn-10000-v8-0.10-mv2";
	final protected String dataNameChurn10000v8_0_10_mv15 = "bank-churn-10000-v8-0.10-mv1.5";
//	final protected String dataNameChurn10000v8_0_15_mv2 = "bank-churn-10000-v8-0.15-mv2";
	final protected String dataNameChurn10000v8_0_15_mv15 = "bank-churn-10000-v8-0.15-mv1.5";
//	final protected String dataNameChurn10000v8_0_20_mv2 = "bank-churn-10000-v8-0.20-mv2";
	final protected String dataNameChurn10000v8_0_20_mv15 = "bank-churn-10000-v8-0.20-mv1.5";
//	final protected String dataNameChurn10000v8_0_25_mv2 = "bank-churn-10000-v8-0.25-mv2";
	final protected String dataNameChurn10000v8_0_25_mv15 = "bank-churn-10000-v8-0.25-mv1.5";
	
	public BatchExperimentSetupChurn10000v8(long[] seeds, int k, DataProcessor dataProcessor) {
		super(seeds, k, dataProcessor);
	}
	
	@Override
	public List<DataProvider> getDataProviders() {
		if (dataProviders == null) {
			dataProviders = new ArrayList<DataProvider>();
			
			dataProviders.add(getDataProviderChurn10000v8(dataNameChurn10000v8, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn10000v8_0_05_mv2(dataNameChurn10000v8_0_05_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn10000v8_0_05_mv15(dataNameChurn10000v8_0_05_mv15, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn10000v8_0_10_mv2(dataNameChurn10000v8_0_10_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn10000v8_0_10_mv15(dataNameChurn10000v8_0_10_mv15, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn10000v8_0_15_mv2(dataNameChurn10000v8_0_15_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn10000v8_0_15_mv15(dataNameChurn10000v8_0_15_mv15, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn10000v8_0_20_mv2(dataNameChurn10000v8_0_20_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn10000v8_0_20_mv15(dataNameChurn10000v8_0_20_mv15, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn10000v8_0_25_mv2(dataNameChurn10000v8_0_25_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn10000v8_0_25_mv15(dataNameChurn10000v8_0_25_mv15, seeds, k));
		}
		
		return dataProviders;
	}

	abstract protected DataProvider getDataProviderChurn10000v8(String dataSetName, long[] seeds, int k);
//	abstract protected DataProvider getDataProviderChurn10000v8_0_05_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn10000v8_0_05_mv15(String dataSetName, long[] seeds, int k);
//	abstract protected DataProvider getDataProviderChurn10000v8_0_10_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn10000v8_0_10_mv15(String dataSetName, long[] seeds, int k);
//	abstract protected DataProvider getDataProviderChurn10000v8_0_15_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn10000v8_0_15_mv15(String dataSetName, long[] seeds, int k);
//	abstract protected DataProvider getDataProviderChurn10000v8_0_20_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn10000v8_0_20_mv15(String dataSetName, long[] seeds, int k);
//	abstract protected DataProvider getDataProviderChurn10000v8_0_25_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn10000v8_0_25_mv15(String dataSetName, long[] seeds, int k);
	
}

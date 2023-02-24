package org.rulelearn.experiments.setup;

import java.util.ArrayList;
import java.util.List;

import org.rulelearn.experiments.AcceptingDataProcessor;
import org.rulelearn.experiments.DataProcessor;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.WEKAClassifierLearner;

import weka.classifiers.misc.OSDL;

/**
 * Batch experiment setup for churn4000v8 data set.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class BatchExperimentSetupChurn4000v8 extends BatchExperimentSetup {
	
	final protected String dataNameChurn4000v8 = "bank-churn-4000-v8";
	//-----
	final protected String dataNameChurn4000v8_0_05_mv2 = "bank-churn-4000-v8-0.05-mv2";
	final protected String dataNameChurn4000v8_0_05_mv15 = "bank-churn-4000-v8-0.05-mv1.5";
	final protected String dataNameChurn4000v8_0_10_mv2 = "bank-churn-4000-v8-0.10-mv2";
	final protected String dataNameChurn4000v8_0_10_mv15 = "bank-churn-4000-v8-0.10-mv1.5";
	final protected String dataNameChurn4000v8_0_15_mv2 = "bank-churn-4000-v8-0.15-mv2";
	final protected String dataNameChurn4000v8_0_15_mv15 = "bank-churn-4000-v8-0.15-mv1.5";
	final protected String dataNameChurn4000v8_0_20_mv2 = "bank-churn-4000-v8-0.20-mv2";
	final protected String dataNameChurn4000v8_0_20_mv15 = "bank-churn-4000-v8-0.20-mv1.5";
	final protected String dataNameChurn4000v8_0_25_mv2 = "bank-churn-4000-v8-0.25-mv2";
	final protected String dataNameChurn4000v8_0_25_mv15 = "bank-churn-4000-v8-0.25-mv1.5";
	
	public BatchExperimentSetupChurn4000v8(long[] seeds, int k) {
		super(seeds, k);
	}
	
	@Override
	public List<DataProvider> getDataProviders() {
		if (dataProviders == null) {
			dataProviders = new ArrayList<DataProvider>();
			
			dataProviders.add(getDataProviderChurn4000v8(dataNameChurn4000v8, seeds, k));
			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn4000v8_0_05_mv2(dataNameChurn4000v8_0_05_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn4000v8_0_05_mv15(dataNameChurn4000v8_0_05_mv15, seeds, k));
//			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn4000v8_0_10_mv2(dataNameChurn4000v8_0_10_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn4000v8_0_10_mv15(dataNameChurn4000v8_0_10_mv15, seeds, k));
//			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn4000v8_0_15_mv2(dataNameChurn4000v8_0_15_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn4000v8_0_15_mv15(dataNameChurn4000v8_0_15_mv15, seeds, k));
//			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn4000v8_0_20_mv2(dataNameChurn4000v8_0_20_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn4000v8_0_20_mv15(dataNameChurn4000v8_0_20_mv15, seeds, k));
//			/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//			dataProviders.add(getDataProviderChurn4000v8_0_25_mv2(dataNameChurn4000v8_0_25_mv2, seeds, k));
			dataProviders.add(getDataProviderChurn4000v8_0_25_mv15(dataNameChurn4000v8_0_25_mv15, seeds, k));
		}
		
		return dataProviders;
	}

	@Override
	public DataProcessor getDataProcessor() {
		return new AcceptingDataProcessor();
	}
	
	abstract protected DataProvider getDataProviderChurn4000v8(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_05_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_05_mv15(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_10_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_10_mv15(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_15_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_15_mv15(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_20_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_20_mv15(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_25_mv2(String dataSetName, long[] seeds, int k);
	abstract protected DataProvider getDataProviderChurn4000v8_0_25_mv15(String dataSetName, long[] seeds, int k);
	
}

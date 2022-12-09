/**
 * 
 */
package org.rulelearn.experiments;

import java.io.IOException;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableBuilder;
import org.rulelearn.data.InformationTableWithDecisionDistributions;

/**
 * Provides {@link VCDRSAData VC-DRSA data} concerning full information table, seeds for subsequent cross-validations, and number of folds in each cross-validation.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BasicDataProvider implements DataProvider {
	
	private abstract class Params {
		String metadataJSONFilePath;

		public Params(String metadataJSONFilePath) {
			this.metadataJSONFilePath = metadataJSONFilePath;
		}
		
		abstract InformationTableWithDecisionDistributions loadInformationTable() throws IOException;
	}
	
	private class JSONParams extends Params {
		String objectsJSONFilePath;
		
		public JSONParams(String metadataJSONFilePath, String objectsJSONFilePath) {
			super(metadataJSONFilePath);
			this.objectsJSONFilePath = objectsJSONFilePath;
		}

		@Override
		InformationTableWithDecisionDistributions loadInformationTable() throws IOException {
			InformationTable informationTable;
			
			try {
				informationTable = InformationTableBuilder.safelyBuildFromJSONFile(metadataJSONFilePath, objectsJSONFilePath);
			} catch (IOException e) {
				throw e;
			}
			
			return new InformationTableWithDecisionDistributions(informationTable);
		}
	}
	
	private class CSVParams extends Params {
		String objectsCSVFilePath;
		boolean header;
		char separator;
		
		public CSVParams(String metadataJSONFilePath, String objectsCSVFilePath, boolean header, char separator) {
			super(metadataJSONFilePath);
			this.objectsCSVFilePath = objectsCSVFilePath;
			this.header = header;
			this.separator = separator;
		}
		
		@Override
		InformationTableWithDecisionDistributions loadInformationTable() throws IOException {
			InformationTable informationTable;
			
			try {
				informationTable = InformationTableBuilder.safelyBuildFromCSVFile(metadataJSONFilePath, objectsCSVFilePath, header, separator);
			} catch (IOException e) {
				throw e;
			}
			
			return new InformationTableWithDecisionDistributions(informationTable);
		}
	}
	
	InformationTableWithDecisionDistributions informationTable = null;
	Params params;
	String dataName;
	long[] seeds;
	int numberOfFolds;
	
	boolean done = false;
	
	public BasicDataProvider(String metadataJSONFilePath, String objectsJSONFilePath,
			String dataName,
			long[] seeds,
			int numberOfFolds) {
		
		this.params = new JSONParams(metadataJSONFilePath, objectsJSONFilePath);
		this.dataName = dataName;
		this.seeds = seeds;
		this.numberOfFolds = numberOfFolds;
	}
	
	public BasicDataProvider(String metadataJSONFilePath, String objectsCSVFilePath, boolean header, char separator,
			String dataName,
			long[] seeds,
			int numberOfFolds) {

		this.params = new CSVParams(metadataJSONFilePath, objectsCSVFilePath, header, separator);
		this.dataName = dataName;
		this.seeds = seeds;
		this.numberOfFolds = numberOfFolds;
	}
	
	/**
	 * @param crossValidationNumber number of cross-validation for which data should be provided
	 * 
	 * @throws UnsupportedOperationException if this provider has already {@link #done() done} his job.
	 * @throws InvalidValueException if this method is called for the first time, and information table cannot be loaded from disk for the parameters given in constructor
	 */
	@Override
	public Data provide(int crossValidationNumber) {
		if (!done) {
			if (informationTable == null) {
				try {
					informationTable = params.loadInformationTable();
				} catch (IOException e) {
					throw new InvalidValueException("Could not load information table from disk.");
				}
			}
			return new Data(informationTable, dataName, seeds[crossValidationNumber]);
		} else {
			throw new UnsupportedOperationException("Data provider has already done his job.");
		}
	}
	
	/**
	 * @throws UnsupportedOperationException if this provider has already {@link #done() done} his job.
	 * @throws InvalidValueException if this method is called for the first time, and information table cannot be loaded from disk for the parameters given in constructor
	 */
	@Override
	public Data provideOriginalData() {
		if (!done) {
			if (informationTable == null) {
				try {
					informationTable = params.loadInformationTable();
				} catch (IOException e) {
					throw new InvalidValueException("Could not load information table from disk.");
				}
			}
			return new Data(informationTable, dataName);
		} else {
			throw new UnsupportedOperationException("Data provider has already done his job.");
		}
	}

	/**
	 * Gets seeds for subsequent cross-validations.
	 * 
	 * @throws UnsupportedOperationException if this provider has already {@link #done() done} his job.
	 */
	@Override
	public long[] getSeeds() {
		if (!done) {
			return seeds;
		} else {
			throw new UnsupportedOperationException("Data provider has already done his job.");
		}
	}

	/**
	 * Gets number of folds in each cross-validation.
	 * 
	 * @throws UnsupportedOperationException if this provider has already {@link #done() done} his job.
	 */
	@Override
	public int getNumberOfFolds() {
		if (!done) {
			return numberOfFolds;
		} else {
			throw new UnsupportedOperationException("Data provider has already done his job.");
		}
	}

	/**
	 * @throws UnsupportedOperationException if this provider has already {@link #done() done} his job.
	 */
	@Override
	public String getDataName() {
		return dataName;
	}
	
	@Override
	public void done() {
		this.informationTable = null;
		this.seeds = null;
		this.params = null;
		done = true;
	}
	
}

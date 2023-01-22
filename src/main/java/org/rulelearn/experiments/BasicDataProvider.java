/**
 * 
 */
package org.rulelearn.experiments;

import static org.rulelearn.core.Precondition.notNull;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Attribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableBuilder;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.ObjectParseException;
import org.rulelearn.data.csv.ObjectParser;
import org.rulelearn.data.json.AttributeDeserializer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

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
		abstract InformationTable previewInformationTable() throws IOException; //loads information table but does not calculate its decision distributions
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
				informationTable = safelyBuildFromJSONFileStream(metadataJSONFilePath, objectsJSONFilePath);
			} catch (IOException e) {
				throw e;
			}
			
			long start = System.currentTimeMillis();
			InformationTableWithDecisionDistributions itwd = new InformationTableWithDecisionDistributions(informationTable);
			System.out.println("Information table transformation time: "+(System.currentTimeMillis()-start));
			
			return itwd;
		}
		
		@Override
		InformationTable previewInformationTable() throws IOException {
			InformationTable informationTable;
			
			try {
				informationTable = safelyBuildFromJSONFileStream(metadataJSONFilePath, objectsJSONFilePath);
			} catch (IOException e) {
				throw e;
			}
			
			return informationTable;
		}
		
		public InformationTable safelyBuildFromJSONFileStream(String pathToJSONAttributeFile, String pathToJSONObjectFile) throws IOException {
			notNull(pathToJSONAttributeFile, "Path to JSON file with attributes is null.");
			notNull(pathToJSONObjectFile, "Path to JSON file with objects is null.");
			
			Attribute [] attributes = null;
			List<String[]> objects = null;
			InformationTableBuilder informationTableBuilder = null;
			InformationTable informationTable = null;
			
			// load attributes
			GsonBuilder gsonBuilder = new GsonBuilder();
			gsonBuilder.registerTypeAdapter(Attribute.class, new AttributeDeserializer());
			Gson gson = gsonBuilder.setPrettyPrinting().create();
			
			try (
				InputStream inputAttributeStream = this.getClass().getClassLoader().getResourceAsStream(pathToJSONAttributeFile);
				InputStreamReader inputAttributeStreamReader = new InputStreamReader(inputAttributeStream);
				JsonReader jsonAttributesReader = new JsonReader(inputAttributeStreamReader)) {
				
				attributes = gson.fromJson(jsonAttributesReader, Attribute[].class);
				
				// load objects
				JsonElement json = null;
				try (
					InputStream inputObjectStream = this.getClass().getClassLoader().getResourceAsStream(pathToJSONObjectFile);
					InputStreamReader inputObjectStreamReader = new InputStreamReader(inputObjectStream);
					JsonReader jsonObjectsReader = new JsonReader(inputObjectStreamReader)) {
					
					json = JsonParser.parseReader(jsonObjectsReader);
				}
				org.rulelearn.data.json.ObjectBuilder ob = new org.rulelearn.data.json.ObjectBuilder.Builder(attributes).build();
				objects = ob.getObjects(json);
				
				// construct information table builder
				if (attributes != null) {
					informationTableBuilder = new InformationTableBuilder(attributes, new String[] {org.rulelearn.data.json.ObjectBuilder.DEFAULT_MISSING_VALUE_STRING});
					if (objects != null) {
						for (int i = 0; i < objects.size(); i++) {
							try {
								informationTableBuilder.addObject(objects.get(i)); //uses volatile cache
							} catch (ObjectParseException exception) {
								throw new ObjectParseException(new StringBuilder("Error while parsing object no. ").append(i+1).append(" from JSON. ").append(exception.toString()).toString()); //if exception was thrown, re-throw it
							}
						}
					}
				}
			}
			
			// build information table
			if (informationTableBuilder != null) {
				//clear volatile caches of all used evaluation field caching factories
				informationTableBuilder.clearVolatileCaches();
				informationTable = informationTableBuilder.build();
			}

			return informationTable;
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
				informationTable = safelyBuildFromCSVFileStream(metadataJSONFilePath, objectsCSVFilePath, header, separator);
			} catch (IOException e) {
				throw e;
			}
			
			long start = System.currentTimeMillis();
			InformationTableWithDecisionDistributions itwd = new InformationTableWithDecisionDistributions(informationTable);
			System.out.println("Information table transformation time: "+(System.currentTimeMillis()-start));
			
			return itwd;
		}
		
		@Override
		InformationTable previewInformationTable() throws IOException {
			InformationTable informationTable;
			
			try {
				informationTable = safelyBuildFromCSVFileStream(metadataJSONFilePath, objectsCSVFilePath, header, separator);
			} catch (IOException e) {
				throw e;
			}
			
			return informationTable;
		}
		
		//@throws ObjectParseException if the number of values specified for an object in CSV input exceeds the number of attributes
		public InformationTable safelyBuildFromCSVFileStream(String pathToJSONAttributeFile, String pathToCSVObjectFile, boolean header, char separator) throws IOException {
			notNull(pathToJSONAttributeFile, "Path to JSON file with attributes is null.");
			notNull(pathToCSVObjectFile, "Path to CSV file with objects is null.");
			
			Attribute[] attributes = null;
			InformationTable informationTable = null;
			
			//load attributes
			GsonBuilder gsonBuilder = new GsonBuilder();
			gsonBuilder.registerTypeAdapter(Attribute.class, new AttributeDeserializer());
			Gson gson = gsonBuilder.setPrettyPrinting().create();
			
			try (
				InputStream inputAttributeStream = this.getClass().getClassLoader().getResourceAsStream(pathToJSONAttributeFile);
				InputStreamReader inputAttributeStreamReader = new InputStreamReader(inputAttributeStream);
				JsonReader jsonAttributesReader = new JsonReader(inputAttributeStreamReader)) {
					
				attributes = gson.fromJson(jsonAttributesReader, Attribute[].class);
			
				//construct information table using object parser!
				if (attributes != null) {
					ObjectParser objectParser = new ObjectParser.Builder(attributes).header(header).separator(separator).build();
					
					try (
						InputStream inputObjectStream = this.getClass().getClassLoader().getResourceAsStream(pathToCSVObjectFile);
						InputStreamReader inputObjectStreamReader = new InputStreamReader(inputObjectStream)) {
					
						informationTable = objectParser.parseObjects(inputObjectStreamReader);
					}
				}
			}
			
			return informationTable;
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
	
	@Override
	public Data previewOriginalData() { //returns Data but does not store information table neither calculates its decision distributions
		InformationTable informationTable = null; //no decision distributions - loads faster!
		try {
			informationTable = params.previewInformationTable();
		} catch (IOException e) {
			throw new InvalidValueException("Could not load information table from disk.");
		}
		
		return new Data(informationTable, dataName);
	}

	/**
	 * Gets seeds for subsequent cross-validations. It is possible to obtain seeds even if this provide is {@link #done()}.
	 */
	@Override
	public long[] getSeeds() {
		return seeds;
//		if (!done) {
//			return seeds;
//		} else {
//			throw new UnsupportedOperationException("Data provider has already done his job.");
//		}
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
		//this.seeds = null;
		this.params = null;
		done = true;
	}
	
	@Override
	public void reset() {
		this.informationTable = null;
	}
	
}

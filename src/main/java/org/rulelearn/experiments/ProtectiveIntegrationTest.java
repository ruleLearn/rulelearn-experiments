/**
 * Copyright (C) Jerzy Błaszczyński, Marcin Szeląg
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.rulelearn.experiments;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;

import org.rulelearn.data.Attribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableBuilder;
import org.rulelearn.data.json.AttributeParser;
import org.rulelearn.rules.RuleSet;
import org.rulelearn.rules.ruleml.RuleParser;

/**
 * Tests different operations on PROTECTIVE data (like, reading objects into information table, from JSON and CSV, and reading rules from RuleML followed by printing them to system console as simple text.
 *
 * @author Jerzy Błaszczyński (<a href="mailto:jurek.blaszczynski@cs.put.poznan.pl">jurek.blaszczynski@cs.put.poznan.pl</a>)
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ProtectiveIntegrationTest {

	/**
	 * Application entry point.
	 * 
	 * @param args input arguments of this application (ignored)
	 */
	public static void main(String[] args) {
		readInformationTable(true);
		System.out.println();
		readInformationTable(false);
		System.out.println();
		transformRules();
	}
	
	/**
	 * Reads information table from JSON or CSV file.
	 * 
	 * @param fromJSON flag indicating type of file with objects
	 */
	private static void readInformationTable(boolean fromJSON) { //read objects from JSON file if true, otherwise from CSV file
		InformationTable informationTable = null;
		try {
			if (fromJSON) {
				informationTable = InformationTableBuilder.safelyBuildFromJSONFile("src/test/resources/data/json-metadata/prioritisation-no-rank.json", "src/test/resources/data/json-objects/LearningSet_2604v1.json");
			} else {
				informationTable = InformationTableBuilder.safelyBuildFromCSVFile("src/test/resources/data/json-metadata/prioritisation-no-rank.json", "src/test/resources/data/csv/LearningSet_2604v1.csv", false, ',');
			}
			
		}
		catch (FileNotFoundException ex) {
			System.out.println(ex);
			return;
		}
		catch (IOException ex) {
			System.out.println(ex);
			return;
		}
		
		if (informationTable != null) {
			System.out.println("Information table read from file.");
			System.out.println("# objects: "+informationTable.getNumberOfObjects());
		} else {
			System.out.println("Error reading information table from JSON file.");
		}
	}
	
	/**
	 * Transforms rules from RuleML file (xml) to text, sorting conditions.
	 */
	private static void transformRules() {
		Attribute [] attributes = null;
		AttributeParser attributeParser = new AttributeParser();
		try (FileReader attributeReader = new FileReader("src/test/resources/data/json-metadata/prioritisation-no-rank.json")) {
			attributes = attributeParser.parseAttributes(attributeReader);
			if (attributes != null) {
				Map<Integer, RuleSet> rules = null;
				RuleParser ruleParser = new RuleParser(attributes);
				try (FileInputStream fileRulesStream = new FileInputStream("src/test/resources/data/ruleml/rules6.xml")) {
					rules = ruleParser.parseRules(fileRulesStream);
					if (rules != null) {
						RuleSet ruleSet = rules.get(1);
						System.out.println(ruleSet.size() + " rules read from file:");
						
						for (int i = 0; i < ruleSet.size(); i++) {
							System.out.println(ruleSet.getRule(i).toString(true)); //true <==> sort rules' conditions
						}
					}
					else {
						System.out.println("Unable to load RuleML file.");
						return;
					}
				}
				catch (FileNotFoundException ex) {
					System.out.println(ex.toString());
					return;
				}
			}
			else {
				System.out.println("Unable to load JSON file with meta-data.");
				return;
			}
		}
		catch (FileNotFoundException ex) {
			System.out.println(ex.toString());
			return;
		}
		catch (IOException ex) {
			System.out.println(ex.toString());
			return;
		}
	}

}

/**
 * 
 */
package org.rulelearn.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.core.Precondition;
import org.rulelearn.data.Attribute;
import org.rulelearn.data.AttributePreferenceType;
import org.rulelearn.data.AttributeType;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableBuilder;
import org.rulelearn.types.EnumerationField;
import org.rulelearn.types.IntegerField;
import org.rulelearn.types.RealField;
import org.rulelearn.types.UnknownSimpleField;
import org.rulelearn.types.UnknownSimpleFieldMV15;
import org.rulelearn.types.UnknownSimpleFieldMV2;

import keel.Dataset.InstanceAttributes;
import keel.Dataset.InstanceSet;

/**
 * Converts from ruleLearn's {@link InformationTable} to KEEL's {@link InstanceSet}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class InformationTable2InstanceSet {
	
	public static InstanceSet convert(InformationTable informationTable, String relationName, AttributeRanges attributeRanges) {
		Precondition.notNull(informationTable, "Information table to be converted to instances is null.");
		
		Attribute attribute;
		EvaluationAttribute evaluationAttribute;
		int numberOfAttributes = informationTable.getNumberOfAttributes();
		int numberOfObjects = informationTable.getNumberOfObjects();
		
		keel.Dataset.Attribute keelDecisionAttribute = null;
		keel.Dataset.Attribute keelAttribute = null;
		
		ArrayList<keel.Dataset.Attribute> keelAttributesList = new ArrayList<keel.Dataset.Attribute>(numberOfAttributes); //contains KEEL attributes, possibly with spare space
		
		int[] keelAttributeIndex2RuleLearnAttributeIndex = new int[numberOfAttributes]; //encodes mapping between KEEL's attribute index and ruleLearn's attribute index, possibly with spare space (if not all rL attributes will be converted)
		
		int keelAttributeIndex = 0;
		int decisionAttributeIndex = -1;
		String keelDecisionAttributeHeaderLine = null;
		
		String attributeName;
		boolean skipAttribute = false;
		boolean suffixGain;
		boolean suffixCost;
		String keelAttributeDomain;
		String keelAttributeRange; //concerns only integer and real attributes, otherwise is an empty string
		
		StringBuilder headerBuilder = new StringBuilder(256);
		headerBuilder.append("@relation ").append(relationName).append(System.lineSeparator());
		
		for (int j = 0; j < numberOfAttributes; j++) {
			if (skipAttribute) { //second attribute from a pair of ruleLearn's attributes with imposed preference orders
				skipAttribute = false;
				continue;
			}
			
			attribute = informationTable.getAttribute(j);
			
			if (attribute instanceof EvaluationAttribute) {
				evaluationAttribute = (EvaluationAttribute)attribute;
				
				if (evaluationAttribute.isActive() && evaluationAttribute.getType() != AttributeType.DESCRIPTION) { //active condition or decision attribute
					attributeName = attribute.getName();
					
					suffixGain = false;
					suffixCost = false;
					//_g followed by _c attribute or _c followed by _g attribute (pair of attributes resulting from preference order imposition)
					if ((suffixGain = attributeName.endsWith(InformationTable.attributeNameSuffixGain) &&
							j + 1 < numberOfAttributes &&
							informationTable.getAttribute(j + 1).getName().endsWith(InformationTable.attributeNameSuffixCost)) ||
						(suffixCost = attributeName.endsWith(InformationTable.attributeNameSuffixCost) &&
							j + 1 < numberOfAttributes &&
							informationTable.getAttribute(j + 1).getName().endsWith(InformationTable.attributeNameSuffixGain))
						) {
						skipAttribute = true; //skip next attribute
						//do not encode attribute preference type in attribute's name (!) (as merged attribute has NONE preference type)
						//and strip gain/cost suffix
						if (suffixGain) {
							attributeName = attributeName.substring(0, attributeName.lastIndexOf(InformationTable.attributeNameSuffixGain));
						} else if (suffixCost) { //this has to be true if suffixGain == false
							attributeName = attributeName.substring(0, attributeName.lastIndexOf(InformationTable.attributeNameSuffixCost));
						}
						attributeName += Instances2InformationTable.mergedAttributeIndicator; //encode information that produced KEEL's attribute is a merged one! //TODO
					} else {
						//encode preference type in attribute's name
						if (evaluationAttribute.getPreferenceType() == AttributePreferenceType.GAIN) {
							attributeName += Instances2InformationTable.gainAttributeIndicator; //TODO
						} else if (evaluationAttribute.getPreferenceType() == AttributePreferenceType.COST) {
							attributeName += Instances2InformationTable.costAttributeIndicator; //TODO
						}
					}
					
					//encode missing value type in attribute's name
					if (evaluationAttribute.getMissingValueType() instanceof UnknownSimpleFieldMV2) {
						attributeName += Instances2InformationTable.mv2MissingValueTypeIndicator;
					} else if (evaluationAttribute.getMissingValueType() instanceof UnknownSimpleFieldMV15) {
						attributeName += Instances2InformationTable.mv15MissingValueTypeIndicator;
					}

					keelAttribute = new keel.Dataset.Attribute();
					
					if (evaluationAttribute.getValueType() instanceof EnumerationField) {
						keelAttribute.setType(keel.Dataset.Attribute.NOMINAL);
						StringBuilder keelEnumDomainBuilder = new StringBuilder(128);
						boolean first = true;
						keelEnumDomainBuilder.append("{");
						for (String element : ((EnumerationField)evaluationAttribute.getValueType()).getElementList().getElements()) {
							keelAttribute.addNominalValue(element);
							if (first) {
								first = false;
							} else {
								keelEnumDomainBuilder.append(", ");
							}
							keelEnumDomainBuilder.append(element);
						}
						keelEnumDomainBuilder.append("}");
						keelAttributeDomain = keelEnumDomainBuilder.toString();
						keelAttributeRange = "";
						keelAttribute.setFixedBounds(true);
					} else if (evaluationAttribute.getValueType() instanceof RealField) {
						keelAttribute.setType(keel.Dataset.Attribute.REAL);
						keelAttribute.setBounds(attributeRanges.getRange(j).getMin(), attributeRanges.getRange(j).getMax());
						keelAttributeDomain = "real";
						keelAttributeRange = (new StringBuilder(32)).append(" [").append(keelAttribute.getMinAttribute()).append(", ").append(keelAttribute.getMaxAttribute()).append("]").toString();
						//at.setBounds ( min, max );
					} else if (evaluationAttribute.getValueType() instanceof IntegerField) {
						//attributeName += Instances2InformationTable.integerAttributeIndicator;
						keelAttribute.setType(keel.Dataset.Attribute.INTEGER);
						keelAttribute.setBounds(attributeRanges.getRange(j).getMin(), attributeRanges.getRange(j).getMax());
						keelAttributeDomain = "integer";
						keelAttributeRange = (new StringBuilder(32)).append(" [").append(keelAttribute.getMinAttribute()).append(", ").append(keelAttribute.getMaxAttribute()).append("]").toString();
						//at.setBounds ( min, max );
					}
					else {
						throw new UnsupportedOperationException("Not supported attribute's value type for attribute "+evaluationAttribute.getName()+".");
					}
					
					keelAttribute.setName(attributeName);
					String keelAttributeHeaderLine =
							(new StringBuilder(64)).append("@attribute ").append(attributeName).append(" ").append(keelAttributeDomain).append(keelAttributeRange).toString();
					
					if (evaluationAttribute.getType() == AttributeType.DECISION) {
						keelAttribute.setDirectionAttribute(keel.Dataset.Attribute.OUTPUT);
						
						if (keelDecisionAttribute == null) {
							keelDecisionAttribute = keelAttribute; //remember first decision attribute (to add it as the last attribute)
							decisionAttributeIndex = j;
							
							keelDecisionAttributeHeaderLine = keelAttributeHeaderLine;
						} else {
							throw new InvalidValueException("More than one decision attribute found in information table.");
						}
					} else { //condition attribute
						keelAttribute.setDirectionAttribute(keel.Dataset.Attribute.INPUT);
						
						headerBuilder.append(keelAttributeHeaderLine).append(System.lineSeparator());
						
						keelAttributesList.add(keelAttribute);
						keelAttributeIndex2RuleLearnAttributeIndex[keelAttributeIndex++] = j;
					}
				}
			}
		}
		
		if (keelDecisionAttribute != null) {
			keelAttributesList.add(keelDecisionAttribute); //add decision attribute as the last one
			keelAttributeIndex2RuleLearnAttributeIndex[keelAttributeIndex++] = decisionAttributeIndex;
			
			headerBuilder.append(keelDecisionAttributeHeaderLine);
		} else {
			throw new InvalidValueException("No decision attribute found in information table.");
		}
		
		//Create KEEL data set
		InstanceAttributes instanceAttributes = new InstanceAttributes();
		instanceAttributes.setRelationName(relationName);
		keelAttributesList.forEach(aKeelAttribute -> instanceAttributes.addAttribute(aKeelAttribute));
		
		keel.Dataset.Instance[] instances = new keel.Dataset.Instance[numberOfObjects];
		int numberOfKeelAttributes = keelAttributesList.size();
		
		for (int i = 0; i < numberOfObjects; i++) {
			double[] values = new double[instanceAttributes.getNumAttributes()];
			
			for (int j = 0; j < numberOfKeelAttributes; j++) {
				if (informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j]) instanceof RealField) {
					values[j] = ((RealField)informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j])).getValue();
				} else if (informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j]) instanceof IntegerField) {
					values[j] = ((IntegerField)informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j])).getValue();
				} else if (informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j]) instanceof EnumerationField) {
					values[j] = ((EnumerationField)informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j])).getValue(); //int -> double
				} else if (informationTable.getField(i, keelAttributeIndex2RuleLearnAttributeIndex[j]) instanceof UnknownSimpleField) { //handles missing value
					values[j] = Double.NaN;
				}
			}
			
			instances[i] = new keel.Dataset.Instance(values, instanceAttributes);
		} //for
		
		InstanceSet instanceSet = new InstanceSet(instanceAttributes, instances, headerBuilder.toString());
		
		return instanceSet;
	}
	
	public static void main(String[] args) {
		InformationTable informationTable;
		
		try {
			informationTable = InformationTableBuilder.safelyBuildFromJSONFile(
					"src/main/resources/data/json-metadata/bank-churn-4000-v8_0.25_mv1.5_mv2 metadata.json",
					"src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.json");
//			informationTable = InformationTableBuilder.safelyBuildFromCSVFile(
//					"src/main/resources/data/json-metadata/OLM/bank-churn-4000-v8-processed metadata.json",
//					"src/main/resources/data/csv/OLM/bank-churn-4000-v8-processed data.csv",
//					true, ';');
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		InstanceSet instanceSet = convert(informationTable, "churn-4000-v8_0.25", new AttributeRanges(informationTable));
		//InstanceSet instanceSet = convert(informationTable, "churn-4000-v8-processed");
		
		try (PrintWriter printWriter = new PrintWriter(new File("src/main/resources/data/dat/bank-churn-4000-v8_0.25 data.dat")) ) {
			//PrintWriter printWriter = new PrintWriter(new File("src/main/resources/data/dat/bank-churn-4000-v8-processed data.dat")) ) {
			//instanceSet.print(printWriter);
			instanceSet.printAsOriginal(printWriter, 3);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}
}

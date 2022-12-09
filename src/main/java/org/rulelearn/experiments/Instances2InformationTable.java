/**
 * 
 */
package org.rulelearn.experiments;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import org.rulelearn.core.InvalidTypeException;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.AttributePreferenceType;
import org.rulelearn.data.AttributeType;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableBuilder;
import org.rulelearn.data.json.InformationTableWriter;
import org.rulelearn.types.ElementList;
import org.rulelearn.types.EnumerationField;
import org.rulelearn.types.EnumerationFieldFactory;
import org.rulelearn.types.EvaluationField;
import org.rulelearn.types.Field;
import org.rulelearn.types.IntegerField;
import org.rulelearn.types.IntegerFieldFactory;
import org.rulelearn.types.RealField;
import org.rulelearn.types.RealFieldFactory;
import org.rulelearn.types.UnknownSimpleField;
import org.rulelearn.types.UnknownSimpleFieldMV15;
import org.rulelearn.types.UnknownSimpleFieldMV2;

import weka.core.Attribute;
import weka.core.Instances;

/**
 * Converts WEKA's {@link Instances} to ruleLearn's {@link InformationTable}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class Instances2InformationTable {
	
	static final String integerAttributeIndicator = "[i]"; //denotes integer numeric attribute (by default numeric resolves to real attribute) 
	static final String gainAttributeIndicator = "[g]";
	static final String costAttributeIndicator = "[c]";
	static final AttributePreferenceType defaultAttributePreferenceType = AttributePreferenceType.NONE;
	static final String mv2MissingValueTypeIndicator = "[mv2]";
	static final String mv15MissingValueTypeIndicator = "[mv1.5]";
	static final UnknownSimpleField defaultMissingValueType = UnknownSimpleFieldMV2.getInstance();
	static final String mergedAttributeIndicator = "[m]";

	public static InformationTable convert(Instances instances) {
		int numAttributes = instances.numAttributes();
		int numInstances = instances.numInstances();
		Attribute wekaAttribute;
		List<EvaluationAttribute> rLAttributesList = new ArrayList<EvaluationAttribute>();
		boolean isInteger;
		AttributeType attributeType;
		AttributePreferenceType attributePreferenceType;
		UnknownSimpleField missingValueType;
		Enumeration<Object> nominalValuesEnumeration;
		List<String> nominalValues;
		String attributeName;
		String attributeNameSuffix;
		
		//Encodes mapping between ruleLearn's attribute index and WEKA's attribute index; it is not necessarily 1:1,
		//as WEKA'a attributes with [m] indicator are converted to a pair of ruleLearn's attributes,
		//and thus, two ruleLearn's attributes may correspond to one WEKA'a attribute.
		List<Integer>ruleLearnAttributeIndex2WekaAttributeIndex = new ArrayList<Integer>(); //usage: int wekaAttributeIndex = ruleLearnAttributeIndex2WekaAttributeIndex.get(ruleLearnAttributeIndex)
		
		int numberOfCorrespondingRuleLearnAttributes;
		
		for (int j = 0; j < numAttributes; j++) { //go over WEKA's attributes
			wekaAttribute = instances.attribute(j);
			
			if (wekaAttribute.name().contains(mergedAttributeIndicator)) { //attribute that should be translated to two attributes - one of gain type, and one of cost type
				if (wekaAttribute.name().contains(gainAttributeIndicator) || wekaAttribute.name().contains(costAttributeIndicator)) {
					throw new InvalidValueException("Merged attribute should have preference type NONE.");
				}
				if (j == numAttributes - 1) { //last attribute (decision one)
					throw new InvalidValueException("Decision attribute should not be a merged attribute.");
				}
				numberOfCorrespondingRuleLearnAttributes = 2;
			} else {
				numberOfCorrespondingRuleLearnAttributes = 1;
			}
			
			for (int i = 0; i < numberOfCorrespondingRuleLearnAttributes; i++) {
				if (numberOfCorrespondingRuleLearnAttributes == 1) {
					attributeNameSuffix = "";
				} else {
					if (i == 0) {
						attributeNameSuffix = InformationTable.attributeNameSuffixGain;
					} else { //i == 1
						attributeNameSuffix = InformationTable.attributeNameSuffixCost;
					}
				}
			
				attributeType = (j < numAttributes - 1) ? AttributeType.CONDITION : AttributeType.DECISION; //last attribute is a decision one
				
				if (numberOfCorrespondingRuleLearnAttributes == 1) {
					if (wekaAttribute.name().contains(gainAttributeIndicator)) {
						attributePreferenceType = AttributePreferenceType.GAIN;
					} else if (wekaAttribute.name().contains(costAttributeIndicator)) {
						attributePreferenceType = AttributePreferenceType.COST;
					} else {
						attributePreferenceType = defaultAttributePreferenceType;
					}
				} else {
					if (i == 0) {
						attributePreferenceType = AttributePreferenceType.GAIN;
					} else { //i == 1
						attributePreferenceType = AttributePreferenceType.COST;
					}
				}
				
				if (wekaAttribute.name().contains(mv2MissingValueTypeIndicator)) {
					missingValueType = UnknownSimpleFieldMV2.getInstance();
				} else if (wekaAttribute.name().contains(mv15MissingValueTypeIndicator)) {
					missingValueType = UnknownSimpleFieldMV15.getInstance();
				} else {
					missingValueType = defaultMissingValueType;
				}
				
				attributeName = wekaAttribute.name().replace(integerAttributeIndicator, "").replace(gainAttributeIndicator, "").replace(costAttributeIndicator, "")
						.replace(mv2MissingValueTypeIndicator, "").replace(mv15MissingValueTypeIndicator, "").replace(mergedAttributeIndicator, "") + attributeNameSuffix;
				
				switch (wekaAttribute.type()) {
				case Attribute.NUMERIC:
					isInteger = wekaAttribute.name().contains(integerAttributeIndicator);
					
					if (isInteger) { //assume IntegerField
						rLAttributesList.add(new EvaluationAttribute(
								attributeName, true, attributeType,
								IntegerFieldFactory.getInstance().create(0, attributePreferenceType),
								missingValueType, attributePreferenceType));
					} else { //assume RealField
						rLAttributesList.add(new EvaluationAttribute(
								attributeName, true, attributeType,
								RealFieldFactory.getInstance().create(0.0, attributePreferenceType),
								missingValueType, attributePreferenceType));
					}
					break;
				case Attribute.NOMINAL: //assume EnumerationField
					nominalValuesEnumeration = wekaAttribute.enumerateValues();
					nominalValues = new ArrayList<>();
					
					while (nominalValuesEnumeration.hasMoreElements()) {
						nominalValues.add((String)nominalValuesEnumeration.nextElement());
					}
					
					try {
						rLAttributesList.add(new EvaluationAttribute(
								attributeName, true, attributeType,
								EnumerationFieldFactory.getInstance().create(new ElementList(nominalValues.toArray(new String[0])), 0, attributePreferenceType),
								missingValueType, attributePreferenceType));
					} catch (NoSuchAlgorithmException e) {
						e.printStackTrace();
						return null; //this should not happen
					}
					break;
				default:
					throw new InvalidTypeException("Unsupported WEKA's attribute type.");
				}
				
				ruleLearnAttributeIndex2WekaAttributeIndex.add(j);
			} //for
		}//for
		
		EvaluationAttribute[] rLAttributes = rLAttributesList.toArray(new EvaluationAttribute[0]);
		int numRuleLearnAttributes = rLAttributes.length;
		
		EvaluationField[] fields;
		List<Field[]> listOfFields = new ArrayList<Field[]>();
		double value; //in WEKA numeric and nominal values are internally stored as doubles
		
		for (int i = 0; i < numInstances; i++) {
			fields = new EvaluationField[numRuleLearnAttributes];
			for (int j = 0; j < numRuleLearnAttributes; j++) { //go over ruleLearn's attributes
				if (instances.instance(i).isMissing(ruleLearnAttributeIndex2WekaAttributeIndex.get(j))) { //handles missing value
					fields[j] = rLAttributes[j].getMissingValueType();
				} else {
					value = instances.instance(i).value(ruleLearnAttributeIndex2WekaAttributeIndex.get(j));
					if (rLAttributes[j].getValueType() instanceof IntegerField) {
						fields[j] = IntegerFieldFactory.getInstance().create((int)value, rLAttributes[j].getPreferenceType());
					} else if (rLAttributes[j].getValueType() instanceof RealField) {
						fields[j] = RealFieldFactory.getInstance().create(value, rLAttributes[j].getPreferenceType());
					} else if (rLAttributes[j].getValueType() instanceof EnumerationField) {
						fields[j] = EnumerationFieldFactory.getInstance().create(((EnumerationField)rLAttributes[j].getValueType()).getElementList(), (int)value, rLAttributes[j].getPreferenceType());
					}
				}
			}
			listOfFields.add(fields);
		}
		
		return new InformationTable(rLAttributes, listOfFields);
	}
	
	public static void main(String[] args) {
		InformationTable originalInformationTable;
		
		try {
			originalInformationTable = InformationTableBuilder.safelyBuildFromJSONFile(
					"src/main/resources/data/json-metadata/bank-churn-4000-v8_0.25_mv1.5_mv2 metadata.json",
					"src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.json");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		Instances instances = InformationTable2Instances.convert(originalInformationTable, "churn-4000-v8_0.25");
		
		InformationTable convertedInformationTable = convert(instances);
		
		InformationTableWriter informationTableWriter = new InformationTableWriter(true);
		
		try (FileWriter fileWriter = new FileWriter("src/main/resources/data/json-metadata/bank-churn-4000-v8_0.25_mv1.5_mv2-retrieved metadata.json")) {
			informationTableWriter.writeAttributes(convertedInformationTable, fileWriter);
		}
		catch (IOException exception) {
			exception.printStackTrace();
		}
		
		try (FileWriter fileWriter = new FileWriter("src/main/resources/data/json-objects/bank-churn-4000-v8_0.25-retrieved data.json")) {
			informationTableWriter.writeObjects(convertedInformationTable, fileWriter);
		}
		catch (IOException exception) {
			exception.printStackTrace();
		}
	}
	
}

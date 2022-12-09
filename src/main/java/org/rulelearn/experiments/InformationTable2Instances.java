/**
 * 
 */
package org.rulelearn.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

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

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Converts ruleLearn's {@link InformationTable} to WEKA's {@link Instances}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class InformationTable2Instances {
	
	/**
	 * Converts given {@link InformationTable information table} to WEKA's {@link Instances}.
	 * Takes into account only active evaluation attributes whose {@link EvaluationAttribute#getType() type} is either {@link AttributeType#CONDITION} or {@link AttributeType#DECISION}.
	 * 
	 * @param informationTable input information table, to be converted
	 * @param relationName name of constructed instances
	 * @return resulting WEKA's instances
	 * 
	 * @throws UnsupportedOperationException if type any of the active condition attributes from given information table is none of {@link EnumerationField}, {@link RealField}, or {@link IntegerField}
	 * @throws InvalidValueException if given information table does not contain exactly one active decision attribute
	 */
	public static Instances convert(InformationTable informationTable, String relationName) {
		Precondition.notNull(informationTable, "Information table to be converted to instances is null.");
		
		Attribute attribute;
		EvaluationAttribute evaluationAttribute;
		int numberOfAttributes = informationTable.getNumberOfAttributes();
		int numberOfObjects = informationTable.getNumberOfObjects();
		
		weka.core.Attribute wekaDecisionAttribute = null;
		weka.core.Attribute wekaAttribute = null;
		
		ArrayList<weka.core.Attribute> wekaAttributes = new ArrayList<weka.core.Attribute>(numberOfAttributes); //contains WEKA attributes, possibly with spare space
		
		int[] wekaAttributeIndex2RuleLearnAttributeIndex = new int[numberOfAttributes]; //encodes mapping between WEKA's attribute index and ruleLearn's attribute index, possibly with spare space (if not all rL attributes will be converted)
		
		int wekaAttributeIndex = 0;
		int decisionAttributeIndex = -1;
		
		String attributeName;
		boolean skipAttribute = false;
		boolean suffixGain;
		boolean suffixCost;
		
		for (int j = 0; j < numberOfAttributes; j++) {
			if (skipAttribute) { //second attribute from a pair of attributes with imposed preference orders
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
						attributeName += Instances2InformationTable.mergedAttributeIndicator; //encode information that produced WEKA's attribute is a merged one!
					} else {
						//encode preference type in attribute's name
						if (evaluationAttribute.getPreferenceType() == AttributePreferenceType.GAIN) {
							attributeName += Instances2InformationTable.gainAttributeIndicator;
						} else if (evaluationAttribute.getPreferenceType() == AttributePreferenceType.COST) {
							attributeName += Instances2InformationTable.costAttributeIndicator;
						}
					}
					
					if (evaluationAttribute.getMissingValueType() instanceof UnknownSimpleFieldMV2) {
						attributeName += Instances2InformationTable.mv2MissingValueTypeIndicator;
					} else if (evaluationAttribute.getMissingValueType() instanceof UnknownSimpleFieldMV15) {
						attributeName += Instances2InformationTable.mv15MissingValueTypeIndicator;
					}
					
					if (evaluationAttribute.getValueType() instanceof EnumerationField) {
						wekaAttribute = new weka.core.Attribute(attributeName, Arrays.asList(((EnumerationField)evaluationAttribute.getValueType()).getElementList().getElements()));
					} else if (evaluationAttribute.getValueType() instanceof RealField) {
						wekaAttribute = new weka.core.Attribute(attributeName);
					} else if (evaluationAttribute.getValueType() instanceof IntegerField) {
						attributeName += Instances2InformationTable.integerAttributeIndicator;
						wekaAttribute = new weka.core.Attribute(attributeName);
					}
					else {
						throw new UnsupportedOperationException("Not supported attribute's value type for attribute "+evaluationAttribute.getName()+".");
					}
					
					if (evaluationAttribute.getType() == AttributeType.DECISION) {
						if (wekaDecisionAttribute == null) {
							wekaDecisionAttribute = wekaAttribute; //remember first decision attribute (to add it as the last attribute)
							decisionAttributeIndex = j;
						} else {
							throw new InvalidValueException("More than one decision attribute found in information table.");
						}
					} else { //condition attribute
						wekaAttributes.add(wekaAttribute);
						wekaAttributeIndex2RuleLearnAttributeIndex[wekaAttributeIndex++] = j;
					}
				}
			}
		}
		
		if (wekaDecisionAttribute != null) {
			wekaAttributes.add(wekaDecisionAttribute); //add decision attribute as the last one
			wekaAttributeIndex2RuleLearnAttributeIndex[wekaAttributeIndex++] = decisionAttributeIndex;
		} else {
			throw new InvalidValueException("No decision attribute found in information table.");
		}
		
		int numberOfWekaAttributes = wekaAttributes.size();
		
		Instances instances = new Instances(relationName, wekaAttributes, 0);
		instances.setClassIndex(numberOfWekaAttributes - 1); //last attribute
		Instance instance;		
				
		for (int i = 0; i < numberOfObjects; i++) {
			instance = new DenseInstance(numberOfWekaAttributes);
			instance.setDataset(instances);
			
			for (int j = 0; j < numberOfWekaAttributes; j++) {
				if (informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j]) instanceof RealField) {
					instance.setValue(j, ((RealField)informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j])).getValue());
				} else if (informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j]) instanceof IntegerField) {
					instance.setValue(j, ((IntegerField)informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j])).getValue());
				} else if (informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j]) instanceof EnumerationField) {
					instance.setValue(j, ((EnumerationField)informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j])).getElement());
				} else if (informationTable.getField(i, wekaAttributeIndex2RuleLearnAttributeIndex[j]) instanceof UnknownSimpleField) { //handles missing value
					instance.setMissing(j);
				}
			}
			
			instances.add(instance);
		}
		
		return instances;
	}
	
	public static void main(String[] args) {
		InformationTable informationTable;
		
		try {
			informationTable = InformationTableBuilder.safelyBuildFromJSONFile(
					"src/main/resources/data/json-metadata/bank-churn-4000-v8_0.25_mv1.5_mv2 metadata.json",
					"src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.json");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		Instances instances = convert(informationTable, "churn-4000-v8_0.25");
		//instances.deleteAttributeAt(instances.attribute("NumOfProducts_c").index()); //remove cloned attribute
		
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(instances);
		
		try {
			arffSaver.setFile(new File("src/main/resources/data/json-objects/bank-churn-4000-v8_0.25 data.arff"));
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		try {
			arffSaver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}
}

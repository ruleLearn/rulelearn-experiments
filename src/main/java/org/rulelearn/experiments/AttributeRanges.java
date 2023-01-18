/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.InformationTable;
import org.rulelearn.types.EvaluationField;
import org.rulelearn.types.IntegerField;
import org.rulelearn.types.RealField;

/**
 * Stores ranges of attribute values for subsequent attributes of an {@link InformationTable information table}.
 * Takes into account active condition or decision evaluation attributes whose value type is {@link IntegerField} or {@link RealField}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class AttributeRanges {
	public static class AttributeRange {
		int attributeIndex = -1;
		double min;
		double max;
		
		public AttributeRange(int attributeIndex, double min, double max) {
			if (min > max) {
				throw new InvalidValueException("Min > max.");
			}
			this.attributeIndex = attributeIndex;
			this.min = min;
			this.max = max;
		}

		public int getAttributeIndex() {
			return attributeIndex;
		}

		public double getMin() {
			return min;
		}

		public double getMax() {
			return max;
		}
		
	}
	
	List<AttributeRange> attributeRanges;
	Map<Integer,Integer> attributeIndex2attributeRangeIndex;

	public AttributeRanges(List<AttributeRange> attributeRanges) {
		this.attributeRanges = attributeRanges;
		attributeIndex2attributeRangeIndex = new HashMap<Integer, Integer>();
		
		int index = 0;
		for (AttributeRange attributeRange : attributeRanges) {
			attributeIndex2attributeRangeIndex.put(attributeRange.attributeIndex, index++);
		}
	}
	
	public AttributeRanges(InformationTable informationTable) {
		attributeRanges = new ArrayList<AttributeRange>();
		attributeIndex2attributeRangeIndex = new HashMap<Integer, Integer>();
		
		int numAttributes = informationTable.getNumberOfAttributes();
		int numObjects = informationTable.getNumberOfObjects();
		double min;
		double max;
		EvaluationField evaluationField;
		double value;
		
		for (int j = 0; j < numAttributes; j++) { //search for eligible attributes
			if (informationTable.getAttribute(j).isActive() && informationTable.getAttribute(j).getValueType() instanceof EvaluationField) {
				EvaluationAttribute evaluationAttribute = (EvaluationAttribute)informationTable.getAttribute(j);
				
				if (evaluationAttribute.getValueType() instanceof IntegerField || evaluationAttribute.getValueType() instanceof RealField) { //condition or decision attribute
					min = Double.NaN;
					max = Double.NaN;
					
					for (int i = 0; i < numObjects; i++) { //search min & max in the current column, among all objects
						evaluationField = (EvaluationField)informationTable.getField(i, j);
						
						if (evaluationField instanceof RealField) {
							value = ((RealField)evaluationField).getValue();
						} else if (evaluationField instanceof IntegerField) {
							value = ((IntegerField)evaluationField).getValue();
						} else { //a missing value
							continue;
						}
						
						if (min != min || value < min) { //NaN (first comparison) or smaller value found
							min = value;
						}
						
						if (max != max || value > max) { //NaN (first comparison) or greater value found
							max = value;
						}
					}
					
					attributeRanges.add(new AttributeRange(j, min, max));
					attributeIndex2attributeRangeIndex.put(j, attributeRanges.size() - 1);
				} //if
			} //if
		} //for
	}
	
	public AttributeRange getRange(int attributeIndex) {
		return attributeRanges.get(attributeIndex2attributeRangeIndex.get(attributeIndex)); //can throw an exception
	}
	
}

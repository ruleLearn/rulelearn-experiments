/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.types.EnumerationField;
import org.rulelearn.types.EnumerationFieldFactory;
import org.rulelearn.types.IntegerField;
import org.rulelearn.types.IntegerFieldFactory;
import org.rulelearn.types.RealField;
import org.rulelearn.types.RealFieldFactory;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAClassifer implements ClassificationModel {
	
	AbstractClassifier trainedClassifier; //trained classifier

	public WEKAClassifer(AbstractClassifier trainedClassifier) {
		this.trainedClassifier = trainedClassifier;
	}

	@Override
	public OrdinalMisclassificationMatrix validate(Data testData) {
		Instances test = InformationTable2Instances.convert(testData.getInformationTable(), testData.getName());
		double value;
		
		int testDataSize = testData.getInformationTable().getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testData.getInformationTable().getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testData.getInformationTable().getDecisions(true);
		SimpleDecision[] assignedDecisions = new SimpleDecision[testDataSize]; //will contain assigned decisions
		
		int decisionAttributeIndex = ((SimpleDecision)orderOfDecisions[0]).getAttributeIndex();
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)testData.getInformationTable().getAttribute(decisionAttributeIndex);
		
		for (int i = 0; i < test.numInstances(); i++) {
			try {
				value = trainedClassifier.classifyInstance(test.instance(i));
				
				if (decisionAttribute.getValueType() instanceof IntegerField) {
					assignedDecisions[i] = new SimpleDecision(IntegerFieldFactory.getInstance().create((int)value, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
				} else if (decisionAttribute.getValueType() instanceof RealField) {
					assignedDecisions[i] = new SimpleDecision(RealFieldFactory.getInstance().create(value, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
				} else if (decisionAttribute.getValueType() instanceof EnumerationField) {
					assignedDecisions[i] = new SimpleDecision(EnumerationFieldFactory.getInstance().create(
							((EnumerationField)decisionAttribute.getValueType()).getElementList(), (int)value, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
				}
			} catch (Exception e) {
				e.printStackTrace();
				return null; //TODO
			}
		}
		
		return new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
	}

	@Override
	public String getModelDescription() {
		String options = Arrays.asList(trainedClassifier.getOptions()).stream().collect(Collectors.joining(" "));
		return "[Options: " + options + "]" + System.lineSeparator() + trainedClassifier.toString();
	}

}

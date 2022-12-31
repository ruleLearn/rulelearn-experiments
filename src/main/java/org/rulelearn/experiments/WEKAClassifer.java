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
 * Generic WEKA classifier.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAClassifer implements ClassificationModel {
	
	public static class ValidationSummary extends ClassificationModel.ValidationSummary {
		@Override
		public String toString() {
			return "[Classification]: --.";
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		String options;
		String trainedClassifier;
		
		public ModelDescription(String options, String trainedClassifier) {
			this.options = options;
			this.trainedClassifier = trainedClassifier;
		}

		@Override
		public String toString() {
			return "[Options: " + options + "]" + System.lineSeparator() + trainedClassifier;
		}
	}
	
	AbstractClassifier trainedClassifier; //trained classifier
	
	ValidationSummary validationSummary = null;
	String modelLearnerDescription;
	ModelDescription modelDescription = null;

	public WEKAClassifer(AbstractClassifier trainedClassifier, String modelLearnerDescription) {
		this.trainedClassifier = trainedClassifier;
		this.modelLearnerDescription = modelLearnerDescription;
	}

	@Override
	public ModelValidationResult validate(Data testData) {
		int testDataSize = testData.getInformationTable().getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testData.getInformationTable().getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testData.getInformationTable().getDecisions(true);
		SimpleDecision[] assignedDecisions = new SimpleDecision[testDataSize]; //will contain assigned decisions
		
		int decisionAttributeIndex = ((SimpleDecision)orderOfDecisions[0]).getAttributeIndex();
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)testData.getInformationTable().getAttribute(decisionAttributeIndex);
		
		Instances instances = testData.getInstances(); //InformationTable2Instances.convert(testData.getInformationTable(), testData.getName());
		double value;
		
		for (int i = 0; i < instances.numInstances(); i++) {
			try {
				value = trainedClassifier.classifyInstance(instances.instance(i));
				assignedDecisions[i] = wekaClassificationResult2SimpleDecision(value, decisionAttribute, decisionAttributeIndex);
			} catch (Exception e) {
				e.printStackTrace();
				return null; //TODO: handle exception?
			}
		}
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		validationSummary = new ValidationSummary();
		
		return new ModelValidationResult(ordinalMisclassificationMatrix, (long)ordinalMisclassificationMatrix.getNumberOfCorrectAssignments(), (long)instances.numInstances(), 0L, 0L); //all decisions assigned by main model (no abstaining!)
	}
	
	private SimpleDecision wekaClassificationResult2SimpleDecision(double wekaClassificationResult, EvaluationAttribute decisionAttribute, int decisionAttributeIndex) {
		if (decisionAttribute.getValueType() instanceof IntegerField) {
			return new SimpleDecision(IntegerFieldFactory.getInstance().create((int)wekaClassificationResult, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
		} else if (decisionAttribute.getValueType() instanceof RealField) {
			return new SimpleDecision(RealFieldFactory.getInstance().create(wekaClassificationResult, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
		} else if (decisionAttribute.getValueType() instanceof EnumerationField) {
			return new SimpleDecision(EnumerationFieldFactory.getInstance().create(
					((EnumerationField)decisionAttribute.getValueType()).getElementList(), (int)wekaClassificationResult, decisionAttribute.getPreferenceType()), decisionAttributeIndex);
		} else {
			return null;
		}
	}
	
	public ValidationSummary getValidationSummary() {
		return validationSummary;
	}

	@Override
	public ModelDescription getModelDescription() {
		if (modelDescription == null) {
			String options = Arrays.asList(trainedClassifier.getOptions()).stream().collect(Collectors.joining(" "));
			
			modelDescription = new ModelDescription(options, trainedClassifier.toString());
		}
		
		return modelDescription;
	}

	@Override
	public SimpleDecision classify(int i, Data data) {
		int decisionAttributeIndex = ((SimpleDecision)data.getInformationTable().getDecisions()[0]).getAttributeIndex(); //takes decision from the first object, just to get decision attribute index
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)data.getInformationTable().getAttribute(decisionAttributeIndex);

		double wekaClassificationResult;
		try {
			wekaClassificationResult = trainedClassifier.classifyInstance(data.getInstances().instance(i));
		} catch (Exception e) {
			e.printStackTrace();
			return null; //TODO: handle exception?
		}
		
		return wekaClassificationResult2SimpleDecision(wekaClassificationResult, decisionAttribute, decisionAttributeIndex);
	}
	
	@Override
	public String getModelLearnerDescription() {
		return modelLearnerDescription;
	}

}

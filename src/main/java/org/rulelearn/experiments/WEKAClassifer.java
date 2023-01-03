/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.ModeRuleClassifier.ValidationSummary;
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
		double originalDecisionsQualityOfApproximation; //not used if -1.0
		double assignedDecisionsQualityOfApproximation; //not used if -1.0
		
		public ValidationSummary(double originalDecisionsQualityOfApproximation,
				double assignedDecisionsQualityOfApproximation) {
			this.originalDecisionsQualityOfApproximation = originalDecisionsQualityOfApproximation;
			this.assignedDecisionsQualityOfApproximation = assignedDecisionsQualityOfApproximation;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder(120);
			
			boolean appended = false;
			sb.append("[Summary]: ");
			if (originalDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, "original quality: %.4f", originalDecisionsQualityOfApproximation));
				appended = true;
			}
			if (assignedDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, ", assigned quality: %.4f", assignedDecisionsQualityOfApproximation));
				appended = true;
			}
			if (!appended) {
				sb.append("--");
			}
			sb.append(".");
			
			return sb.toString();
		}
	}
	
	public static class ModelDescriptionBuilder extends ClassificationModel.ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array is not an instance of {@link ModelDescription[]}.
		 */
		@Override
		ModelDescription build(ClassificationModel.ModelDescription... genericModelDescriptions) {
			ModelDescription[] modelDescriptions = new ModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (ModelDescription)genericModelDescription;
			}
			return new ModelDescription(modelDescriptions);
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		String options;
		String trainedClassifier;
		boolean aggregated = false;
		
		public ModelDescription(String options, String trainedClassifier) {
			this.options = options;
			this.trainedClassifier = trainedClassifier;
			aggregated = false;
		}
		
		/**
		 * @throws InvalidValueException if given array is empty
		 * 
		 * @param modelDescriptions array with model descriptions
		 */
		public ModelDescription(ModelDescription... modelDescriptions) {
			if (modelDescriptions.length == 0) {
				throw new InvalidValueException("Cannot aggregate over an empty array of model descriptions.");
			} else if (modelDescriptions.length == 1) {
				options = modelDescriptions[0].options;
				trainedClassifier = modelDescriptions[0].trainedClassifier;
				aggregated = false;
			} else {
				options = modelDescriptions[0].options;
				trainedClassifier = "classifier not available when aggregating model descriptions";
				aggregated = true;
			}
		}

		@Override
		public String toString() {
			if (!aggregated) {
				return "[Options: " + options + "]" + (BatchExperiment.printWEKATrainedClassifiers ?  System.lineSeparator() + trainedClassifier : "");
			} else {
				return "[Options: " + options + "]";
			}
		}

		@Override
		public ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new ModelDescriptionBuilder();
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
		
		for (int i = 0; i < testDataSize; i++) {
			try {
				value = trainedClassifier.classifyInstance(instances.instance(i));
				assignedDecisions[i] = wekaClassificationResult2SimpleDecision(value, decisionAttribute, decisionAttributeIndex);
			} catch (Exception e) {
				e.printStackTrace();
				return null; //TODO: handle exception?
			}
		}
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		double originalDecisionsQualityOfApproximation = -1.0;
		long originalDecisionsConsistentObjectsCount = -1L;
		
		double assignedDecisionsQualityOfApproximation = -1.0;
		long assignedDecisionsConsistentObjectsCount = -1L;
		
		if (BatchExperiment.checkConsistencyOfAssignedDecisions) {
			originalDecisionsQualityOfApproximation = getQualityOfApproximation(testData.getInformationTable(), 0.0);
			originalDecisionsConsistentObjectsCount = Math.round(originalDecisionsQualityOfApproximation * testDataSize); //go back to integer number
			
			assignedDecisionsQualityOfApproximation = getQualityOfApproximationForDecisions(testData.getInformationTable(), assignedDecisions, 0.0);
			assignedDecisionsConsistentObjectsCount = Math.round(assignedDecisionsQualityOfApproximation * testDataSize); //go back to integer number
		}
		
		this.validationSummary = new ValidationSummary(originalDecisionsQualityOfApproximation, assignedDecisionsQualityOfApproximation);
		
		return new ModelValidationResult(ordinalMisclassificationMatrix, (long)ordinalMisclassificationMatrix.getNumberOfCorrectAssignments(), (long)instances.numInstances(),
				0L, 0L, //all decisions assigned by main model (no abstaining of main model!), so default model is not used
				getModelDescription(),
				0L, testDataSize); //no rules
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

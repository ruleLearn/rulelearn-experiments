/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;
import org.rulelearn.experiments.ModelValidationResult.ClassifierType;
import org.rulelearn.experiments.ModelValidationResult.DefaultClassificationType;
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
	
	public static class ModelDescriptionBuilder extends ClassificationModel.ModelDescriptionBuilder {
		
		/**
		 * @throws ClassCastException if given array is not an instance of {@link ModelDescription[]}.
		 */
		@Override
		ModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) { //aggregationMode is ignored
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
		public String toShortString() {
			return "[Options: " + options + "]";
		}

		@Override
		public ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new ModelDescriptionBuilder();
		}

	}
	
	AbstractClassifier trainedClassifier; //trained classifier
	ModelLearningStatistics modelLearningStatistics;
	ModelDescription modelDescription = null;

	public WEKAClassifer(AbstractClassifier trainedClassifier, ModelLearningStatistics modelLearningStatistics) {
		this.trainedClassifier = trainedClassifier;
		this.modelLearningStatistics = modelLearningStatistics;
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
		
		ClassificationStatistics classificationStatistics = new ClassificationStatistics(DefaultClassificationType.NONE, ClassifierType.WEKA_CLASSIFIER);
		
		for (int i = 0; i < testDataSize; i++) {
			try {
				value = trainedClassifier.classifyInstance(instances.instance(i));
				assignedDecisions[i] = wekaClassificationResult2SimpleDecision(value, decisionAttribute, decisionAttributeIndex);
				if (assignedDecisions[i].equals(originalDecisions[i])) {
					classificationStatistics.increaseMainModelCorrectCount(1);
				} else {
					classificationStatistics.increaseMainModelIncorrectCount(1);
				}
			} catch (Exception e) {
				e.printStackTrace();
				return null; //TODO: handle exception?
			}
		}
		
		classificationStatistics.totalNumberOfClassifiedObjects = testDataSize;
		
		OrdinalMisclassificationMatrix ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions, originalDecisions, assignedDecisions);
		
		if (BatchExperiment.checkConsistencyOfTestDataDecisions) {
			long start = System.currentTimeMillis();
			
			classificationStatistics.totalNumberOfPreConsistentTestObjects =
					ClassificationModel.getNumberOfConsistentObjects(testData.getInformationTable(), 0.0);
			classificationStatistics.totalNumberOfPostConsistentTestObjectsIfDecisionsAssignedByMainModelAndDefaultClass = -1L; //not used
			classificationStatistics.totalNumberOfPostConsistentTestObjectsIfDecisionsAssignedByMainAndDefaultModel =
					ClassificationModel.getNumberOfConsistentObjects(testData.getInformationTable(), assignedDecisions, 0.0);
			classificationStatistics.totalNumberOfPreAndPostConsistentTestObjectsIfDecisionsAssignedByMainModelAndDefaultClass = -1L; //not used
			classificationStatistics.totalNumberOfPreAndPostConsistentTestObjectsIfDecisionsAssignedByMainAndDefaultModel =
					ClassificationModel.getNumberOfPreAndPostConsistentObjects(testData.getInformationTable(), assignedDecisions, 0.0);
			
			classificationStatistics.avgQualityOfClassification = (double)classificationStatistics.totalNumberOfPreConsistentTestObjects / classificationStatistics.totalNumberOfClassifiedObjects;
			
			classificationStatistics.avgAccuracy = classificationStatistics.getOverallAccuracy();
			
			classificationStatistics.totalStatisticsCountingTime = System.currentTimeMillis() - start;
		}
		
		return new ModelValidationResult(ordinalMisclassificationMatrix, classificationStatistics, modelLearningStatistics, getModelDescription());
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
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}

}

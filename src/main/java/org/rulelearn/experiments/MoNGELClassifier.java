package org.rulelearn.experiments;

import java.util.Locale;

import org.rulelearn.core.InvalidTypeException;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.EvaluationAttribute;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.ModelValidationResult.ClassificationStatistics;
import org.rulelearn.experiments.ModelValidationResult.ClassifierType;
import org.rulelearn.experiments.ModelValidationResult.DefaultClassificationType;
import org.rulelearn.types.EnumerationField;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

import keel.Algorithms.Monotonic_Classification.MoNGEL.MoNGEL;
import keel.Dataset.InstanceSet;

/**
 * KEEL classifier applying rules induced using MoNGEL algorithm.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class MoNGELClassifier extends KEELClassifier {
	
	public static class ModelDescriptionBuilder extends ClassificationModel.ModelDescriptionBuilder {
		
		/**
		 * @throws ClassCastException if given array does not contain only {@link ModelDescription} objects.
		 */
		@Override
		ModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) { //aggregationMode is ignored
			ModelDescription[] modelDescriptions = new ModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (ModelDescription)genericModelDescription;
			}
			return new ModelDescription(aggregationMode, modelDescriptions);
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		long totalRulesCount = 0L;
		String trainedClassifier;
		int aggregationCount = 0; //tells how many ModelDescription objects have been used to build this object
		AggregationMode aggregationMode = AggregationMode.NONE;
		
		public ModelDescription(long rulesCount, String trainedClassifier) {
			this.totalRulesCount = rulesCount;
			this.trainedClassifier = trainedClassifier;
			aggregationCount = 1;
			aggregationMode = AggregationMode.NONE;
		}
		
		/**
		 * Aggregates given model descriptions.
		 * 
		 * @param aggregationMode aggregation mode
		 * @param modelDescriptions array with model descriptions to be aggregated
		 * 
		 * @throws InvalidValueException if aggregation mode is {@code null} or equal to {@link AggregationMode#NONE}
		 */
		public ModelDescription(AggregationMode aggregationMode, ModelDescription... modelDescriptions) {
			if (aggregationMode == null || aggregationMode == AggregationMode.NONE) {
				throw new InvalidValueException("Incorrect aggregation mode.");
			}
			
			for (ModelDescription modelDescription : modelDescriptions) {
				totalRulesCount += modelDescription.totalRulesCount;
				aggregationCount += modelDescription.aggregationCount;
			}
			
			trainedClassifier = "classifier not available when aggregating model descriptions";
			this.aggregationMode = aggregationMode;

			if (this.aggregationMode == AggregationMode.MEAN_AND_DEVIATION) {
				//TODO: calculate means and standard deviations
			}
		}
		
		@Override
		public String toString() { //TODO: if aggregationMode == AggregationMode.MEAN_AND_DEVIATION, then print also standard deviations calculated in constructor
			StringBuilder sb = new StringBuilder(128);
			
			if (aggregationCount == 1) {
				sb.append("number of rules: ").append(totalRulesCount);
				
				if (BatchExperiment.printTrainedClassifiers) {
					sb.append(System.lineSeparator());
					sb.append(trainedClassifier);
				}
			} else {
				sb.append(String.format(Locale.US, "avg. number of rules: %.2f", aggregationCount > 0 ? (double)totalRulesCount / aggregationCount : 0.0));
			}
			
			return sb.toString();
		}
		
		@Override
		public String toShortString() {
			return aggregationCount == 1 ?
					String.format(Locale.US, "number of rules: %d", totalRulesCount) :
					String.format(Locale.US, "avg. number of rules: %.2f",
							aggregationCount > 0 ?
									(double)totalRulesCount / aggregationCount :
									0.0);
		}
		
		@Override
		public ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new ModelDescriptionBuilder();
		}
		
	}
	
	//******************** BEGIN class members ********************
	
	MoNGEL trainedClassifier;
	AttributeRanges attributeRanges;
	ModelDescription modelDescription = null;

	/**
	 * Class constructor.
	 * 
	 * @param modelLearningStatistics
	 */
	public MoNGELClassifier(MoNGEL trainedClassifier, AttributeRanges attributeRanges, ModelLearningStatistics modelLearningStatistics) {
		super(modelLearningStatistics);
		this.attributeRanges = attributeRanges;
		this.trainedClassifier = trainedClassifier;
	}

	@Override
	public ModelValidationResult validate(Data testData) { //values of decision attribute has to belong to an enumerated domain (be enumeration fields)
		int testDataSize = testData.getInformationTable().getNumberOfObjects(); //it is assumed that testDataSize > 0
		Decision[] orderOfDecisions = testData.getInformationTable().getOrderedUniqueFullyDeterminedDecisions();
		Decision[] originalDecisions = testData.getInformationTable().getDecisions(true);
		SimpleDecision[] assignedDecisions = new SimpleDecision[testDataSize]; //will contain assigned decisions
		
		int decisionAttributeIndex = ((SimpleDecision)orderOfDecisions[0]).getAttributeIndex();
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)testData.getInformationTable().getAttribute(decisionAttributeIndex);
		
		InstanceSet testInstanceSet = InformationTable2InstanceSet.convert(testData.getInformationTable(), testData.getName(), attributeRanges);
		//trainedClassifier.loadTestData(testInstanceSet); //IMPORTANT!
		//int[] testPredictions = trainedClassifier.classifyTestData();
		int[] testPredictions = trainedClassifier.classify(testInstanceSet); //MoNGEL prediction == index in the enum domain of the decision attribute
		
		ClassificationStatistics classificationStatistics = new ClassificationStatistics(DefaultClassificationType.NONE, ClassifierType.OTHER_CLASSIFIER);
		
		if (decisionAttribute.getValueType() instanceof EnumerationField) {
			EnumerationField decisionAttributeValueType = (EnumerationField)decisionAttribute.getValueType();
			
			for (int i = 0; i < testDataSize; i++) {
				assignedDecisions[i] = new SimpleDecision(
						decisionAttributeValueType.getDefaultFactory().create(decisionAttributeValueType.getElementList(), testPredictions[i], decisionAttribute.getPreferenceType()),
						decisionAttributeIndex);
				if (assignedDecisions[i].equals(originalDecisions[i])) {
					classificationStatistics.increaseMainModelCorrectCount(1);
				} else {
					classificationStatistics.increaseMainModelIncorrectCount(1);
				}
			}
		} else {
			throw new InvalidTypeException("Decision attribute values for MoNGEL classifier should be enumeration fields.");
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

	@Override
	public SimpleDecision classify(int i, Data data) {
		//TODO: implement if this classifier is to be used as a default classifier for VC-DRSA rules classifier
		
		throw new UnsupportedOperationException("Not implemented yet!");
	}

	@Override
	public ModelDescription getModelDescription() {
		if (modelDescription == null) {
			modelDescription = new ModelDescription(trainedClassifier.numRule(), trainedClassifier.getRulesAsText());
		}
		return modelDescription;
	}

}

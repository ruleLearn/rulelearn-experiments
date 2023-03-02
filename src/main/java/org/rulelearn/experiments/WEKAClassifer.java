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
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OLM;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * Generic WEKA classifier.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class WEKAClassifer implements ClassificationModel {
	
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
			return new ModelDescription(modelDescriptions);
		}
	}
	
	private class J48ModelDescriptionBuilder extends ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array does not contain only {@link J48ModelDescription} objects.
		 */
		@Override
		J48ModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) { //aggregationMode is ignored
			J48ModelDescription[] modelDescriptions = new J48ModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (J48ModelDescription)genericModelDescription;
			}
			return new J48ModelDescription(modelDescriptions);
		}
	}
	
	private class JRipModelDescriptionBuilder extends ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array does not contain only {@link JRipModelDescription} objects.
		 */
		@Override
		JRipModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) { //aggregationMode is ignored
			JRipModelDescription[] modelDescriptions = new JRipModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (JRipModelDescription)genericModelDescription;
			}
			return new JRipModelDescription(modelDescriptions);
		}
	}
	
	private class OLMModelDescriptionBuilder extends ModelDescriptionBuilder {
		/**
		 * @throws ClassCastException if given array does not contain only {@link OLMModelDescription} objects.
		 */
		@Override
		OLMModelDescription build(AggregationMode aggregationMode, ClassificationModel.ModelDescription... genericModelDescriptions) { //aggregationMode is ignored
			OLMModelDescription[] modelDescriptions = new OLMModelDescription[genericModelDescriptions.length];
			int index = 0;
			for (ClassificationModel.ModelDescription genericModelDescription : genericModelDescriptions) {
				modelDescriptions[index++] = (OLMModelDescription)genericModelDescription;
			}
			return new OLMModelDescription(modelDescriptions);
		}
	}
	
	/**
	 * {@link J48} Model description.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	private class J48ModelDescription extends ModelDescription {
		long totalSize = 0L;
		long totalNumLeaves = 0L;
		
		public J48ModelDescription(String options, String trainedClassifier, long size, long numLeaves) {
			super(options, trainedClassifier);
			this.totalSize = size;
			this.totalNumLeaves = numLeaves;
		}
		
		/**
		 * @throws InvalidValueException if given array is empty
		 * 
		 * @param modelDescriptions array with model descriptions
		 */
		public J48ModelDescription(J48ModelDescription... modelDescriptions) {
			if (modelDescriptions.length == 0) {
				throw new InvalidValueException("Cannot aggregate over an empty array of model descriptions.");
			} else if (modelDescriptions.length == 1) {
				options = modelDescriptions[0].options;
				trainedClassifier = modelDescriptions[0].trainedClassifier;
				aggregated = false;
				aggregationCount = 1;
			} else {
				options = modelDescriptions[0].options;
				trainedClassifier = "classifier not available when aggregating model descriptions";
				aggregated = true;
				
				for (J48ModelDescription modelDescription : modelDescriptions) {
					totalSize += modelDescription.totalSize;
					totalNumLeaves += modelDescription.totalNumLeaves;
					aggregationCount += modelDescription.aggregationCount;
				}
			}
		}
		
		@Override
		public String toShortString() {
			if (aggregated) {
				return String.format(Locale.US, "[Options: %s] Avg. num nodes: %.2f, avg. num leaves: %.2f", options, (double)totalSize / aggregationCount, (double)totalNumLeaves / aggregationCount);
			} else {
				return String.format(Locale.US, "[Options: %s] Num nodes: %d, num leaves: %d", options, totalSize, totalNumLeaves);
			}
		}
		
		@Override
		public String toCompressedShortString() {
			return String.format(Locale.US, "n: %.2f, r: %.2f", (double)totalSize / aggregationCount, (double)totalNumLeaves / aggregationCount);
		}
		
		@Override
		public J48ModelDescriptionBuilder getModelDescriptionBuilder() {
			return new J48ModelDescriptionBuilder();
		}
	}
	
	/**
	 * {@link JRip} Model description.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	private class JRipModelDescription extends ModelDescription {
		long totalNumRules = 0L;
		
		public JRipModelDescription(String options, String trainedClassifier, long numRules) {
			super(options, trainedClassifier);
			this.totalNumRules = numRules;
		}
		
		/**
		 * @throws InvalidValueException if given array is empty
		 * 
		 * @param modelDescriptions array with model descriptions
		 */
		public JRipModelDescription(JRipModelDescription... modelDescriptions) {
			if (modelDescriptions.length == 0) {
				throw new InvalidValueException("Cannot aggregate over an empty array of model descriptions.");
			} else if (modelDescriptions.length == 1) {
				options = modelDescriptions[0].options;
				trainedClassifier = modelDescriptions[0].trainedClassifier;
				aggregated = false;
				aggregationCount = 1;
			} else {
				options = modelDescriptions[0].options;
				trainedClassifier = "classifier not available when aggregating model descriptions";
				aggregated = true;
				
				for (JRipModelDescription modelDescription : modelDescriptions) {
					totalNumRules += modelDescription.totalNumRules;
					aggregationCount += modelDescription.aggregationCount;
				}
			}
		}
		
		@Override
		public String toShortString() {
			if (aggregated) {
				return String.format(Locale.US, "[Options: %s] Avg. num rules: %.2f", options, (double)totalNumRules / aggregationCount);
			} else {
				return String.format(Locale.US, "[Options: %s] Num rules: %d", options, totalNumRules);
			}
		}
		
		@Override
		public String toCompressedShortString() {
			return String.format(Locale.US, "r: %.2f", (double)totalNumRules / aggregationCount);
		}
		
		@Override
		public JRipModelDescriptionBuilder getModelDescriptionBuilder() {
			return new JRipModelDescriptionBuilder();
		}
	}
	
	/**
	 * {@link OLM} Model description.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	private class OLMModelDescription extends ModelDescription {
		long totalNumRules = 0L;
		
		public OLMModelDescription(String options, String trainedClassifier, long numRules) {
			super(options, trainedClassifier);
			this.totalNumRules = numRules;
		}
		
		/**
		 * @throws InvalidValueException if given array is empty
		 * 
		 * @param modelDescriptions array with model descriptions
		 */
		public OLMModelDescription(OLMModelDescription... modelDescriptions) {
			if (modelDescriptions.length == 0) {
				throw new InvalidValueException("Cannot aggregate over an empty array of model descriptions.");
			} else if (modelDescriptions.length == 1) {
				options = modelDescriptions[0].options;
				trainedClassifier = modelDescriptions[0].trainedClassifier;
				aggregated = false;
				aggregationCount = 1;
			} else {
				options = modelDescriptions[0].options;
				trainedClassifier = "classifier not available when aggregating model descriptions";
				aggregated = true;
				
				for (OLMModelDescription modelDescription : modelDescriptions) {
					totalNumRules += modelDescription.totalNumRules;
					aggregationCount += modelDescription.aggregationCount;
				}
			}
		}
		
		@Override
		public String toShortString() {
			if (aggregated) {
				return String.format(Locale.US, "[Options: %s] Avg. num rules: %.2f", options, (double)totalNumRules / aggregationCount);
			} else {
				return String.format(Locale.US, "[Options: %s] Num rules: %d", options, totalNumRules);
			}
		}
		
		@Override
		public String toCompressedShortString() {
			return String.format(Locale.US, "r: %.2f", (double)totalNumRules / aggregationCount);
		}
		
		@Override
		public OLMModelDescriptionBuilder getModelDescriptionBuilder() {
			return new OLMModelDescriptionBuilder();
		}
	}
	
	public static class ModelDescription extends ClassificationModel.ModelDescription {
		String options;
		String trainedClassifier = null;
		boolean aggregated = false;
		int aggregationCount = 0; //tells how many ModelDescription objects have been used to build this object
		
		private ModelDescription() {} //just used by subclasses
		
		public ModelDescription(String options, String trainedClassifier) {
			this.options = options;
			this.trainedClassifier = trainedClassifier;
			aggregated = false;
			aggregationCount = 1;
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
				aggregationCount = 1;
			} else {
				options = modelDescriptions[0].options;
				trainedClassifier = "classifier not available when aggregating model descriptions";
				aggregated = true;
				
				for (ModelDescription modelDescription : modelDescriptions) {
					aggregationCount += modelDescription.aggregationCount;
				}
			}
		}

		@Override
		public String toString() {
			if (!aggregated) {
				return "[Options: " + options + "]" + (trainedClassifier != null && BatchExperiment.printTrainedClassifiers ?  System.lineSeparator() + trainedClassifier : "");
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

		@Override
		public long getModelDescriptionCalculationTime() {
			return 0L;
		}

		@Override
		public void compress() {
			this.trainedClassifier = null;
		}

		@Override
		public String toCompressedShortString() {
			return "";
		}

	}
	
	//******************** BEGIN class members ********************
	
	AbstractClassifier trainedClassifier; //trained classifier
	Filter[] filters; //filters used during learning, in order, if any
	ModelLearningStatistics modelLearningStatistics;
	ModelDescription modelDescription = null;

	public WEKAClassifer(AbstractClassifier trainedClassifier, Filter[] filters, ModelLearningStatistics modelLearningStatistics) {
		this.trainedClassifier = trainedClassifier;
		this.filters = filters;
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
		
		Instances instances = testData.getInstances();
		
		if (filters != null) {
			for (Filter filter : filters) { //use subsequent filters, if there are any filters meant to be used (array is not empty)
				try {
					//filter.setInputFormat(instances);
					instances = Filter.useFilter(instances, filter);
				} catch (Exception exception) {
					exception.printStackTrace();
					return null; //TODO: handle exception?
				}
			}
		}
		
		double value;
		
		ClassificationStatistics classificationStatistics = new ClassificationStatistics(DefaultClassificationType.NONE, ClassifierType.OTHER_CLASSIFIER);
		
		for (int i = 0; i < testDataSize; i++) {
			try {
				value = trainedClassifier.classifyInstance(instances.instance(i));
				assignedDecisions[i] = wekaClassificationResult2SimpleDecision(value, decisionAttribute, decisionAttributeIndex);
				if (assignedDecisions[i].equals(originalDecisions[i])) {
					classificationStatistics.increaseMainModelCorrectCount(1);
				} else {
					classificationStatistics.increaseMainModelIncorrectCount(1);
				}
			} catch (Exception exception) {
				exception.printStackTrace();
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
			
			if (trainedClassifier instanceof J48) {
				try {
					modelDescription = new J48ModelDescription(options, BatchExperiment.printTrainedClassifiers ? trainedClassifier.toString() : null,
							(long)((J48)trainedClassifier).measureTreeSize(), (long)((J48)trainedClassifier).measureNumLeaves());
				} catch (Exception exception) {
					exception.printStackTrace(); //TODO: handle exception?
				}
			} else if (trainedClassifier instanceof JRip) {
				modelDescription = new JRipModelDescription(options, BatchExperiment.printTrainedClassifiers ? trainedClassifier.toString() : null,
						(long)((JRip)trainedClassifier).getRuleset().size());
			} else if (trainedClassifier instanceof OLM) {
				modelDescription = new OLMModelDescription(options, BatchExperiment.printTrainedClassifiers ? trainedClassifier.toString() : null,
						((OLM)trainedClassifier).getNumberOfRules());
			} else {
				modelDescription = new ModelDescription(options, BatchExperiment.printTrainedClassifiers ? trainedClassifier.toString() : null);
			}
		}
		
		return modelDescription;
	}

	@Override
	public SimpleDecision classify(int i, Data data) {
		int decisionAttributeIndex = ((SimpleDecision)data.getInformationTable().getDecisions()[0]).getAttributeIndex(); //takes decision from the first object, just to get decision attribute index
		EvaluationAttribute decisionAttribute = (EvaluationAttribute)data.getInformationTable().getAttribute(decisionAttributeIndex);
		
		Instances instances = data.getInstances();
		
		if (filters != null && filters.length > 0) {
			instances = new Instances(instances, i, 1); //if there are filters to use, then creates instances with just one instance
		}
		
		if (filters != null) {
			for (Filter filter : filters) { //use subsequent filters, if there are any filters meant to be used (array is not empty)
				try {
					//filter.setInputFormat(instances);
					instances = Filter.useFilter(instances, filter);
				} catch (Exception exception) {
					exception.printStackTrace();
					return null; //TODO: handle exception?
				}
			}
		}

		double wekaClassificationResult;
		try {
			wekaClassificationResult = trainedClassifier.classifyInstance(instances.instance(i));
		} catch (Exception exception) {
			exception.printStackTrace();
			return null; //TODO: handle exception?
		}
		
		return wekaClassificationResult2SimpleDecision(wekaClassificationResult, decisionAttribute, decisionAttributeIndex);
	}
	
	@Override
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}

}

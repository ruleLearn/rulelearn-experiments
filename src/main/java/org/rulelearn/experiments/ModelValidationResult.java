/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.rulelearn.core.InvalidSizeException;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.core.Precondition;
import org.rulelearn.data.Decision;
import org.rulelearn.experiments.ClassificationModel.ModelDescription;
import org.rulelearn.experiments.ModelValidationResult.DefaultClassificationType;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Compound model validation result, extending over {@link OrdinalMisclassificationMatrix}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ModelValidationResult {
	
	public enum DefaultClassificationType {
		USING_DEFAULT_CLASS,
		USING_DEFAULT_CLASSIFIER,
		NONE; //concern WEKA classifiers
	}
	
	public static class ClassificationStatistics {
		/* Main model counters */
		long preciseCorrectCount = 0L; //concerns precise classification using (VC-)DRSA rules and classification using a WEKA classifier
		long preciseIncorrectCount = 0L; //concerns precise classification using (VC-)DRSA rules and classification using a WEKA classifier
		
		long resolvingConflictCorrectCount = 0L; //concerns classification using (VC-)DRSA rules that involved conflict resolving (e.g., using "mode")
		long resolvingConflictIncorrectCount = 0L; //concerns classification using (VC-)DRSA rules that involved conflict resolving (e.g., using "mode")
		
		/* Default model counters */
		long defaultClassCorrectCount = 0L; //concerns classification using (VC-)DRSA rules when no rule covers object and default class is used
		long defaultClassIncorrectCount = 0L; //concerns classification using (VC-)DRSA rules when no rule covers object and default class is used
		
		long defaultClassifierCorrectCount = 0L; //concerns classification using (VC-)DRSA rules when no rule covers object and default classifier is used
		long defaultClassifierIncorrectCount = 0L; //concerns classification using (VC-)DRSA rules when no rule covers object and default classifier is used
		
		DefaultClassificationType defaultClassificationType;
		
		//in case of single validation (on a validation set) this is a sum of the numbers of covering rules over all classified objects;
		//e.g., if there are two objects, first covered by 3 rules, and second covered by 4 rules, then total number of covering rules is 7;
		//in case of aggregated model validation result, this is the sum of total number of covering rules over all validated models
		long totalNumberOfCoveringRules = 0L; //concerns classification using (VC-)DRSA rules; for WEKA classifiers remains zero
		long totalNumberOfClassifiedObjects = 0L;
		
		long originalDecisionsConsistentObjectsCount = -1L; //not used if -1.0
		long assignedDefaultClassDecisionsConsistentObjectsCount = -1L; //concerns classification using (VC-)DRSA rules (number of consistent objects if default model employs default decision class); not used if -1.0
		long assignedDecisionsConsistentObjectsCount = -1L; //not used if -1.0
		
		/**
		 * Constructs these classification statistics.
		 * 
		 * @param defaultClassificationType type of default classification (whether performed using default class or using default classifier)
		 */
		public ClassificationStatistics(DefaultClassificationType defaultClassificationType) {
			this.defaultClassificationType = Precondition.notNull(defaultClassificationType, "Default classification type is null.");
		}
		
		/**
		 * Constructs these classification statistics by summing all respective counters from given classification statistics.
		 * {@link #getDefaultClassificationType Default classification type} is taken from the first classification statistics.
		 * 
		 * @param classificationStatisticsSet array with classification statistics
		 * @throws InvalidSizeException if given array is empty
		 */
		public ClassificationStatistics(ClassificationStatistics... classificationStatisticsSet) {
			Precondition.nonEmpty(classificationStatisticsSet, "Set of classification statistics is empty.");
			
			this.defaultClassificationType = classificationStatisticsSet[0].defaultClassificationType;
			
			for (ClassificationStatistics classificationStatistics : classificationStatisticsSet) {
				preciseCorrectCount += classificationStatistics.preciseCorrectCount;
				preciseIncorrectCount += classificationStatistics.preciseIncorrectCount;
				resolvingConflictCorrectCount += classificationStatistics.resolvingConflictCorrectCount;
				resolvingConflictIncorrectCount += classificationStatistics.resolvingConflictIncorrectCount;
				
				defaultClassCorrectCount += classificationStatistics.defaultClassCorrectCount;
				defaultClassIncorrectCount += classificationStatistics.defaultClassIncorrectCount;
				defaultClassifierCorrectCount += classificationStatistics.defaultClassifierCorrectCount;
				defaultClassifierIncorrectCount += classificationStatistics.defaultClassifierIncorrectCount;
				
				totalNumberOfCoveringRules += classificationStatistics.totalNumberOfCoveringRules;
				totalNumberOfClassifiedObjects += classificationStatistics.totalNumberOfClassifiedObjects;
				
				if (classificationStatistics.originalDecisionsConsistentObjectsCount >= 0) {
					originalDecisionsConsistentObjectsCount += classificationStatistics.originalDecisionsConsistentObjectsCount;
				}
				if (classificationStatistics.assignedDefaultClassDecisionsConsistentObjectsCount >= 0) {
					assignedDefaultClassDecisionsConsistentObjectsCount += classificationStatistics.assignedDefaultClassDecisionsConsistentObjectsCount;
				}
				if (classificationStatistics.assignedDecisionsConsistentObjectsCount >= 0) {
					assignedDecisionsConsistentObjectsCount += classificationStatistics.assignedDecisionsConsistentObjectsCount;
				}
			}
		}
		
		public long getMainModelCorrectCount() {
			return preciseCorrectCount + resolvingConflictCorrectCount;
		}
		
		public long getMainModelIncorrectCount() {
			return preciseIncorrectCount + resolvingConflictIncorrectCount;
		}
		
		public long getMainModelCount() {
			return preciseCorrectCount + preciseIncorrectCount + resolvingConflictCorrectCount + resolvingConflictIncorrectCount;
		}
		
		public long getDefaultModelCorrectCount() {
			switch (defaultClassificationType) {
			case USING_DEFAULT_CLASS:
				return defaultClassCorrectCount;
			case USING_DEFAULT_CLASSIFIER:
				return defaultClassifierCorrectCount;
			case NONE:
				return 0L;
			default:
				throw new InvalidValueException("Incorrect value of default classification type.");
			}
		}
		
		public long getDefaultModelIncorrectCount() {
			switch (defaultClassificationType) {
			case USING_DEFAULT_CLASS:
				return defaultClassIncorrectCount;
			case USING_DEFAULT_CLASSIFIER:
				return defaultClassifierIncorrectCount;
			case NONE:
				return 0L;
			default:
				throw new InvalidValueException("Incorrect value of default classification type.");
			}
		}
		
		public long getDefaultModelCount() {
			return getDefaultModelCorrectCount() + getDefaultModelIncorrectCount();
		}
		
		public void increaseDefaultModelCorrectCount(int count) {
			switch (defaultClassificationType) {
			case USING_DEFAULT_CLASS:
				defaultClassCorrectCount += count;
				break;
			case USING_DEFAULT_CLASSIFIER:
				defaultClassifierCorrectCount += count;
				break;
			default:
				throw new InvalidValueException("Incorrect value of default classification type.");
			}
		}
		
		public void increaseDefaultModelIncorrectCount(int count) {
			switch (defaultClassificationType) {
			case USING_DEFAULT_CLASS:
				defaultClassIncorrectCount += count;
				break;
			case USING_DEFAULT_CLASSIFIER:
				defaultClassifierIncorrectCount += count;
				break;
			default:
				throw new InvalidValueException("Incorrect value of default classification type.");
			}
		}
		
		public long getDefaultClassCorrectCount() {
			return defaultClassCorrectCount;
		}
		public long getDefaultClassIncorrectCount() {
			return defaultClassIncorrectCount;
		}
		public long getDefaultClassCount() {
			return defaultClassCorrectCount + defaultClassIncorrectCount;
		}
		
		public long getDefaultClassifierCorrectCount() {
			return defaultClassifierCorrectCount;
		}
		public long getDefaultClassifierIncorrectCount() {
			return defaultClassifierIncorrectCount;
		}
		public long getDefaultClassifierCount() {
			return defaultClassifierCorrectCount + defaultClassifierIncorrectCount;
		}
		
		public long getCorrectCount() {
			return getMainModelCorrectCount() + getDefaultModelCorrectCount();
		}
		
		public long getIncorrectCount() {
			return getMainModelIncorrectCount() + getDefaultModelIncorrectCount();
		}
		
		public long getPreciseCorrectCount() {
			return preciseCorrectCount;
		}
		
		public long getPreciseIncorrectCount() {
			return preciseIncorrectCount;
		}
		
		public long getPreciseCount() {
			return preciseCorrectCount + preciseIncorrectCount;
		}

		public long getResolvingConflictCorrectCount() {
			return resolvingConflictCorrectCount;
		}
		
		public long getResolvingConflictIncorrectCount() {
			return resolvingConflictIncorrectCount;
		}
		
		public long getResolvingConflictCount() {
			return resolvingConflictCorrectCount + resolvingConflictIncorrectCount;
		}
		
		public double getMainModelAccuracy() { //gets accuracy concerning the part of validation data for which main model was applied to classify objects
			long mainModelCount = getMainModelCount();
			return mainModelCount > 0 ? (double)getMainModelCorrectCount() / mainModelCount : 0.0;
		}
		
		public double getDefaultModelAccuracy() { //gets accuracy concerning the part of validation data for which default model was applied to classify objects
			long defaultModelCount = getDefaultModelCount();
			return defaultModelCount > 0 ? (double)getDefaultModelCorrectCount() / defaultModelCount : 0.0;
		}
		
		public double getDefaultClassAccuracy() { //gets accuracy concerning the part of validation data for which default model was applied to classify objects
			long defaultClassCount = getDefaultClassCount();
			return defaultClassCount > 0 ? (double)getDefaultClassCorrectCount() / defaultClassCount : 0.0;
		}
		
		public double getDefaultClassifierAccuracy() { //gets accuracy concerning the part of validation data for which default model was applied to classify objects
			long defaultClassifierCount = getDefaultClassifierCount();
			return defaultClassifierCount > 0 ? (double)getDefaultClassifierCorrectCount() / defaultClassifierCount : 0.0;
		}
		
		public double getMainModelDecisionsRatio() { //gets percent of situations when main model suggested decision
			return (double)getMainModelCount() / (getMainModelCount() + getDefaultModelCount());
		}
		
		public DefaultClassificationType getDefaultClassificationType () {
			return defaultClassificationType;
		}
		
		public long getTotalNumberOfCoveringRules() {
			return totalNumberOfCoveringRules;
		}

		public long getTotalNumberOfClassifiedObjects() {
			return totalNumberOfClassifiedObjects;
		}
		
		public double getAverageNumberOfCoveringRules() {
			return totalNumberOfClassifiedObjects > 0L ? (double)totalNumberOfCoveringRules / totalNumberOfClassifiedObjects : 0.0;
		}
		
		public long getOriginalDecisionsConsistentObjectsCount() {
			return originalDecisionsConsistentObjectsCount;
		}

		public long getAssignedDefaultClassDecisionsConsistentObjectsCount() {
			return assignedDefaultClassDecisionsConsistentObjectsCount;
		}

		public long getAssignedDecisionsConsistentObjectsCount() {
			return assignedDecisionsConsistentObjectsCount;
		}
		
		public double getAverageOriginalDecisionsConsistentObjectsCount() {
			return totalNumberOfClassifiedObjects > 0L ? (double)originalDecisionsConsistentObjectsCount / totalNumberOfClassifiedObjects : 0.0;
		}
		
		public double getAverageAssignedDefaultClassDecisionsConsistentObjectsCount() {
			return totalNumberOfClassifiedObjects > 0L ? (double)assignedDefaultClassDecisionsConsistentObjectsCount / totalNumberOfClassifiedObjects : 0.0;
		}
		
		public double getAverageAssignedDecisionsConsistentObjectsCount() {
			return totalNumberOfClassifiedObjects > 0L ? (double)assignedDecisionsConsistentObjectsCount / totalNumberOfClassifiedObjects : 0.0; 
		}
		
	}
	
	OrdinalMisclassificationMatrix ordinalMisclassificationMatrix;
	
	ClassificationStatistics classificationStatistics;
	
	ModelDescription modelDescription;
	
//	//total numbers of decisions
//	long numberOfCorrectDecisionsAssignedByMainModel = 0L; //concerns the part of validation data for which main model was applied to classify objects (e.g., covering decision rule(s))
//	long numberOfAllDecisionsAssignedByMainModel = 0L;
//	long numberOfCorrectDecisionsAssignedByDefaultModel = 0L; //concerns the part of validation data for which default model was applied to classify objects (e.g., assigning default decision class)
//	long numberOfAllDecisionsAssignedByDefaultModel = 0L; //can be 0 if default model is not used (which is the case when the main model classifies all objects itself)

//	//in case of single validation (on a validation set) this is a sum of the numbers of covering rules over all classified objects;
//	//e.g., if there are two objects, first covered by 3 rules, and second covered by 4 rules, then total number of covering rules is 7;
//	//in case of aggregated model validation result, this is the sum of total number of covering rules over all validated models
//	long totalNumberOfCoveringRules = 0L;
//	long totalNumberOfClassifiedObjects = 0L;
	
	public ModelValidationResult(OrdinalMisclassificationMatrix ordinalMisclassificationMatrix,
			ClassificationStatistics classificationStatistics,
//			long numberOfCorrectDecisionsAssignedByMainModel, long numberOfAllDecisionsAssignedByMainModel,
//			long numberOfCorrectDecisionsAssignedByDefaultModel, long numberOfAllDecisionsAssignedByDefaultModel,
			ModelDescription modelDescription) {
//			long totalNumberOfCoveringRules, long totalNumberOfClassifiedObjects) {
		
		this.ordinalMisclassificationMatrix = ordinalMisclassificationMatrix;
		this.classificationStatistics = classificationStatistics;
		this.modelDescription = modelDescription;
		
//		this.numberOfCorrectDecisionsAssignedByMainModel = numberOfCorrectDecisionsAssignedByMainModel;
//		this.numberOfAllDecisionsAssignedByMainModel = numberOfAllDecisionsAssignedByMainModel;
//		this.numberOfCorrectDecisionsAssignedByDefaultModel = numberOfCorrectDecisionsAssignedByDefaultModel;
//		this.numberOfAllDecisionsAssignedByDefaultModel = numberOfAllDecisionsAssignedByDefaultModel;
//		
//		this.totalNumberOfCoveringRules = totalNumberOfCoveringRules; //should be zero for classifiers not using rules
//		this.totalNumberOfClassifiedObjects = totalNumberOfClassifiedObjects;
	}
	
	public ModelValidationResult(Decision[] orderOfDecisions, ModelValidationResult... modelValidationResults) {
		ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions,
				Arrays.asList(modelValidationResults).stream().map(m -> m.getOrdinalMisclassificationMatrix()).collect(Collectors.toList()).toArray(new OrdinalMisclassificationMatrix[0]));
		
		classificationStatistics = new ClassificationStatistics(Arrays.asList(modelValidationResults).stream().map(m -> m.getClassificationStatistics()).collect(Collectors.toList()).toArray(new ClassificationStatistics[0]));
		
		ModelDescription[] modelDescriptions = Arrays.asList(modelValidationResults).stream().map(m -> m.getModelDescription()).collect(Collectors.toList()).toArray(new ModelDescription[0]);
		modelDescription = modelDescriptions[0].getModelDescriptionBuilder().build(modelDescriptions);
		
//		//just do summation, like in MisclassificationMatrix
//		for (ModelValidationResult modelValidationResult : modelValidationResults) {
//			numberOfCorrectDecisionsAssignedByMainModel += modelValidationResult.numberOfCorrectDecisionsAssignedByMainModel;
//			numberOfAllDecisionsAssignedByMainModel += modelValidationResult.numberOfAllDecisionsAssignedByMainModel;
//			numberOfCorrectDecisionsAssignedByDefaultModel += modelValidationResult.numberOfCorrectDecisionsAssignedByDefaultModel;
//			numberOfAllDecisionsAssignedByDefaultModel += modelValidationResult.numberOfAllDecisionsAssignedByDefaultModel;
//			
//			totalNumberOfCoveringRules += modelValidationResult.totalNumberOfCoveringRules;
//			totalNumberOfClassifiedObjects += modelValidationResult.totalNumberOfClassifiedObjects;
//		}
	}

	public OrdinalMisclassificationMatrix getOrdinalMisclassificationMatrix() {
		return ordinalMisclassificationMatrix;
	}
	
	public ClassificationStatistics getClassificationStatistics() {
		return classificationStatistics;
	}
	
	public double getOverallAccuracy() {
		return ordinalMisclassificationMatrix.getAccuracy();
	}
	
	public double getMainModelAccuracy() { //gets accuracy concerning the part of validation data for which main model was applied to classify objects
		return classificationStatistics.getMainModelAccuracy();
//		return numberOfAllDecisionsAssignedByMainModel > 0 ? (double)numberOfCorrectDecisionsAssignedByMainModel / numberOfAllDecisionsAssignedByMainModel : 0.0;
	}
	
	public double getDefaultModelAccuracy() { //gets accuracy concerning the part of validation data for which default model was applied to classify objects
		return classificationStatistics.getDefaultModelAccuracy();
//		return numberOfAllDecisionsAssignedByDefaultModel > 0 ? (double)numberOfCorrectDecisionsAssignedByDefaultModel / numberOfAllDecisionsAssignedByDefaultModel : 0.0;
	}
	
	public double getMainModelDecisionsRatio() { //percent of situations when main model suggested decision
		return classificationStatistics.getMainModelDecisionsRatio();
//		return (double)numberOfAllDecisionsAssignedByMainModel / (numberOfAllDecisionsAssignedByMainModel + numberOfAllDecisionsAssignedByDefaultModel);
	}

//	public long getNumberOfCorrectDecisionsAssignedByMainModel() {
//		return numberOfCorrectDecisionsAssignedByMainModel;
//	}

	public long getNumberOfAllDecisionsAssignedByMainModel() {
		return classificationStatistics.getMainModelCount();
//		return numberOfAllDecisionsAssignedByMainModel;
	}

//	public long getNumberOfCorrectDecisionsAssignedByDefaultModel() {
//		return numberOfCorrectDecisionsAssignedByDefaultModel;
//	}

	public long getNumberOfAllDecisionsAssignedByDefaultModel() {
		return classificationStatistics.getDefaultModelCount();
//		return numberOfAllDecisionsAssignedByDefaultModel;
	}
	
	public ModelDescription getModelDescription() {
		return modelDescription;
	}
	
	public long getTotalNumberOfCoveringRules() {
		return classificationStatistics.getTotalNumberOfCoveringRules();
//		return totalNumberOfCoveringRules;
	}

	public long getTotalNumberOfClassifiedObjects() {
		return classificationStatistics.getTotalNumberOfClassifiedObjects();
//		return totalNumberOfClassifiedObjects;
	}

//	public double getAvgNumberOfCoveringRules() {
//		return (double)totalNumberOfCoveringRules / totalNumberOfClassifiedObjects;
//	}

}

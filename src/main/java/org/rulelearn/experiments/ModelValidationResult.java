/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import org.rulelearn.core.InvalidSizeException;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.core.Precondition;
import org.rulelearn.data.Decision;
import org.rulelearn.experiments.ClassificationModel.ModelDescription;
import org.rulelearn.experiments.ClassificationModel.ModelLearningStatistics;
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
		NONE; //concerns WEKA classifiers
	}
	
	public enum ClassifierType {
		VCDRSA_RULES_CLASSIFIER,
		WEKA_CLASSIFIER
	}
	
	public static class MeansAndStandardDeviations {
		MeanAndStandardDeviation overallAverageAccuracy;
		MeanAndStandardDeviation mainModelAverageAccuracy;
		MeanAndStandardDeviation defaultModelAverageAccuracy;
		MeanAndStandardDeviation defaultClassAverageAccuracy;
		MeanAndStandardDeviation defaultClassifierAverageAccuracy;
		
		public MeansAndStandardDeviations(MeanAndStandardDeviation overallAverageAccuracy,
				MeanAndStandardDeviation mainModelAverageAccuracy, MeanAndStandardDeviation defaultModelAverageAccuracy,
				MeanAndStandardDeviation defaultClassAverageAccuracy, MeanAndStandardDeviation defaultClassifierAverageAccuracy) {
			this.overallAverageAccuracy = overallAverageAccuracy;
			this.mainModelAverageAccuracy = mainModelAverageAccuracy;
			this.defaultModelAverageAccuracy = defaultModelAverageAccuracy;
			this.defaultClassAverageAccuracy = defaultClassAverageAccuracy;
			this.defaultClassifierAverageAccuracy = defaultClassifierAverageAccuracy;
		}

		public MeanAndStandardDeviation getOverallAverageAccuracy() {
			return overallAverageAccuracy;
		}

		public MeanAndStandardDeviation getMainModelAverageAccuracy() {
			return mainModelAverageAccuracy;
		}

		public MeanAndStandardDeviation getDefaultModelAverageAccuracy() {
			return defaultModelAverageAccuracy;
		}

		public MeanAndStandardDeviation getDefaultClassAverageAccuracy() {
			return defaultClassAverageAccuracy;
		}

		public MeanAndStandardDeviation getDefaultClassifierAverageAccuracy() {
			return defaultClassifierAverageAccuracy;
		}

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
		
		//in case of single validation (on a validation set) this is a sum of the numbers of covering rules over all classified objects;
		//e.g., if there are two objects, first covered by 3 rules, and second covered by 4 rules, then total number of covering rules is 7;
		//in case of aggregated model validation result, this is the sum of total number of covering rules over all validated models
		long totalNumberOfCoveringRules = 0L; //concerns classification using (VC-)DRSA rules; for WEKA classifiers remains zero
		long totalNumberOfClassifiedObjects = 0L;
		
		long originalDecisionsConsistentTestObjectsTotalCount = -1L; //not used if < 0
		long assignedDefaultClassDecisionsConsistentTestObjectsTotalCount = -1L; //concerns classification using (VC-)DRSA rules (total number of consistent objects if default model employs default decision class); not used if < 0
		long assignedDecisionsConsistentTestObjectsTotalCount = -1L; //not used if < 0

		DefaultClassificationType defaultClassificationType;
		ClassifierType classifierType;
		
		AggregationMode aggregationMode = AggregationMode.NONE;
		MeansAndStandardDeviations meansAndStdDevs = null; //get calculated when aggregationMode == AggregationMode.MEAN_AND_DEVIATION 
		
		/**
		 * Constructs these classification statistics.
		 * 
		 * @param defaultClassificationType type of default classification (whether performed using default class or using default classifier)
		 * @param classifierType type of classifier
		 */
		public ClassificationStatistics(DefaultClassificationType defaultClassificationType, ClassifierType classifierType) {
			this.defaultClassificationType = Precondition.notNull(defaultClassificationType, "Default classification type is null.");
			this.classifierType = Precondition.notNull(classifierType, "Classifier type is null.");
		}
		
		/**
		 * Constructs these classification statistics by summing all respective counters from given classification statistics.
		 * {@link #getDefaultClassificationType Default classification type} is taken from the first classification statistics.
		 * 
		 * @param aggregationMode aggregation mode used when aggregating given classification statistics;
		 *        if equals {@link AggregationMode#MEAN_AND_DEVIATION}, then apart from normal calculations (sums),
		 *        also means and standard deviations are additionally calculated
		 * @param classificationStatisticsSet array with classification statistics
		 * @throws InvalidSizeException if given array is empty
		 */
		public ClassificationStatistics(AggregationMode aggregationMode, ClassificationStatistics... classificationStatisticsSet) {
			Precondition.nonEmpty(classificationStatisticsSet, "Set of classification statistics is empty.");
			
			if (aggregationMode == null || aggregationMode == AggregationMode.NONE) {
				throw new InvalidValueException("Incorrect aggregation mode.");
			}
			
			this.defaultClassificationType = classificationStatisticsSet[0].defaultClassificationType;
			this.classifierType = classificationStatisticsSet[0].classifierType;
			this.aggregationMode = aggregationMode;
			
			long originalDecisionsConsistentTestObjectsTotalCountSum = 0L;
			boolean originalDecisionsConsistentTestObjectsTotalCountIsUsed = false;
			long assignedDefaultClassDecisionsConsistentTestObjectsTotalCountSum = 0L;
			boolean assignedDefaultClassDecisionsConsistentTestObjectsTotalCountIsUsed = false;
			long assignedDecisionsConsistentTestObjectsTotalCountSum = 0L;
			boolean assignedDecisionsConsistentTestObjectsTotalCountIsUsed = false;

			//calculate sums
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
				
				if (classificationStatistics.originalDecisionsConsistentTestObjectsTotalCount >= 0) {
					originalDecisionsConsistentTestObjectsTotalCountSum += classificationStatistics.originalDecisionsConsistentTestObjectsTotalCount;
					originalDecisionsConsistentTestObjectsTotalCountIsUsed = true;
				}
				if (classificationStatistics.assignedDefaultClassDecisionsConsistentTestObjectsTotalCount >= 0) {
					assignedDefaultClassDecisionsConsistentTestObjectsTotalCountSum += classificationStatistics.assignedDefaultClassDecisionsConsistentTestObjectsTotalCount;
					assignedDefaultClassDecisionsConsistentTestObjectsTotalCountIsUsed = true;
				}
				if (classificationStatistics.assignedDecisionsConsistentTestObjectsTotalCount >= 0) {
					assignedDecisionsConsistentTestObjectsTotalCountSum += classificationStatistics.assignedDecisionsConsistentTestObjectsTotalCount;
					assignedDecisionsConsistentTestObjectsTotalCountIsUsed = true;
				}
			}
			
			if (originalDecisionsConsistentTestObjectsTotalCountIsUsed) {
				originalDecisionsConsistentTestObjectsTotalCount = originalDecisionsConsistentTestObjectsTotalCountSum;
			}
			if (assignedDefaultClassDecisionsConsistentTestObjectsTotalCountIsUsed) {
				assignedDefaultClassDecisionsConsistentTestObjectsTotalCount = assignedDefaultClassDecisionsConsistentTestObjectsTotalCountSum;
			}
			if (assignedDecisionsConsistentTestObjectsTotalCountIsUsed) {
				assignedDecisionsConsistentTestObjectsTotalCount = assignedDecisionsConsistentTestObjectsTotalCountSum;
			}
			
			//additionally calculate means and standard deviations
			if (aggregationMode == AggregationMode.MEAN_AND_DEVIATION) {
				int averagedStatisticsCount = 5;
				
				BiFunction<Integer, ClassificationStatistics, Double> modelIndex2Accuracy = (statisticIndex, classificationStatistics) -> {
					if (statisticIndex == 0) {
						return classificationStatistics.getOverallAccuracy();
					} else if (statisticIndex == 1) {
						return classificationStatistics.getMainModelAccuracy();
					} else if (statisticIndex == 2) {
						return classificationStatistics.getDefaultModelAccuracy(); //TODO: get both default models
					} else if (statisticIndex == 3) {
						return classificationStatistics.getDefaultClassAccuracy(); //TODO: get both default models
					} else if (statisticIndex == 4) {
						return classificationStatistics.getDefaultClassifierAccuracy(); //TODO: get both default models
					} else {
						throw new InvalidValueException("Wrong statistic index.");
					}
				};
				
				List<MeanAndStandardDeviation> meansAndStdDevList = new ArrayList<MeanAndStandardDeviation>(3);
				int n = classificationStatisticsSet.length;
				
				for (int statisticIndex = 0; statisticIndex < averagedStatisticsCount; statisticIndex++) {
					double sumAccuracies = 0.0;
					List<Double> accuracies = new ArrayList<Double>();
					
					for (int i = 0; i < n; i++) {
						sumAccuracies += modelIndex2Accuracy.apply(statisticIndex, classificationStatisticsSet[i]);
						accuracies.add(modelIndex2Accuracy.apply(statisticIndex, classificationStatisticsSet[i]));
					}
					
					double average = 0.0;
					double stdDev = 0.0;
					
					if (n >= 1) {
						average = sumAccuracies / n;
						if (n > 1) {
							final double streamAverage = average;
							stdDev = Math.sqrt(((double)1 / (n - 1)) * accuracies.stream().map(a -> Math.pow(a - streamAverage, 2)).collect(Collectors.summingDouble(x -> x))); //divide by (n-1)!
						}
					}
					
					meansAndStdDevList.add(new MeanAndStandardDeviation(average, stdDev)); //TODO: store average evaluations in these statistics
				} //for
				
				meansAndStdDevs = new MeansAndStandardDeviations(meansAndStdDevList.get(0), meansAndStdDevList.get(1), meansAndStdDevList.get(2), meansAndStdDevList.get(3), meansAndStdDevList.get(4)); // TODO: extend
			}
		}
		
		public long getMainModelCorrectCount() {
			return preciseCorrectCount + resolvingConflictCorrectCount;
		}
		
		public void increaseMainModelCorrectCount(int count) {
			if (classifierType == ClassifierType.WEKA_CLASSIFIER) {
				preciseCorrectCount += count;
			} else {
				throw new UnsupportedOperationException("Increasing main model correct count is only meant for a WEKA classifier.");
			}
		}
		
		public long getMainModelIncorrectCount() {
			return preciseIncorrectCount + resolvingConflictIncorrectCount;
		}
		
		public void increaseMainModelIncorrectCount(int count) {
			if (classifierType == ClassifierType.WEKA_CLASSIFIER) {
				preciseIncorrectCount += count;
			} else {
				throw new UnsupportedOperationException("Increasing main model incorrect count is only meant for a WEKA classifier.");
			}
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
		
		public double getOverallAccuracy() { //gets overall accuracy
			long allClassifiedObjectsCount = getCorrectCount() + getIncorrectCount();
			return allClassifiedObjectsCount > 0L ? (double)getCorrectCount() / allClassifiedObjectsCount : 0.0;
		}
		
		public double getMainModelAccuracy() { //gets accuracy concerning the part of validation data for which main model was applied to classify objects
			long mainModelCount = getMainModelCount();
			return mainModelCount > 0 ? (double)getMainModelCorrectCount() / mainModelCount : 0.0;
		}
		
		public double getPreciseAccuracy() {
			long preciseCount = getPreciseCount();
			return preciseCount > 0 ? (double)getPreciseCorrectCount() / preciseCount : 0.0;
		}
		
		public double getResolvingConflictAccuracy() {
			long resolvingConflictCount = getResolvingConflictCount();
			return resolvingConflictCount > 0 ? (double)getResolvingConflictCorrectCount() / resolvingConflictCount : 0.0;
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
			return totalNumberOfClassifiedObjects > 0L ? (double)getMainModelCount() / totalNumberOfClassifiedObjects : 0.0;
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
		
		public long getOriginalDecisionsConsistentTestObjectsTotalCount() {
			return originalDecisionsConsistentTestObjectsTotalCount;
		}

		public long getAssignedDefaultClassDecisionsConsistentTestObjectsTotalCount() {
			return assignedDefaultClassDecisionsConsistentTestObjectsTotalCount;
		}

		public long getAssignedDecisionsConsistentTestObjectsTotalCount() {
			return assignedDecisionsConsistentTestObjectsTotalCount;
		}
		
		public double getAverageOriginalDecisionsConsistentTestObjectsTotalCount() { //gets average quality of classification over test data for original decisions
			if (originalDecisionsConsistentTestObjectsTotalCount >= 0L) {
				return totalNumberOfClassifiedObjects > 0L ? ((double)originalDecisionsConsistentTestObjectsTotalCount / totalNumberOfClassifiedObjects) : 0.0;
			} else {
				return -1.0; //not being calculated
			}
		}
		
		public double getAverageAssignedDefaultClassDecisionsConsistentTestObjectsTotalCount() { //gets average quality of classification over test data for assigned decisions, using default class when no rule matches object
			if (assignedDefaultClassDecisionsConsistentTestObjectsTotalCount >= 0L) {
				return totalNumberOfClassifiedObjects > 0L ? (double)assignedDefaultClassDecisionsConsistentTestObjectsTotalCount / totalNumberOfClassifiedObjects : 0.0;
			} else {
				return -1.0; //not being calculated
			}
		}
		
		public double getAverageAssignedDecisionsConsistentTestObjectsTotalCount() { //gets average quality of classification over test data for assigned decisions, using default model when no rule matches object
			if (assignedDecisionsConsistentTestObjectsTotalCount >= 0L) {
				return totalNumberOfClassifiedObjects > 0L ? (double)assignedDecisionsConsistentTestObjectsTotalCount / totalNumberOfClassifiedObjects : 0.0;
			} else {
				return -1.0; //not being calculated
			}
		}
		
		public DefaultClassificationType getDefaultClassificationType () {
			return defaultClassificationType;
		}
		
		public ClassifierType getClassifierType() {
			return classifierType;
		}
		
		public AggregationMode getAggregationMode() {
			return aggregationMode;
		}

		public MeansAndStandardDeviations getMeansAndStandardDeviations() { //can be null, depending on aggregationMode
			return meansAndStdDevs;
		}

		public String toString() {
			if (classifierType == ClassifierType.VCDRSA_RULES_CLASSIFIER) {
				StringBuilder sb = new StringBuilder(256);
				
				double preciseClassificationPercentage = 100 * ( (double)getPreciseCount() / totalNumberOfClassifiedObjects );
				double correctPreciseClassificationPercentage = 100 * ( (double)preciseCorrectCount / totalNumberOfClassifiedObjects );
				double modeClassificationPercentage = 100 * ( (double)getResolvingConflictCount() / totalNumberOfClassifiedObjects );
				double correctModeClassificationPercentage = 100 * ( (double)resolvingConflictCorrectCount / totalNumberOfClassifiedObjects );
				double defaultClassClassificationPercentage = 100 * ( (double)getDefaultClassCount()/ totalNumberOfClassifiedObjects );
				double correctDefaultClassClassificationPercentage = 100 * ( (double)defaultClassCorrectCount / totalNumberOfClassifiedObjects );
				double defaultClassifierClassificationPercentage = 100 * ( (double)getDefaultClassifierCount() / totalNumberOfClassifiedObjects );
				double correctDefaultClassifierClassificationPercentage = 100 * ( (double)defaultClassifierCorrectCount / totalNumberOfClassifiedObjects );
				double accuracy = 100 * ( (double)getCorrectCount() / totalNumberOfClassifiedObjects ); //ordinalMisclassificationMatrix.getAccuracy();
				double accuracyWhenClassifiedByRules = 100 * getMainModelAccuracy(); //divide by the number of objects not-classified to the default class
				double accuracyWhenClassifiedByRulesPrecise = 100 * getPreciseAccuracy(); //accuracy when classified by precise rules
				double accuracyWhenClassifiedByRulesResolvingConflict = 100 * getResolvingConflictAccuracy();  //accuracy when classified after resolving conflict
				double accuracyWhenClassifiedByDefaultClass = 100 * getDefaultClassAccuracy();
				double accuracyWhenClassifiedByDefaultClassifier = 100 * getDefaultClassifierAccuracy();
				double avgNumberOfCoveringRules = getAverageNumberOfCoveringRules();
				
				sb.append("[Testing]: ");
				sb.append(String.format(Locale.US, "precise: %.2f%% (%.2f%% hit)", preciseClassificationPercentage, correctPreciseClassificationPercentage));
				sb.append(String.format(Locale.US, ", mode: %.2f%% (%.2f%% hit)", modeClassificationPercentage, correctModeClassificationPercentage));
				sb.append(String.format(Locale.US, ", default class: %.2f%% (%.2f%% hit)", defaultClassClassificationPercentage, correctDefaultClassClassificationPercentage));
				sb.append(String.format(Locale.US, ", default classifier: %.2f%% (%.2f%% hit);", defaultClassifierClassificationPercentage, correctDefaultClassifierClassificationPercentage));
				sb.append(String.format(Locale.US, "%n[Testing]: "));
				sb.append(String.format(Locale.US, "by rules: %.2f%% r.hit", accuracyWhenClassifiedByRules)); //accuracy among objects covered by 1+ rule
				sb.append(accuracyWhenClassifiedByRules > accuracy ? " [UP]" : " [!UP]");
				sb.append(String.format(Locale.US, " (precise: %.2f%% r.hit", accuracyWhenClassifiedByRulesPrecise)); //accuracy among objects covered by 1+ rule(s) of the same type (at least or at most)
				sb.append(String.format(Locale.US, ", mode: %.2f%% r.hit)", accuracyWhenClassifiedByRulesResolvingConflict)); //accuracy among objects covered by 1+ rule(s) of different types (at least and at most)
				sb.append(String.format(Locale.US, ", by default class: %.2f%% r.hit", accuracyWhenClassifiedByDefaultClass)); //accuracy among objects not covered by any rule
				sb.append(String.format(Locale.US, ", by default classifier: %.2f%% r.hit;", accuracyWhenClassifiedByDefaultClassifier)); //accuracy among objects not covered by any rule
				sb.append(String.format(Locale.US, "%n[Testing]: "));
				sb.append(String.format(Locale.US, ModeRuleClassifier.avgNumberOfCoveringRulesIndicator+": %.2f;", avgNumberOfCoveringRules));
				sb.append(String.format(Locale.US, "%n[Testing]: "));
				
				String qualitiesOfApproximation = getQualitiesOfApproximation();
				if (!qualitiesOfApproximation.equals("")) {
					//sb.append("; ");
					sb.append(getQualitiesOfApproximation());
				} else {
					//do nothing
				}
				
				sb.append(".");
				
				return sb.toString();
			} else { //WEKA_CLASSIFIER
				StringBuilder sb = new StringBuilder(128);
				String qualitiesOfApproximation = getQualitiesOfApproximation();
				
				sb.append("[Testing]: ");
				
				
				if (!qualitiesOfApproximation.equals("")) {
					sb.append(qualitiesOfApproximation);
				} else {
					sb.append("--");
				}
				
				sb.append(".");
				
				return sb.toString();
			}
		}
		
		public String getQualitiesOfApproximation() {
			double originalDecisionsQualityOfApproximation = getAverageOriginalDecisionsConsistentTestObjectsTotalCount();
			double assignedDefaultClassDecisionsQualityOfApproximation = getAverageAssignedDefaultClassDecisionsConsistentTestObjectsTotalCount();
			double assignedDecisionsQualityOfApproximation = getAverageAssignedDecisionsConsistentTestObjectsTotalCount();
			
			StringBuilder sb = new StringBuilder(128);
			sb.append("");
			
			if (originalDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, "%soriginal quality: %s", aggregationMode != AggregationMode.NONE ? "avg. " : "",
						BatchExperiment.round(originalDecisionsQualityOfApproximation)));
			}
			if (assignedDefaultClassDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, "; %sassigned default class quality: %s", aggregationMode != AggregationMode.NONE ? "avg. " : "",
						BatchExperiment.round(assignedDefaultClassDecisionsQualityOfApproximation)));
			}
			if (assignedDecisionsQualityOfApproximation >= 0.0) {
				sb.append(String.format(Locale.US, ", %sassigned quality: %s", aggregationMode != AggregationMode.NONE ? "avg. " : "",
						BatchExperiment.round(assignedDecisionsQualityOfApproximation)));
			}
			
			return sb.toString();
		}
		
	}
	
	OrdinalMisclassificationMatrix ordinalMisclassificationMatrix;
	ClassificationStatistics classificationStatistics;
	ModelLearningStatistics modelLearningStatistics;
	ModelDescription modelDescription;
	
	AggregationMode aggregationMode = AggregationMode.NONE;
	Decision[] orderOfDecisions = null;
	
	public ModelValidationResult(OrdinalMisclassificationMatrix ordinalMisclassificationMatrix,
			ClassificationStatistics classificationStatistics,
			ModelLearningStatistics modelLearningStatistics,
			ModelDescription modelDescription) {
		
		this.ordinalMisclassificationMatrix = ordinalMisclassificationMatrix;
		this.classificationStatistics = classificationStatistics;
		this.modelLearningStatistics = modelLearningStatistics;
		this.modelDescription = modelDescription;
	}
	
	public ModelValidationResult(AggregationMode aggregationMode, Decision[] orderOfDecisions, ModelValidationResult... modelValidationResults) {
		if (aggregationMode == null || aggregationMode == AggregationMode.NONE) {
			throw new InvalidValueException("Incorrect aggregation mode.");
		}
		
		ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(aggregationMode == AggregationMode.SUM, orderOfDecisions,
				Arrays.asList(modelValidationResults).stream().map(m -> m.getOrdinalMisclassificationMatrix()).collect(Collectors.toList()).toArray(new OrdinalMisclassificationMatrix[0]));
		classificationStatistics = new ClassificationStatistics(aggregationMode, Arrays.asList(modelValidationResults).stream().map(m -> m.getClassificationStatistics()).collect(Collectors.toList()).toArray(new ClassificationStatistics[0]));
		modelLearningStatistics = new ModelLearningStatistics(aggregationMode, Arrays.asList(modelValidationResults).stream().map(m -> m.getModelLearningStatistics()).collect(Collectors.toList()).toArray(new ModelLearningStatistics[0]));
		ModelDescription[] modelDescriptions = Arrays.asList(modelValidationResults).stream().map(m -> m.getModelDescription()).collect(Collectors.toList()).toArray(new ModelDescription[0]);
		modelDescription = modelDescriptions[0].getModelDescriptionBuilder().build(aggregationMode, modelDescriptions);
		
		this.aggregationMode = aggregationMode;
		this.orderOfDecisions = orderOfDecisions;
	}

	public OrdinalMisclassificationMatrix getOrdinalMisclassificationMatrix() {
		return ordinalMisclassificationMatrix;
	}
	
	public ClassificationStatistics getClassificationStatistics() {
		return classificationStatistics;
	}
	
	public ModelLearningStatistics getModelLearningStatistics() {
		return modelLearningStatistics;
	}
	
	public ModelDescription getModelDescription() {
		return modelDescription;
	}
	
	public AggregationMode getAggregationMode() {
		return aggregationMode;
	}

	public Decision[] getOrderOfDecisions() {
		return orderOfDecisions;
	}

}

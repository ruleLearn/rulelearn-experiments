/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
import org.rulelearn.core.InvalidValueException;
import org.rulelearn.data.Decision;
import org.rulelearn.data.InformationTable;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.measures.dominance.EpsilonConsistencyMeasure;

/**
 * Classification model learned from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface ClassificationModel {
	
	static int getNumberOfConsistentObjects(InformationTable informationTable, double consistencyThreshold) {
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = (informationTable instanceof InformationTableWithDecisionDistributions ?
				(InformationTableWithDecisionDistributions)informationTable : new InformationTableWithDecisionDistributions(informationTable, true));
		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
		
		return unions.getNumberOfConsistentObjects();
	}
	
	/**
	 * Gets number of consistent objects in the given information table, for given threshold, assuming objects from given information table have given decisions.
	 * 
	 * @param informationTable tested information table
	 * @param decisions new decisions for subsequent objects from given information table
	 * @param consistencyThreshold consistency threshold for the calculation of the quality of approximation of classification
	 * 
	 * @return the number of consistent objects in the given information table, for given threshold, assuming objects from given information table have given decisions
	 */
	static int getNumberOfConsistentObjects(InformationTable informationTable, Decision[] decisions, double consistencyThreshold) {
		InformationTable informationTableWithAssignedDecisions = new InformationTable(informationTable, decisions, true);
		return getNumberOfConsistentObjects(informationTableWithAssignedDecisions, consistencyThreshold);
	}
	
	public abstract class ModelDescriptionBuilder {
		abstract ModelDescription build(AggregationMode aggregationMode, ModelDescription... modelDescriptions); //builds new model description from given array of model descriptions
	}
	
	public abstract class ModelDescription {
		public abstract String toString();
		public abstract String toShortString(); //one line model description
		public abstract ModelDescriptionBuilder getModelDescriptionBuilder();
	}
	
	public static class ModelLearningStatistics {
		int totalNumberOfLearningObjects = 0; //total number of learning objects
		int totalNumberOfConsistentLearningObjects = 0; //for epsilon consistency threshold 0.0
		double consistencyThreshold = -1.0; //<0.0 if not used
		int totalNumberOfConsistentLearningObjectsForConsistencyThreshold = -1; //< 0 if not used
		String modelLearnerDescription = null;
		long totalStatisticsCountingTime = 0L; //to be subtracted from model learning time
		
		int aggregationCount = 0; //tells how many ModelLearningStatistics objects have been used to build this object
		AggregationMode aggregationMode = AggregationMode.NONE;
		
		public ModelLearningStatistics(int numberOfLearningObjects, int numberOfConsistentLearningObjects,
				double consistencyThreshold, int numberOfConsistentLearningObjectsForConsistencyThreshold,
				String modelLearnerDescription, long statisticsCountingTime) {
			this.totalNumberOfLearningObjects = numberOfLearningObjects;
			this.totalNumberOfConsistentLearningObjects = numberOfConsistentLearningObjects;
			this.consistencyThreshold = consistencyThreshold;
			this.totalNumberOfConsistentLearningObjectsForConsistencyThreshold = numberOfConsistentLearningObjectsForConsistencyThreshold;
			this.modelLearnerDescription = modelLearnerDescription;
			this.totalStatisticsCountingTime = statisticsCountingTime;
			
			aggregationCount = 1;
			aggregationMode = AggregationMode.NONE;
		}
		
		public ModelLearningStatistics(AggregationMode aggregationMode, ModelLearningStatistics... modelLearningStatisticsSet) { //assumes presence of at least one statistics
			if (aggregationMode == null || aggregationMode == AggregationMode.NONE) {
				throw new InvalidValueException("Incorrect aggregation mode.");
			}
			
			int numberOfConsistentLearningObjectsForConsistencyThresholdSum = 0;
			boolean numberOfConsistentLearningObjectsForConsistencyThresholdIsUsed = false;
			
			//calculate sums
			for (ModelLearningStatistics modelLearningStatistics : modelLearningStatisticsSet) {
				totalNumberOfLearningObjects += modelLearningStatistics.totalNumberOfLearningObjects;
				totalNumberOfConsistentLearningObjects += modelLearningStatistics.totalNumberOfConsistentLearningObjects;
				if (modelLearningStatistics.totalNumberOfConsistentLearningObjectsForConsistencyThreshold >= 0) {
					numberOfConsistentLearningObjectsForConsistencyThresholdSum += modelLearningStatistics.totalNumberOfConsistentLearningObjectsForConsistencyThreshold;
					numberOfConsistentLearningObjectsForConsistencyThresholdIsUsed = true;
				}
				totalStatisticsCountingTime += modelLearningStatistics.totalStatisticsCountingTime;
				
				aggregationCount += modelLearningStatistics.aggregationCount;
			}
			
			if (numberOfConsistentLearningObjectsForConsistencyThresholdIsUsed) {
				totalNumberOfConsistentLearningObjectsForConsistencyThreshold = numberOfConsistentLearningObjectsForConsistencyThresholdSum;
			}
			
			this.consistencyThreshold = modelLearningStatisticsSet[0].consistencyThreshold; //all statistics should be for the same consistency threshold
			this.modelLearnerDescription = modelLearningStatisticsSet[0].modelLearnerDescription; //all statistics should be for the same model learner
			
			this.aggregationMode = aggregationMode;
			
			if (this.aggregationMode == AggregationMode.MEAN_AND_DEVIATION) {
				//TODO: calculate means and standard deviations
			}
		}

		public long getTotalNumberOfLearningObjects() {
			return totalNumberOfLearningObjects;
		}

		public long getTotalNumberOfConsistentLearningObjects() {
			return totalNumberOfConsistentLearningObjects;
		}
		
		public double getConsistencyThreshold() {
			return consistencyThreshold;
		}

		public int getTotalNumberOfConsistentLearningObjectsForConsistencyThreshold() {
			return totalNumberOfConsistentLearningObjectsForConsistencyThreshold;
		}

		public String getModelLearnerDescription() {
			return modelLearnerDescription;
		}

		public long getTotalStatisticsCountingTime() {
			return totalStatisticsCountingTime;
		}
		
		public AggregationMode getAggregationMode() {
			return aggregationMode;
		}
		
		public double getAverageNumberOfConsistentLearningObjects() { //gets average quality of classification over train data
			return totalNumberOfLearningObjects > 0L ? ((double)totalNumberOfConsistentLearningObjects / totalNumberOfLearningObjects) : 0.0;
		}
		
		public double getAverageNumberOfConsistentLearningObjectsForConsistencyThreshold() { //gets average quality of classification over train data
			if (totalNumberOfConsistentLearningObjectsForConsistencyThreshold >= 0) {
				return totalNumberOfLearningObjects > 0L ? ((double)totalNumberOfConsistentLearningObjectsForConsistencyThreshold / totalNumberOfLearningObjects) : 0.0;
			} else {
				return -1.0; //not being calculated
			}
		}
		
		public String toString() { //TODO: if aggregationMode == AggregationMode.MEAN_AND_DEVIATION, then print also standard deviations calculated in constructor
			StringBuilder sb = new StringBuilder(100);
			
			String roundedValue = BatchExperiment.round(getAverageNumberOfConsistentLearningObjects());
			if (aggregationCount == 1) {
				sb.append("DRSA quality of classification: ").append(roundedValue);
			} else {
				sb.append("Avg. DRSA quality of classification: ").append(roundedValue);
			}

			double averageNumberOfConsistentLearningObjectsForConsistencyThreshold = getAverageNumberOfConsistentLearningObjectsForConsistencyThreshold();
			
			if (averageNumberOfConsistentLearningObjectsForConsistencyThreshold >= 0) { //used
				roundedValue = BatchExperiment.round(averageNumberOfConsistentLearningObjectsForConsistencyThreshold);
				
				if (aggregationCount == 1) {
					sb.append(", VC-DRSA quality of classification for epsilon="+consistencyThreshold+": ").append(roundedValue);
				} else {
					sb.append(", avg. VC-DRSA quality of classification for epsilon="+consistencyThreshold+": ").append(roundedValue);
				}
			}
			
			return sb.toString();
		}

	}
	
	ModelValidationResult validate(Data testData);
	SimpleDecision classify(int i, Data data); //gets simple decision of a single object from data
	ModelDescription getModelDescription();
	ModelLearningStatistics getModelLearningStatistics();
}

/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.approximations.Unions;
import org.rulelearn.approximations.UnionsWithSingleLimitingDecision;
import org.rulelearn.approximations.VCDominanceBasedRoughSetCalculator;
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
	
//	default double getQualityOfApproximation(InformationTable informationTable, double consistencyThreshold) {
//		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = (informationTable instanceof InformationTableWithDecisionDistributions ?
//				(InformationTableWithDecisionDistributions)informationTable : new InformationTableWithDecisionDistributions(informationTable, true));
//		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
//		return unions.getQualityOfApproximation();
//	}
//	
//	/**
//	 * Gets quality of approximation of classification, for given threshold, assuming objects from given information table have given decisions.
//	 * 
//	 * @param informationTable tested information table
//	 * @param decisions new decisions for subsequent objects from given information table
//	 * @param consistencyThreshold consistency threshold for the calculation of the quality of approximation of classification
//	 * 
//	 * @return the quality of approximation of classification, for given threshold, assuming objects from given information table have given decisions
//	 */
//	default double getQualityOfApproximationForDecisions(InformationTable informationTable, Decision[] decisions, double consistencyThreshold) {
//		InformationTable informationTableWithAssignedDecisions = new InformationTable(informationTable, decisions, true);
//		return getQualityOfApproximation(informationTableWithAssignedDecisions, consistencyThreshold);
//	}
	
	default int getNumberOfConsistentObjects(InformationTable informationTable, double consistencyThreshold) {
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
	default int getNumberOfConsistentObjects(InformationTable informationTable, Decision[] decisions, double consistencyThreshold) {
		InformationTable informationTableWithAssignedDecisions = new InformationTable(informationTable, decisions, true);
		return getNumberOfConsistentObjects(informationTableWithAssignedDecisions, consistencyThreshold);
	}
	
//	public abstract class ValidationSummary {
//		public abstract String toString();
//	}
	
	public abstract class ModelDescriptionBuilder {
		abstract ModelDescription build(AggregationMode aggregationMode, ModelDescription... modelDescriptions); //builds new model description from given array of model descriptions
	}
	
	public abstract class ModelDescription {
		public abstract String toString();
		public abstract ModelDescriptionBuilder getModelDescriptionBuilder();
	}
	
	ModelValidationResult validate(Data testData);
	SimpleDecision classify(int i, Data data); //gets simple decision of a single object from data
//	ValidationSummary getValidationSummary();
	ModelDescription getModelDescription();
	String getModelLearnerDescription();
}

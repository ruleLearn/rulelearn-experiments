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
	
	default double getQualityOfApproximation(InformationTable informationTable, double consistencyThreshold) {
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = (informationTable instanceof InformationTableWithDecisionDistributions ?
				(InformationTableWithDecisionDistributions)informationTable : new InformationTableWithDecisionDistributions(informationTable, true));
		
		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
		
		return unions.getQualityOfApproximation();
	}
	
	/**
	 * Gets quality of approximation of classification, for given threshold, assuming objects from given information table have given decisions.
	 * 
	 * @param informationTable tested information table
	 * @param decisions new decisions for subsequent objects from given information table
	 * @param consistencyThreshold consistency threshold for the calculation of the quality of approximation of classification
	 * 
	 * @return quality of approximation of classification, for given threshold, assuming objects from given information table have given decisions
	 */
	default double getQualityOfApproximationForDecisions(InformationTable informationTable, Decision[] decisions, double consistencyThreshold) {
		InformationTable informationTableWithAssignedDecisions = new InformationTable(informationTable, decisions, true);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributions = new InformationTableWithDecisionDistributions(informationTableWithAssignedDecisions, true);
		
//		InformationTableWriter informationTableWriter = new InformationTableWriter(true);
//		try (FileWriter fileWriter = new FileWriter("src/main/resources/data/json-objects/bank-churn-4000-v8_0.05-assigned-decisions data.json")) {
//			informationTableWriter.writeObjects(informationTableWithAssignedDecisions, fileWriter);
//			System.out.println("JSON objects file written.");
//		}
//		catch (IOException exception) {
//			exception.printStackTrace();
//		}
		
//		int size = informationTableWithAssignedDecisions.getNumberOfObjects();
//		for (int i = 0; i < size; i++) {
//			if ( ((EnumerationField)((SimpleDecision)informationTableWithAssignedDecisions.getDecision(i)).getEvaluation()).getElement().equals("0") ) { //object from better class
//				IntSortedSet sortedSet = DominanceConeCalculator.INSTANCE.calculatePositiveDCone(i, informationTableWithAssignedDecisions);
//				System.out.print((i+1)+": ");
//				sortedSet.forEach(objIndex -> {
//					if ( ((EnumerationField)((SimpleDecision)informationTableWithAssignedDecisions.getDecision(objIndex)).getEvaluation()).getElement().equals("1") ) { //object from worse class
//						System.out.print((objIndex + 1)+" ");
//					}
//				});
//				System.out.println();
//			}
//		}
		
		Unions unions = new UnionsWithSingleLimitingDecision(informationTableWithDecisionDistributions, new VCDominanceBasedRoughSetCalculator(EpsilonConsistencyMeasure.getInstance(), consistencyThreshold));
		
		return unions.getQualityOfApproximation();
	}
	
	public abstract class ValidationSummary {
		public abstract String toString();
	}
	
	public abstract class ModelDescriptionBuilder {
		abstract ModelDescription build(ModelDescription... modelDescriptions); //builds new model description from given array of model descriptions
	}
	
	public abstract class ModelDescription {
		public abstract String toString();
		public abstract ModelDescriptionBuilder getModelDescriptionBuilder();
	}
	
	ModelValidationResult validate(Data testData);
	SimpleDecision classify(int i, Data data); //gets simple decision of a single object from data
	ValidationSummary getValidationSummary();
	ModelDescription getModelDescription();
	String getModelLearnerDescription();
}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.rulelearn.data.Decision;
import org.rulelearn.experiments.ClassificationModel.ModelDescription;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Compound model validation result, extending over {@link OrdinalMisclassificationMatrix}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ModelValidationResult {
	
	OrdinalMisclassificationMatrix ordinalMisclassificationMatrix;
	
	//total numbers of decisions
	long numberOfCorrectDecisionsAssignedByMainModel = 0L; //concerns the part of validation data for which main model was applied to classify objects (e.g., covering decision rule(s))
	long numberOfAllDecisionsAssignedByMainModel = 0L;
	long numberOfCorrectDecisionsAssignedByDefaultModel = 0L; //concerns the part of validation data for which default model was applied to classify objects (e.g., assigning default decision class)
	long numberOfAllDecisionsAssignedByDefaultModel = 0L; //can be 0 if default model is not used (which is the case when the main model classifies all objects itself)
	
	ModelDescription modelDescription;
	
	//in case of single validation (on a validation set) this is a sum of the numbers of covering rules over all classified objects;
	//e.g., if there are two objects, first covered by 3 rules, and second covered by 4 rules, then total number of covering rules is 7;
	//in case of aggregated model validation result, this is the sum of total number of covering rules over all validated models
	long totalNumberOfCoveringRules = 0L;
	long totalNumberOfClassifiedObjects = 0L;
	
	public ModelValidationResult(OrdinalMisclassificationMatrix ordinalMisclassificationMatrix,
			long numberOfCorrectDecisionsAssignedByMainModel, long numberOfAllDecisionsAssignedByMainModel,
			long numberOfCorrectDecisionsAssignedByDefaultModel, long numberOfAllDecisionsAssignedByDefaultModel,
			ModelDescription modelDescription,
			long totalNumberOfCoveringRules, long totalNumberOfClassifiedObjects) {
		
		this.ordinalMisclassificationMatrix = ordinalMisclassificationMatrix;
		
		this.numberOfCorrectDecisionsAssignedByMainModel = numberOfCorrectDecisionsAssignedByMainModel;
		this.numberOfAllDecisionsAssignedByMainModel = numberOfAllDecisionsAssignedByMainModel;
		this.numberOfCorrectDecisionsAssignedByDefaultModel = numberOfCorrectDecisionsAssignedByDefaultModel;
		this.numberOfAllDecisionsAssignedByDefaultModel = numberOfAllDecisionsAssignedByDefaultModel;
		
		this.modelDescription = modelDescription;
		
		this.totalNumberOfCoveringRules = totalNumberOfCoveringRules; //should be zero for classifiers not using rules
		this.totalNumberOfClassifiedObjects = totalNumberOfClassifiedObjects;
	}
	
	public ModelValidationResult(Decision[] orderOfDecisions, ModelValidationResult... modelValidationResults) {
		ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions,
				Arrays.asList(modelValidationResults).stream().map(m -> m.getOrdinalMisclassificationMatrix()).collect(Collectors.toList()).toArray(new OrdinalMisclassificationMatrix[0]));
		
		ModelDescription[] modelDescriptions = Arrays.asList(modelValidationResults).stream().map(m -> m.getModelDescription()).collect(Collectors.toList()).toArray(new ModelDescription[0]);
		modelDescription = modelDescriptions[0].getModelDescriptionBuilder().build(modelDescriptions);
		
		//just do summation, like in MisclassificationMatrix
		for (ModelValidationResult modelValidationResult : modelValidationResults) {
			numberOfCorrectDecisionsAssignedByMainModel += modelValidationResult.numberOfCorrectDecisionsAssignedByMainModel;
			numberOfAllDecisionsAssignedByMainModel += modelValidationResult.numberOfAllDecisionsAssignedByMainModel;
			numberOfCorrectDecisionsAssignedByDefaultModel += modelValidationResult.numberOfCorrectDecisionsAssignedByDefaultModel;
			numberOfAllDecisionsAssignedByDefaultModel += modelValidationResult.numberOfAllDecisionsAssignedByDefaultModel;
			
			totalNumberOfCoveringRules += modelValidationResult.totalNumberOfCoveringRules;
			totalNumberOfClassifiedObjects += modelValidationResult.totalNumberOfClassifiedObjects;
		}
	}

	public OrdinalMisclassificationMatrix getOrdinalMisclassificationMatrix() {
		return ordinalMisclassificationMatrix;
	}
	
	public double getOverallAccuracy() {
		return ordinalMisclassificationMatrix.getAccuracy();
	}
	
	public double getMainModelAccuracy() { //gets accuracy concerning the part of validation data for which main model was applied to classify objects
		return numberOfAllDecisionsAssignedByMainModel > 0 ? (double)numberOfCorrectDecisionsAssignedByMainModel / numberOfAllDecisionsAssignedByMainModel : 0.0;
	}
	
	public double getDefaultModelAccuracy() { //gets accuracy concerning the part of validation data for which default model was applied to classify objects
		return numberOfAllDecisionsAssignedByDefaultModel > 0 ? (double)numberOfCorrectDecisionsAssignedByDefaultModel / numberOfAllDecisionsAssignedByDefaultModel : 0.0;
	}
	
	public double getMainModelDecisionsRatio() { //percent of situations when main model suggested decision
		return (double)numberOfAllDecisionsAssignedByMainModel / (numberOfAllDecisionsAssignedByMainModel + numberOfAllDecisionsAssignedByDefaultModel);
	}

//	public long getNumberOfCorrectDecisionsAssignedByMainModel() {
//		return numberOfCorrectDecisionsAssignedByMainModel;
//	}

	public long getNumberOfAllDecisionsAssignedByMainModel() {
		return numberOfAllDecisionsAssignedByMainModel;
	}

//	public long getNumberOfCorrectDecisionsAssignedByDefaultModel() {
//		return numberOfCorrectDecisionsAssignedByDefaultModel;
//	}

	public long getNumberOfAllDecisionsAssignedByDefaultModel() {
		return numberOfAllDecisionsAssignedByDefaultModel;
	}
	
	public ModelDescription getModelDescription() {
		return modelDescription;
	}
	
	public long getTotalNumberOfCoveringRules() {
		return totalNumberOfCoveringRules;
	}

	public long getTotalNumberOfClassifiedObjects() {
		return totalNumberOfClassifiedObjects;
	}

//	public double getAvgNumberOfCoveringRules() {
//		return (double)totalNumberOfCoveringRules / totalNumberOfClassifiedObjects;
//	}

}

/**
 * 
 */
package org.rulelearn.experiments;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.rulelearn.data.Decision;
import org.rulelearn.validation.OrdinalMisclassificationMatrix;

/**
 * Compound model validation result, extending over {@link OrdinalMisclassificationMatrix}.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ModelValidationResult {
	
	OrdinalMisclassificationMatrix ordinalMisclassificationMatrix;
	
	long numberOfCorrectDecisionsAssignedByMainModel = 0; //concerns the part of validation data for which main model was applied to classify objects (e.g., covering decision rule(s))
	long numberOfAllDecisionsAssignedByMainModel = 0;
	long numberOfCorrectDecisionsAssignedByDefaultModel = 0; //concerns the part of validation data for which default model was applied to classify objects (e.g., assigning default decision class)
	long numberOfAllDecisionsAssignedByDefaultModel = 0; //can be 0 if default model is not used (which is the case when the main model classifies all objects itself)
	
	public ModelValidationResult(OrdinalMisclassificationMatrix ordinalMisclassificationMatrix,
			long numberOfCorrectDecisionsAssignedByMainModel, long numberOfAllDecisionsAssignedByMainModel,
			long numberOfCorrectDecisionsAssignedByDefaultModel, long numberOfAllDecisionsAssignedByDefaultModel) {
		
		this.ordinalMisclassificationMatrix = ordinalMisclassificationMatrix;
		this.numberOfCorrectDecisionsAssignedByMainModel = numberOfCorrectDecisionsAssignedByMainModel;
		this.numberOfAllDecisionsAssignedByMainModel = numberOfAllDecisionsAssignedByMainModel;
		this.numberOfCorrectDecisionsAssignedByDefaultModel = numberOfCorrectDecisionsAssignedByDefaultModel;
		this.numberOfAllDecisionsAssignedByDefaultModel = numberOfAllDecisionsAssignedByDefaultModel;
	}
	
	public ModelValidationResult(Decision[] orderOfDecisions, ModelValidationResult... modelValidationResults) {
		ordinalMisclassificationMatrix = new OrdinalMisclassificationMatrix(orderOfDecisions,
				Arrays.asList(modelValidationResults).stream().map(m -> m.getOrdinalMisclassificationMatrix()).collect(Collectors.toList()).toArray(new OrdinalMisclassificationMatrix[0]));

		//just do summation, like in MisclassificationMatrix
		for (ModelValidationResult modelValidationResult : modelValidationResults) {
			numberOfCorrectDecisionsAssignedByMainModel += modelValidationResult.numberOfCorrectDecisionsAssignedByMainModel;
			numberOfAllDecisionsAssignedByMainModel += modelValidationResult.numberOfAllDecisionsAssignedByMainModel;
			numberOfCorrectDecisionsAssignedByDefaultModel += modelValidationResult.numberOfCorrectDecisionsAssignedByDefaultModel;
			numberOfAllDecisionsAssignedByDefaultModel += modelValidationResult.numberOfAllDecisionsAssignedByDefaultModel;
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
}

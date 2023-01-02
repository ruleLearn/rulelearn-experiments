/**
 * 
 */
package org.rulelearn.experiments;

import org.rulelearn.data.SimpleDecision;

/**
 * Classification model learned from data.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public interface ClassificationModel {
	
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

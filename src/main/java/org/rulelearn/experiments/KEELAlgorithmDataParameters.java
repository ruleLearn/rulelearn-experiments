package org.rulelearn.experiments;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class KEELAlgorithmDataParameters implements LearningAlgorithmDataParameters {
	
	AttributeRanges attributeRanges;

	public KEELAlgorithmDataParameters(AttributeRanges attributeRanges) {
		this.attributeRanges = attributeRanges;
	}
	
	public AttributeRanges getAttributeRanges() {
		return attributeRanges;
	}

	@Override
	public String getParameter(String parameterName) {
		return null; //there are no parameters
	}
	
	public String toString() {
		return attributeRanges.toString();
	}
}

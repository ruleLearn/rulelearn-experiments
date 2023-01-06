package org.rulelearn.experiments;

public class MeanAndStandardDeviation {
	double average;
	double stdDev; //standard deviation
	
	public MeanAndStandardDeviation(double average) {
		this.average = average;
		this.stdDev = 0.0;
	}
	
	public MeanAndStandardDeviation(double average, double stdDev) {
		this.average = average;
		this.stdDev = stdDev;
	}
	
	public double getMean() {
		return average;
	}
	public double getStdDev() {
		return stdDev;
	}
}
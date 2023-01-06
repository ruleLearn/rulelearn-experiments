package org.rulelearn.experiments;

public enum AggregationMode {
	NONE, //no aggregation at all
	SUM, //only sum counters
	MEAN_AND_DEVIATION; //average counters + calculate variances / std devs.
}
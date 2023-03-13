/**
 * 
 */
package org.rulelearn.experiments.setup;

import java.util.ArrayList;
import java.util.List;

import org.rulelearn.experiments.DataProcessorProvider;
import org.rulelearn.experiments.DataProvider;
import org.rulelearn.experiments.LearningAlgorithm;
import org.rulelearn.experiments.LearningAlgorithmDataParametersContainer;
import org.rulelearn.experiments.MoNGELClassifierLearner;
import org.rulelearn.experiments.VCDomLEMModeRuleClassifierLearner;
import org.rulelearn.experiments.WEKAClassifierLearner;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.misc.OSDL;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OLM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public abstract class BatchExperimentSetup {
	
	protected long[] seeds;
	protected int k;
	DataProcessorProvider dataProcessorProvider;
	
	protected List<DataProvider> dataProviders = null;
	protected List<LearningAlgorithm> learningAlgorithms = null;
	protected LearningAlgorithmDataParametersContainer parametersContainer = null;
	
	public BatchExperimentSetup(long[] seeds, int k, DataProcessorProvider dataProcessorProvider) {
		this.seeds = seeds;
		this.k = k;
		this.dataProcessorProvider = dataProcessorProvider;
	}
	
	abstract public List<DataProvider> getDataProviders();
	abstract public List<LearningAlgorithm> getLearningAlgorithms();
	abstract public LearningAlgorithmDataParametersContainer getLearningAlgorithmDataParametersContainer();
	
	public DataProcessorProvider getDataProcessorProvider() {
		return dataProcessorProvider;
	}
	
	public List<LearningAlgorithm> getLearningAlgorithmsForOriginalData() {
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new VCDomLEMModeRuleClassifierLearner());
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new J48()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new NaiveBayes()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new SMO()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new RandomForest()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new MultilayerPerceptron()));
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new JRip()));
		return learningAlgorithms;
	}
	
	public List<LearningAlgorithm> getLearningAlgorithmsForOLM_OSDLData() {
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OLM())); //uses special version of data!
		learningAlgorithms.add(new WEKAClassifierLearner(() -> new OSDL())); //uses special version of data! //weka.core.UnsupportedAttributeTypeException: weka.classifiers.misc.OSDL: Cannot handle numeric attributes!
		return learningAlgorithms;
	}
	
	public List<LearningAlgorithm> getLearningAlgorithmsForMoNGELData() {
		List<LearningAlgorithm> learningAlgorithms = new ArrayList<LearningAlgorithm>();
		learningAlgorithms.add(new MoNGELClassifierLearner()); //uses special version of data!
		return learningAlgorithms;
	}
	
}

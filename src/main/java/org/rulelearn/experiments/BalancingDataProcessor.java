/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.rulelearn.core.InvalidValueException;
//import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.rulelearn.data.Decision;
import org.rulelearn.data.DecisionDistribution;
import org.rulelearn.data.InformationTableWithDecisionDistributions;

/**
 * Processes data to balance decision classes (using over-sampling, under-sampling, or simultaneous under- and over-sampling).
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BalancingDataProcessor implements DataProcessor {
	
	/**
	 * Strategy for balancing decision class distribution in data.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	public enum BalancingStrategy {
		/**
		 * Strategy involving under-sampling of larger classes (so all classes will have roughly the same number of objects), so the resulting set of objects will have smaller size.
		 */
		UNDERSAMPLING,
		/**
		 * Strategy involving over-sampling of smaller classes (so all classes will have roughly the same number of objects), so the resulting set of objects will have larger size.
		 */
		OVERSAMPLING,
		/**
		 * Strategy involving both under-sampling (for larger classes) and over-sampling (for smaller classes), so the resulting set of objects will have the same size.
		 */
		UNDER_AND_OVERSAMPLING
	}
	
	/**
	 * Helps to draw objects randomly.
	 * 
	 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
	 */
	final class DrawingMachine {
		
		private void swap(int[] objectIndices, int i, int j) {
			int temp = objectIndices[i];
			objectIndices[i] = objectIndices[j];
			objectIndices[j] = temp;
		}
		
		private int[] draw(int[] objectIndices, int numObjectsToDraw) {
			objectIndices = objectIndices.clone(); //ensure passed array remains unchanged
			
			int maxSubarrayIndex = objectIndices.length - 1;
			
			while (numObjectsToDraw > 0) { //in each iteration draws one index and moves it to the end of currently considered sub-array
				int subarrayIndex = random.nextInt(maxSubarrayIndex + 1); //0 incl. to maxTableIndex incl.
				swap(objectIndices, subarrayIndex, maxSubarrayIndex);
				maxSubarrayIndex--;
				numObjectsToDraw--;
			}
			
			int[] drawnObjectIndices = new int[objectIndices.length - (maxSubarrayIndex + 1)];
			
			int arrayIndex = 0;
			for (int i = maxSubarrayIndex + 1; i < objectIndices.length; i++) {
				drawnObjectIndices[arrayIndex++] = objectIndices[i];
			}
			
			return drawnObjectIndices;
		}
	} //end class DrawingMachine
	
	/**
	 * Source of randomness.
	 */
	Random random;
	
	/**
	 * Strategy of balancing decision classes distribution.
	 */
	private BalancingStrategy balancingStrategy;
//	private long seed;
	
	/**
	 * Constructor.
	 * 
	 * @param balancingStrategy
	 * @param seed
	 */
	public BalancingDataProcessor(BalancingStrategy balancingStrategy, long seed) {
		this.balancingStrategy = balancingStrategy;
//		this.seed = seed;
		this.random = new Random(seed);
	}
	
	public BalancingStrategy getBalancingStrategy() {
		return balancingStrategy;
	}

	@Override
	public Data process(Data data) {
		//EnumeratedIntegerDistribution distribution; //from Apache Commons Math library
		
		InformationTableWithDecisionDistributions informationTableWithDistributions = data.getInformationTable() instanceof InformationTableWithDecisionDistributions ?
				(InformationTableWithDecisionDistributions)data.getInformationTable() :
				new InformationTableWithDecisionDistributions(data.getInformationTable()); 
		
		int totalNumberOfObjects = informationTableWithDistributions.getNumberOfObjects();
		
		DecisionDistribution decisionDistribution = informationTableWithDistributions.getDecisionDistribution();
		Set<Decision> decisions = decisionDistribution.getDecisions();
		
		int minCount = totalNumberOfObjects + 1;
		int maxCount = 0;
		int newDecisionClassSize;
		
		for (Decision decision : decisions) {
			int count = decisionDistribution.getCount(decision);
			
			if (count < minCount) {
				minCount = count;
			}
			if (count > maxCount) {
				maxCount = count;
			}
		}
		
		switch (balancingStrategy) {
		case UNDERSAMPLING:
			newDecisionClassSize = minCount;
			break;
		case OVERSAMPLING:
			newDecisionClassSize = maxCount;
			break;
		case UNDER_AND_OVERSAMPLING:
			//may drop several objects (e.g., for 5004 objects and 10 classes, we get 500 objects from each class, so 4 objects will be dropped)
			newDecisionClassSize = totalNumberOfObjects / decisions.size(); //integer division by design (drops remainder)!
			break;
		default:
			throw new InvalidValueException("Not supported value of decision class distribution balancing strategy.");
		}
		
		int fullDecisionClassRounds = 0; //how many times all objects from current decision class should be taken (when over-sampling)
		List<int[]> perClassSelectedObjectIndicesList = new ArrayList<>(); //indices of objects selected for subsequent decision classes
		int newTotalNumberOfObjects = 0;
		
		for (Decision decision : decisions) {
			int decisionClassSize = decisionDistribution.getCount(decision);
			
			//gather indices of all objects from current decision class
			int[] decisionClassObjectIndices = new int[decisionClassSize];
			int arrayIndex = 0;
			for (int objectIndex = 0; objectIndex < totalNumberOfObjects; objectIndex++) {
				if (informationTableWithDistributions.getDecision(objectIndex).equals(decision)) {
					decisionClassObjectIndices[arrayIndex++] = objectIndex;
				}
			}
			
			//***** OVERSAMPLING *****
			if (decisionClassSize < newDecisionClassSize) { //do over-sampling of current decision class
				fullDecisionClassRounds = newDecisionClassSize / decisionClassSize; //integer division by design (drops remainder)!
				
				int[] drawnDecisionClassObjectIndices = new DrawingMachine().draw(decisionClassObjectIndices, newDecisionClassSize - fullDecisionClassRounds * decisionClassSize);
				
				//gather all decision class object indices fullDecisionClassRounds times and add drawn objects
				int[] newDecisionClassObjectIndices = new  int[fullDecisionClassRounds * decisionClassSize + drawnDecisionClassObjectIndices.length]; //indices can repeat!
				for (int i = 0; i < fullDecisionClassRounds; i++) {
					System.arraycopy(decisionClassObjectIndices, 0, newDecisionClassObjectIndices, 0 + i * decisionClassObjectIndices.length, decisionClassObjectIndices.length);
				}
				System.arraycopy(drawnDecisionClassObjectIndices, 0, newDecisionClassObjectIndices, fullDecisionClassRounds * decisionClassSize, drawnDecisionClassObjectIndices.length);
				
				perClassSelectedObjectIndicesList.add(newDecisionClassObjectIndices); //remember resulting indices of objects from current class
				newTotalNumberOfObjects += newDecisionClassObjectIndices.length;
			}
			//***** UNDERSAMPLING *****
			else if (decisionClassSize > newDecisionClassSize) { //do under-sampling of current decision class
				int[] drawnDecisionClassObjectIndices = new DrawingMachine().draw(decisionClassObjectIndices, newDecisionClassSize);
				perClassSelectedObjectIndicesList.add(drawnDecisionClassObjectIndices);
				newTotalNumberOfObjects += drawnDecisionClassObjectIndices.length;
			//***** COPYING *****
			} else {
				perClassSelectedObjectIndicesList.add(decisionClassObjectIndices); //select all objects from current class (as it already has target size)
				newTotalNumberOfObjects += decisionClassObjectIndices.length;
			}
		} //for
		
		int[] selectedObjectIndices = new int[newTotalNumberOfObjects];
		int lastDestIndex = 0;
		for (int[] perClassSelectedObjectIndices : perClassSelectedObjectIndicesList) {
			System.arraycopy(perClassSelectedObjectIndices, 0, selectedObjectIndices, lastDestIndex, perClassSelectedObjectIndices.length);
			lastDestIndex += perClassSelectedObjectIndices.length;
		}
		
		Arrays.sort(selectedObjectIndices);
		
		return new Data(informationTableWithDistributions.select(selectedObjectIndices, true), data.getName()+"_balanced");
	}
	
	@Override
	public String toString() {
		return this.getClass().getSimpleName()+"("+balancingStrategy.toString()+")";
	}

}

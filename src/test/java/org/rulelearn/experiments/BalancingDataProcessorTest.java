package org.rulelearn.experiments;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.rulelearn.data.Decision;
import org.rulelearn.data.DecisionDistribution;
import org.rulelearn.data.InformationTableWithDecisionDistributions;
import org.rulelearn.data.SimpleDecision;
import org.rulelearn.experiments.BalancingDataProcessor.BalancingStrategy;

class BalancingDataProcessorTest {
	
	@Test
	void testProcess01() { //2 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDERSAMPLING, 7L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(7);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(10);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision1Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		System.out.println();
		
		assertEquals(selectedObjectIndices.length, 3 + 3);
	}
	
	@Test
	void testProcess02() { //2 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.OVERSAMPLING, 3L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(7);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(10);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision1Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 7 + 7);
		System.out.println();
	}
	
	@Test
	void testProcess03() { //2 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDER_AND_OVERSAMPLING, 5L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(7);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(10);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision1Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 5 + 5);
		System.out.println();
	}
	
	@Test
	void testProcess04() { //3 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDERSAMPLING, 17L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision2Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		decisions.add(decision2Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(5);
		Mockito.when(decisionDistributionMock.getCount(decision2Mock)).thenReturn(8);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(16);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(10)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(11)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(12)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(13)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(14)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(15)).thenReturn(decision2Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 3 + 3 + 3);
		System.out.println();
	}
	
	@Test
	void testProcess05() { //3 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.OVERSAMPLING, 19L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision2Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		decisions.add(decision2Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(5);
		Mockito.when(decisionDistributionMock.getCount(decision2Mock)).thenReturn(8);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(16);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(10)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(11)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(12)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(13)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(14)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(15)).thenReturn(decision2Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 8 + 8 + 8);
		System.out.println();
	}
	
	@Test
	void testProcess06() { //3 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDER_AND_OVERSAMPLING, 19L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision2Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		decisions.add(decision2Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(3);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(5);
		Mockito.when(decisionDistributionMock.getCount(decision2Mock)).thenReturn(8);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(16);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(10)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(11)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(12)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(13)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(14)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(15)).thenReturn(decision2Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 5 + 5 + 5);
		System.out.println();
	}
	
	@Test
	void testProcess07() { //3 classes
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDER_AND_OVERSAMPLING, 23L);
		
		DecisionDistribution decisionDistributionMock = Mockito.mock(DecisionDistribution.class);
		InformationTableWithDecisionDistributions informationTableWithDecisionDistributionsMock = Mockito.mock(InformationTableWithDecisionDistributions.class);
		Data dataMock = Mockito.mock(Data.class);
		SimpleDecision decision0Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision1Mock = Mockito.mock(SimpleDecision.class);
		SimpleDecision decision2Mock = Mockito.mock(SimpleDecision.class);
		Set<Decision> decisions = new HashSet<Decision>();
		decisions.add(decision0Mock);
		decisions.add(decision1Mock);
		decisions.add(decision2Mock);
		
		Mockito.when(decisionDistributionMock.getDecisions()).thenReturn(decisions);
		Mockito.when(decisionDistributionMock.getCount(decision0Mock)).thenReturn(2);
		Mockito.when(decisionDistributionMock.getCount(decision1Mock)).thenReturn(4);
		Mockito.when(decisionDistributionMock.getCount(decision2Mock)).thenReturn(5);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.getNumberOfObjects()).thenReturn(11);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecisionDistribution()).thenReturn(decisionDistributionMock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(0)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(1)).thenReturn(decision0Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(2)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(3)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(4)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(5)).thenReturn(decision1Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(6)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(7)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(8)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(9)).thenReturn(decision2Mock);
		Mockito.when(informationTableWithDecisionDistributionsMock.getDecision(10)).thenReturn(decision2Mock);
		
		Mockito.when(informationTableWithDecisionDistributionsMock.select(Mockito.any(), Mockito.anyBoolean())).thenReturn(null);
		
		Mockito.when(dataMock.getInformationTable()).thenReturn(informationTableWithDecisionDistributionsMock);
		
		balancingDataProcessor.process(dataMock);

		ArgumentCaptor<int[]> selectedObjectIndicesCaptor = ArgumentCaptor.forClass(int[].class);
		Mockito.verify(informationTableWithDecisionDistributionsMock).select(selectedObjectIndicesCaptor.capture(), Mockito.anyBoolean());
		
		int[] selectedObjectIndices = selectedObjectIndicesCaptor.getValue();
		
		for (int index : selectedObjectIndices) {
			System.out.print(index + " ");
		}
		
		assertEquals(selectedObjectIndices.length, 3 + 3 + 3);
		System.out.println();
	}
	
	@Test
	void testToString01() {
		BalancingDataProcessor balancingDataProcessor = new BalancingDataProcessor(BalancingStrategy.UNDER_AND_OVERSAMPLING, 23L);
		System.out.println(balancingDataProcessor.toString());
	}

}

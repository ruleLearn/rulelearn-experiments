/**
 * 
 */
package org.rulelearn.experiments;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.rulelearn.experiments.BalancingDataProcessor.BalancingStrategy;

/**
 * @author Marcin
 *
 */
class BalancingDataProcessorProviderTest {

	/**
	 * Test method for {@link org.rulelearn.experiments.BalancingDataProcessorProvider#convert(java.lang.String)}.
	 */
	@Test
	void testConvert() {
		BalancingDataProcessorProvider balancingDataProcessorProvider = new BalancingDataProcessorProvider(BalancingStrategy.UNDERSAMPLING, 0L);
		System.out.println(String.format("%02X", balancingDataProcessorProvider.convert("abcdef")));
		assertEquals(balancingDataProcessorProvider.convert("abcdef"), Long.parseUnsignedLong("E80B5017098950FC", 16));
	}

}

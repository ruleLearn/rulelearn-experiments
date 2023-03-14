package org.rulelearn.experiments;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Random;

import org.rulelearn.experiments.BalancingDataProcessor.BalancingStrategy;

/**
 * Provides a fresh instance of {@link BalancingDataProcessor}.
 *
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class BalancingDataProcessorProvider implements DataProcessorProvider {
	
	BalancingStrategy balancingStrategy;
	long basicSeed; //can be modified when providing a balancing data processor (depending on parameters)
	
	public BalancingDataProcessorProvider(BalancingStrategy balancingStrategy, long basicSeed) {
		this.balancingStrategy = balancingStrategy;
		this.basicSeed = basicSeed;
	}

	@Override
	public BalancingDataProcessor provide() {
		return new BalancingDataProcessor(balancingStrategy, basicSeed);
	}
	
	@Override
	public String toString() {
		return BalancingDataProcessor.serialize(balancingStrategy);
	}

	@Override
	public DataProcessor provide(String dataGroupName) {
		long seed = basicSeed + convert(dataGroupName) * 15_485_863; //accepts overflow
		seed = (new Random(seed)).nextLong();
		return new BalancingDataProcessor(balancingStrategy, seed);
	}

	@Override
	public DataProcessor provide(String dataGroupName, long crossValidationSelector, int foldSelector) {
		long seed = basicSeed + convert(dataGroupName) * 15_485_863 + crossValidationSelector * 472_882_027^2 + foldSelector * 982_451_653^3; //accepts overflow
		seed = (new Random(seed)).nextLong();
		return new BalancingDataProcessor(balancingStrategy, seed);
	}
	
	/**
	 * Convert gives text to long by calculating an MD5 shortcut (16 bytes, 32 hex digits), and then taking the first 8 bytes (16 hex digits)
	 * and transforming such string to unsigned long. It is not guaranteed that for two different strings the results will also be different (but it is very unlikely).
	 * 
	 * @param text converted text
	 * @return long integer corresponding to given text
	 */
	long convert(String text) {
		MessageDigest md = null;
		try {
			md = MessageDigest.getInstance("MD5");
		} catch (NoSuchAlgorithmException e) {
			//do nothing - this should not happen
		}
		
		md.update(text.getBytes(StandardCharsets.UTF_8));
		byte[] hashBytes = md.digest(); //32 hex digits = 16 bytes
		
		StringBuilder reducedHashBuilder = new StringBuilder(8);
		
		for (int i = 0; i < 8; i++) {
			reducedHashBuilder.append(String.format("%02X", hashBytes[i]));
		}
		
		return Long.parseUnsignedLong(reducedHashBuilder.toString(), 16);
	}

}

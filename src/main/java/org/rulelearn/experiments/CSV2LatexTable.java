/**
 * 
 */
package org.rulelearn.experiments;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class CSV2LatexTable {
	
	private static class ReplacementSpecification {
		String toBeReplaced;
		String replacement;
		
		public ReplacementSpecification(String toBeReplaced, String replacement) {
			this.toBeReplaced = toBeReplaced;
			this.replacement = replacement;
		}
	}
	
	public static void main(String[] args) throws IOException { //pass e.g. "temp/csv.txt" "temp/tabular.txt"
		if (args.length != 2) {
			System.out.println("Synopsis: CSV2LatexTable csvInputFilePath latexTableOutputFilePath");
			return;
		}
		String separator1 = ";"; //TODO: adjust!
		String separator2 = "\t"; //alternative separator
		
		ReplacementSpecification[] csvLineReplacementSpecification = new ReplacementSpecification[] { //specify pairs of what to replace with what (in given order!)
				//new ReplacementSpecification(",", "."),
				new ReplacementSpecification("% missing", "PERCENTmv"),
				new ReplacementSpecification("%", ""),
				new ReplacementSpecification("PERCENTmv", "\\textbf{\\%mv}"),
				
				new ReplacementSpecification("bank-churn-10000-v8-0.05-mv1.5", "\\textbf{5}"),
				new ReplacementSpecification("bank-churn-10000-v8-0.10-mv1.5", "\\textbf{10}"),
				new ReplacementSpecification("bank-churn-10000-v8-0.15-mv1.5", "\\textbf{15}"),
				new ReplacementSpecification("bank-churn-10000-v8-0.20-mv1.5", "\\textbf{20}"),
				new ReplacementSpecification("bank-churn-10000-v8-0.25-mv1.5", "\\textbf{25}"),
				new ReplacementSpecification("bank-churn-10000-v8", "\\textbf{0}"),
				
				new ReplacementSpecification("VCDomLEM+mode-mv2", "$\\epsilon$-$\\mathbb{D}^{mv}_{2}$"),
				new ReplacementSpecification("mode-mv2", "$\\epsilon$-$\\mathbb{D}^{mv}_{2}$"),
				
				new ReplacementSpecification("VCDomLEM+mode-mv1.5", "$\\epsilon$-$\\mathbb{D}^{mv}_{1.5}$"),
				new ReplacementSpecification("mode-mv1.5", "$\\epsilon$-$\\mathbb{D}^{mv}_{1.5}$"),
				new ReplacementSpecification("VCDomLEMModeRuleClassifierLearner", "$\\epsilon$-$\\mathbb{D}^{mv}_{1.5}$"),
				
				new ReplacementSpecification("WEKAClassifierLearner(J48)", "C4.5"),
				new ReplacementSpecification("J48", "C4.5"),
				new ReplacementSpecification("WEKAClassifierLearner(NaiveBayes)", "NB"),
				new ReplacementSpecification("NaiveBayes -D", "NB"),
				new ReplacementSpecification("WEKAClassifierLearner(SMO)", "SVM"),
				new ReplacementSpecification("SMO", "SVM"),
				new ReplacementSpecification("WEKAClassifierLearner(RandomForest)", "RF"),
				new ReplacementSpecification("RandomForest", "RF"),
				new ReplacementSpecification("WEKAClassifierLearner(MultilayerPerceptron)", "MP"),
				new ReplacementSpecification("MultilayerPerceptron", "MP"),
				new ReplacementSpecification("WEKAClassifierLearner(JRip)", "RIPP"),
				new ReplacementSpecification("jRip", "RIPP"),
				//--
				new ReplacementSpecification("WEKAClassifierLearner(OSDL)", "OSDL"),
				new ReplacementSpecification("WEKAClassifierLearner(OLM)", "OLM"),
				new ReplacementSpecification("WEKAClassifierLearner(MoNGEL)", "MoNGEL"),
				
				new ReplacementSpecification("e<=", "$\\theta_X =$"),
				new ReplacementSpecification("cF>=", "$\\lfloor{cov}\\rfloor =$"),
				new ReplacementSpecification("conf>0.6666", "$\\lfloor{conf}\\rfloor = 2/3$")
		};
		
		String csvFilePath = args[0];
		String latexTabularFilePath = args[1];
		
		List<String> csvLines = new ArrayList<>(16);
		
        try (FileReader fileReader = new FileReader(csvFilePath);
        	BufferedReader reader = new BufferedReader(fileReader)) {
            String inputLine;
            while ((inputLine = reader.readLine()) != null) {
                csvLines.add(inputLine);
            }
        } catch (IOException exception) {
            System.out.println("Cannot read CSV file: "+exception.getMessage());
            return;
        }
        
        String[] tokens;
        StringBuilder tabularBuilder = new StringBuilder();
        
        tabularBuilder.append("\\begin{table}[h!]").append(System.lineSeparator());
        tabularBuilder.append("  \\caption{TODO}\\label{tab:TODO}").append(System.lineSeparator());
        tabularBuilder.append("  \\centering").append(System.lineSeparator());
        tabularBuilder.append("  \\scalebox{0.76}{").append(System.lineSeparator());
        
        int lineIndex = 0;
        for (String csvLine : csvLines) {
        	for (int i = 0; i < csvLineReplacementSpecification.length; i++) {
        		csvLine = csvLine.replace(csvLineReplacementSpecification[i].toBeReplaced, csvLineReplacementSpecification[i].replacement);
        	}
        	
        	if (csvLine.contains(separator1)) {
        		tokens = csvLine.split(separator1);
        	} else if (csvLine.contains(separator2)) {
        		tokens = csvLine.split(separator2);
        	} else {
        		throw new IOException("Cannot tokenize input line.");
        	}
        	
        	if (lineIndex == 0) {
        		//print tabular definition
        		tabularBuilder.append("  \\begin{tabular}{|");
        		for (int tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
        			tabularBuilder.append("c|");
        		}
        		tabularBuilder.append("}");
        		
        		tabularBuilder.append(System.lineSeparator());
            	tabularBuilder.append("    \\hline");
            	tabularBuilder.append(System.lineSeparator());
        	}
        	
        	int tokenIndex = 0;
        	tabularBuilder.append("    ");
        	for (String token : tokens) {
        		tabularBuilder.append(token);
        		if (tokenIndex < tokens.length - 1) {
        			tabularBuilder.append(" & ");
        		} else {
        			tabularBuilder.append("\\\\");
        		}
        		
        		tokenIndex++;
        	}
        	tabularBuilder.append(System.lineSeparator());
        	tabularBuilder.append("    \\hline");
        	tabularBuilder.append(System.lineSeparator());
        	
        	lineIndex++;
        }
        
        tabularBuilder.append("  \\end{tabular}").append(System.lineSeparator());
        tabularBuilder.append("  }").append(System.lineSeparator());
        tabularBuilder.append("\\end{table}");
        
        //print to console
        System.out.println(tabularBuilder.toString());
        
        //write to file
        try (FileOutputStream fileStream = new FileOutputStream(latexTabularFilePath);
        		OutputStreamWriter writer = new OutputStreamWriter(fileStream)) {
        	writer.write(tabularBuilder.toString());
        } catch (IOException exception) {
        	System.out.println("Could not write to file "+latexTabularFilePath);
			return;
        }
	}
}

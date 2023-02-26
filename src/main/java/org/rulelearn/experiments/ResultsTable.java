/**
 * 
 */
package org.rulelearn.experiments;

import java.util.ArrayList;
import java.util.List;

/**
 * Class representing table that can be pasted to a spreadsheet.
 * 
 * @author Marcin Szeląg (<a href="mailto:marcin.szelag@cs.put.poznan.pl">marcin.szelag@cs.put.poznan.pl</a>)
 */
public class ResultsTable<ContentType> {
	
	List<String> columnHeaders;
	List<String> rowHeaders;
	List<List<ContentType>> cells;
	int rowIndex = -1;
	int colIndex = -1;
	int expectedColsCount;
	String topLeftCell;
	
	public ResultsTable(int expectedRowsCount, int expectedColsCount) {
		cells = new ArrayList<List<ContentType>>(expectedRowsCount);
		this.expectedColsCount = expectedColsCount;
		rowHeaders = new ArrayList<String>(expectedRowsCount);
	}
	
	public void setTopLeftCell(String value) {
		topLeftCell = value;
	}
	
	public void setColumnHeaders(List<String> columnHeaders) {
		this.columnHeaders = columnHeaders;
	}
	
	public void newRow(String rowHeader) {
		cells.add(new ArrayList<ContentType>(expectedColsCount));
		rowHeaders.add(rowHeader);
		rowIndex++;
		colIndex = 0;
	}
	
	public void addRowValue(ContentType value) {
		cells.get(rowIndex).add(value);
		colIndex++;
	}
	
	public String toString(String separator) {
		StringBuilder sB = new StringBuilder();
		sB.append(topLeftCell);
		
		if (columnHeaders.size() > 0) {
			sB.append(separator);
		}
		
		int j = 0; //column index
		for (String columnHeader : columnHeaders) {
			sB.append(columnHeader);
			if (j < columnHeaders.size() - 1) {
				sB.append(separator);
			}
			j++;
		}
		sB.append(System.lineSeparator());
		
		int i = 0; //row index
		for (List<ContentType> row : cells) {
			sB.append(rowHeaders.get(i));
			if (row.size() > 0) {
				sB.append(separator);
			}
			
			j = 0; //column index
			for (ContentType cell : row) {
				sB.append(cell.toString());
				if (j < row.size() - 1) {
					sB.append(separator);
				}
				j++;
			}
			i++;
			sB.append(System.lineSeparator());
		}
		
		return sB.toString();
	}
	
}

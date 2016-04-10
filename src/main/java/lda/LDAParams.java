package lda;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public class LDAParams {
	private String[] vocabulary;
	private JavaRDD<Vector> countVectors;
	private long totalTokens;
	
	public long getTotalTokens() {
		return totalTokens;
	}

	public void setTotalTokens(long totalTokens) {
		this.totalTokens = totalTokens;
	}

	public LDAParams(String[] vocabulary, JavaRDD<Vector> countVectors, long totalTokens) {
		super();
		this.vocabulary = vocabulary;
		this.countVectors = countVectors;
		this.totalTokens = totalTokens;
	}
	
	public String[] getVocabulary() {
		return vocabulary;
	}

	public void setVocabulary(String[] vocabulary) {
		this.vocabulary = vocabulary;
	}
	public JavaRDD<Vector> getCountVectors() {
		return countVectors;
	}
	public void setCountVectors(JavaRDD<Vector> countVectors) {
		this.countVectors = countVectors;
	}
	
	
}

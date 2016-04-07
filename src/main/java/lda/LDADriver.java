package lda;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.EMLDAOptimizer;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.SQLContext;

import preprocessing.Preprocessor;
import scala.Tuple2;

public class LDADriver {

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("ParseXML").setMaster("local[2]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		int numTopics = LDAConstants.NUM_TOPICS;
		
		String inputLocation = args[0];

		//Reducing logging for better debugging
		Logger.getRootLogger().setLevel(Level.ERROR);
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);
		
		LDAParams params = Preprocessor.preprocess(inputLocation, sqlContext);
		
	    JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(params.getCountVectors().zipWithIndex().map(
		        new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
		          public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
		            return doc_id.swap();
		          }
		        }
		    ));
			
		    corpus.cache();
		    
		    long actualCorpusSize = corpus.count();
		    String[] vocabulary = params.getVocabulary();
		    int actualVocabSize = params.getVocabulary().length;
		    
		   /* OnlineLDAOptimizer optimizer = new OnlineLDAOptimizer()//
                    .setMiniBatchFraction( 0.05 + 1.0 / actualCorpusSize);	*/
		    
		    EMLDAOptimizer optimizer = new EMLDAOptimizer();
		    
		    // Cluster the documents into three topics using LDA
		    LDA lda = new LDA()
					.setK(numTopics)
					.setOptimizer(optimizer)
					.setMaxIterations(LDAConstants.MAX_ITERATIONS)
					.setDocConcentration(LDAConstants.docConcentration)
				    .setTopicConcentration(LDAConstants.topicConcentration);
		    
		    LDAModel ldaModel = lda.run(corpus);
		    
		   Tuple2<int[], double[]>[] topics = ((DistributedLDAModel) ldaModel).describeTopics(10);

		    int topicCount = topics.length; 
		    for(int t=0; t<topicCount; t++){
		    	Tuple2<int[], double[]> topic = topics[t];
		    	System.out.println("Topic " + t + ":");
		    	int[] indices = topic._1();
		    	double[] values = topic._2();
		    	
		    	for(int i=0; i<indices.length; i++){
		    		System.out.print(vocabulary[indices[i]]+ " ");
		    		System.out.println(values[i]);
		    	}
		    	System.out.println("");
		    }

		 //Remove duplicates from words vector so that the indices are in line with count vector
/*		 UDF1 stringSet = new UDF1<Seq<String>, Seq<String>>() {
		      @Override
		      public Seq<String> call(Seq<String> str) {
		    	 return str.distinct();
		      }
		    };
		 
		 sqlContext.udf().register("StringSet", stringSet , DataTypes.createArrayType(DataTypes.StringType));
		 
		 
		 
		 DataFrame modiDF = fileDF.withColumn
		 					("filteredUDF",fileDF.selectExpr("StringSet(filtered)").apply("filtered"));
		 
		 modiDF.show();*/
	}
}

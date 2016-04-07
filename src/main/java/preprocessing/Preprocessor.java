package preprocessing;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.sweble.wikitext.engine.EngineException;
import org.sweble.wikitext.engine.PageId;
import org.sweble.wikitext.engine.PageTitle;
import org.sweble.wikitext.engine.WtEngineImpl;
import org.sweble.wikitext.engine.config.WikiConfig;
import org.sweble.wikitext.engine.nodes.EngProcessedPage;
import org.sweble.wikitext.engine.utils.DefaultConfigEnWp;
import org.sweble.wikitext.parser.parser.LinkTargetException;

import lda.LDAConstants;
import lda.LDAParams;

public class Preprocessor {
	public static LDAParams preprocess(String inputXMLFile, SQLContext sqlContext){
		 DataFrame df = sqlContext.read()
		         .format("com.databricks.spark.xml")
		         .option("rowTag", "page")
		       	 .load(inputXMLFile);
		 
		 df.registerTempTable("pages");
		 DataFrame newDF = sqlContext.sql("select title,revision.text from pages where revision.text is not null");
		 
		 //newDF.show();
		 JavaRDD<Row> res = newDF.javaRDD();
		 
		 JavaRDD<Document> result = res.map(new Function <Row, Document>(){
			 public Document call(Row entry) throws LinkTargetException, EngineException {
				 	String title = entry.getString(0);
					String wikiText = entry.getString(1);
					
				    // Set-up a simple wiki configuration
				    WikiConfig config = DefaultConfigEnWp.generate();
				    // Instantiate a compiler for wiki pages
				    WtEngineImpl engine = new WtEngineImpl(config);
				    // Retrieve a page
				    PageTitle pageTitle = PageTitle.make(config, title);
				    PageId pageId = new PageId(pageTitle, -1);
				    // Compile the retrieved page
				    EngProcessedPage cp = engine.postprocess(pageId, wikiText, null);
				    TextConverter p = new TextConverter(config, Integer.MAX_VALUE);
				    
				    String processedText = (String)p.go(cp.getPage());
				    
				    return new Document(title, preprocess(processedText));
			 }	 
		 });
		 
		 DataFrame fileDF = sqlContext.createDataFrame(result, Document.class);
		 
		// Configure an ML pipeline
		 RegexTokenizer tokenizer = new RegexTokenizer()
		   .setInputCol("text")
		   .setOutputCol("words");
		 
		 StopWordsRemover remover = new StopWordsRemover()
				  .setInputCol("words")
				  .setOutputCol("filtered");
		 
		/* CountVectorizer cv = new CountVectorizer()
				  .setVocabSize(vocabSize)
				  .setInputCol("filtered")
				  .setOutputCol("features");*/
		 
		 Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {tokenizer, remover});
		 	 
		// Fit the pipeline to train documents.
		 PipelineModel model = pipeline.fit(fileDF);
		 
		 DataFrame tempDF = model.transform(fileDF);
		 
		 //Separate CountVectorizerModel to get the vocabulary
		 CountVectorizerModel cvModel = new CountVectorizer()
				  .setVocabSize(LDAConstants.VOCAB_SIZE)
				  .setInputCol("filtered")
				  .setOutputCol("features").fit(tempDF);
		 
		 JavaRDD<Vector> countVectors = cvModel.transform(tempDF)
		          .select("features").toJavaRDD()
		          .map(new Function<Row, Vector>() {
		            public Vector call(Row row) throws Exception {
		            	return (Vector)row.get(0);
		            }
		          });
		 
		 //Calculate actual num tokens   
		
		return new LDAParams(cvModel.vocabulary(), countVectors);
	}
	
	public static String preprocess(String text){
		//TODO
	/*	1. LowerCase 
	 * 	2. Stemming
	 * http://stackoverflow.com/questions/5391840/stemming-english-words-with-lucene
	 * https://github.com/master/spark-stemming
	 * 
	 * 
	 */	
		text = text.toLowerCase();		
		
		return text;
	}
}
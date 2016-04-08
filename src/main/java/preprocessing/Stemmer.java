package preprocessing;

import java.io.IOException;
import java.io.StringReader;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

public class Stemmer {
	public static String stem(String string) throws IOException {
	    TokenStream tokenizer = new StandardTokenizer(Version.LUCENE_47, new StringReader(string));
	    tokenizer = new StandardFilter(Version.LUCENE_47, tokenizer);
	    tokenizer = new LowerCaseFilter(Version.LUCENE_47, tokenizer);
	    tokenizer = new PorterStemFilter(tokenizer);

	    CharTermAttribute token = tokenizer.getAttribute(CharTermAttribute.class);

	    tokenizer.reset();

	    StringBuilder stringBuilder = new StringBuilder();

	    while(tokenizer.incrementToken()) {
	        if(stringBuilder.length() > 0 ) {
	            stringBuilder.append(" ");
	        }

	        stringBuilder.append(token.toString());
	    }

	    tokenizer.end();
	    tokenizer.close();

	    return stringBuilder.toString();
	}
}

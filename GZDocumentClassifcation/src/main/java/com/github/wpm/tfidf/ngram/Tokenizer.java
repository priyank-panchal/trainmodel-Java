package com.github.wpm.tfidf.ngram;

import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String text);
}

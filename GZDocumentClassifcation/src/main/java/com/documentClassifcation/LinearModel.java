package com.documentClassifcation;

import com.github.wpm.tfidf.ngram.NgramTfIdf;
import com.github.wpm.tfidf.TfIdf;
import com.github.wpm.tfidf.ngram.RegularExpressionTokenizer;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.util.*;


public class LinearModel {
    private Map<String, Integer> vocabulary;
    private List<Integer> ngrams;
    private List<Double> idf;
    private List<List<Double>> weights;
    private List<Double> biases;
    private Map<Integer, Double> idfIndex;
    private static final Logger logger = LogManager.getLogger(LinearModel.class);

    class VocabularyTokenizer extends RegularExpressionTokenizer {
        @Override
        public List<String> tokenize(String text) {
            List<String> tokens = new ArrayList<>();
            for (String token : super.tokenize(text)) {
                if (vocabulary.containsKey(token)) {
                    tokens.add(token);
                }
            }
            return tokens;
        }
    }

    @Override
    public String toString() {
        return String.format("<%d features, %d class, ngrams %s>", features(), classes(), ngrams);
    }

    public int features() {
        return weights.get(0).size();
    }

    private Collection<String> documentTerms(String document) {

        Collection<String> terms = NgramTfIdf.ngramDocumentTerms(
                new VocabularyTokenizer(), ngrams, Arrays.asList(document)).iterator().next();
        //it will part based on Ngram
        Collection<String> recognizedTerms = new ArrayList<>();
        for (String term : terms) {
            if (vocabulary.containsKey(term)) {
                //if check words are occurred into the vocabulary file
                recognizedTerms.add(term);
            }
        }
        return recognizedTerms;
    }

    public int classes() {
        return biases.size();
    }

    public double[] calculation(String document) {
        Collection<String> terms = documentTerms(document); // return the value and tokenize
        Map<String, Double> tf = TfIdf.tf(terms); // find the tf terms frequency
        Map<Integer, Double> occurred = new HashMap<>(); //how many times occurred into the all documents
        for (Map.Entry<String, Double> e : tf.entrySet()) {
            occurred.put(vocabulary.get(e.getKey()), e.getValue());
        }
        Map<Integer, Double> tfIdf = TfIdf.tfIdf(occurred, idfWithIndex(), TfIdf.Normalization.COSINE);//calculate tfidf
        int classSize = classes();
        double[] classScores = new double[classSize];//2 classes 0 Agreements and 1 is other documents
        for (int i = 0; i < classSize; ++i) {
            classScores[i] = biases.get(i);   //class_log_prior_ - It represents log probability of each class.
            for (Map.Entry<Integer, Double> e : tfIdf.entrySet()) {
                int index = e.getKey();
                double tfidf = e.getValue();
                classScores[i] += weights.get(i).get(index) * tfidf;   //Empirical log probability of features given a class, P(x_i|y).
            }
        }
        return classScores;
    }

    public static double logSumOfExponentials(double[] exponentialTerms) {
        if (exponentialTerms.length == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        double maxTerm = Double.NEGATIVE_INFINITY;
        for (double d : exponentialTerms) {
            if (d > maxTerm) {
                maxTerm = d;
            }
        }
        if (maxTerm == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        double sum = 0.;
        for (double d : exponentialTerms) {
            sum += Math.exp(d - maxTerm);
        }
        return maxTerm + Math.log(sum);
    }

    private Map<Integer, Double> idfWithIndex() {
        if (idfIndex == null) {
            idfIndex = new HashMap<>();
            for (int i = 0; i < idf.size(); i++) {
                idfIndex.put(i, idf.get(i));
            }
        }
        return idfIndex;
    }

    //get vocabulary into the map;
    public Map<String, Integer> getVocabulary() {
        return vocabulary;
    }

    //set vocabulary
    public void setVocabulary(Map<String, Integer> vocabulary) {
        this.vocabulary = vocabulary;
    }

    //get a ngrams numbers
    public List<Integer> getNgrams() {
        return ngrams;
    }

    //set Ngrams numbers
    public void setNgrams(List<Integer> ngrams) {
        if (!ngrams.contains(1)) {
            throw new RuntimeException("The tokenizer requires the vocabulary to contain unigrams.");
        }
        this.ngrams = ngrams;
    }

    //get IDF into the list
    public List<Double> getIdf() {
        return idf;
    }


    //set idf
    public void setIdf(List<Double> idf) {
        idfIndex = null;
        this.idf = idf;
    }

    //get weights
    public List<List<Double>> getWeights() {
        return weights;
    }

    //set Weights
    public void setWeights(List<List<Double>> weights) {
        this.weights = weights;
    }


    //getBiases
    public List<Double> getBiases() {
        return biases;
    }

    public void setBiases(List<Double> biases) {
        this.biases = biases;
    }

}

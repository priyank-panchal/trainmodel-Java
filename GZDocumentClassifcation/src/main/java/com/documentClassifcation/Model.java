package com.documentClassifcation;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class Model {
    private final static File MODEL_FILES = new File("C:\\Users\\admin\\IdeaProjects\\GZDocumentClassifcation\\src\\main\\resources\\output.gz");
    private final static File ABSOLUTE_PATH_FILE = new File("C:\\Users\\admin\\IdeaProjects\\GZDocumentClassifcation\\src\\main\\resources\\predictionFile.txt");
    private final static Path STOP_WORDS_TXT_FILE = Paths.get("C:\\Users\\admin\\IdeaProjects\\GZDocumentClassifcation\\src\\main\\resources\\stop_words.txt");
    private static final Logger logger = LogManager.getLogger(Model.class);
    static private int largest(double[] numbers) {
        int maxIndex = 0;
        Double max = null;
        for (int i = 0; i < numbers.length; ++i) {
            if (max == null || numbers[i] > max) {
                maxIndex = i;
                max = numbers[i];
            }
        }
        return maxIndex;
    }

    static private String removeStopWords(String document) {
        String[] allWords = document.replaceAll("[^a-zA-Z]", " ").toLowerCase().split(" ");
        ArrayList<String> wordWithoutStopWord = new ArrayList<>(Arrays.asList(allWords));
        try (Stream<String> lines = Files.lines(STOP_WORDS_TXT_FILE)) {
            ArrayList<String> stopWords = lines.collect(Collectors.toCollection(ArrayList::new));
            wordWithoutStopWord.removeAll(stopWords);
            wordWithoutStopWord.removeAll(Arrays.asList("", null));
        } catch (Exception e) {
            e.printStackTrace();
        }
        StringBuilder sb = new StringBuilder();
        for (String str : wordWithoutStopWord) {
            sb.append(str);
            sb.append(" ");
        }
        return  sb.toString();
    }

    public static void main(String[] args) {
        BasicConfigurator.configure();
        long startTime = System.currentTimeMillis(); //unzip the file and read
        try (GZIPInputStream gzipInputStream = new GZIPInputStream(new FileInputStream(MODEL_FILES))) {
            ObjectMapper mapper = new ObjectMapper(); //read the value
            LinearModel model = mapper.readValue(gzipInputStream, LinearModel.class); //store model parameters put into the list form
            List<String> lines = FileUtils.readLines(ABSOLUTE_PATH_FILE, "UTF-8"); //read predicated files data
            List<File> filesInFolder = new ArrayList<>();
            for (String fileDetails : lines) {
                String[] split = fileDetails.split(",");
                filesInFolder.add(new File(split[1]));
            }
            for (File documentTest : filesInFolder) {
                String document = FileUtils.readFileToString(documentTest, StandardCharsets.UTF_8);
                String content = removeStopWords(document);  //remove all stop_words and more.
                if(content.length() > 20) {
                    double[] scores = model.calculation(content);//perform some calculation
                    double logSumOfExponentials = model.logSumOfExponentials(scores); //perform find the probability
                    int prediction = largest(scores); // find large element
                    System.out.println(documentTest);
                    System.out.println("contract : " + Math.exp(scores[0] - logSumOfExponentials) + ";" + " Non-Contract : " + Math.exp(scores[1] - logSumOfExponentials));
                }
                else{
                    System.out.println("not enough length for predication");
                }
            }
            System.out.println("Take millis seconds " + (System.currentTimeMillis() - startTime));
            System.out.println("Take Seconds " + TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis() - startTime));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
/*
 def predict_proba(self, X):
        joint_prob = self._joint_log_likelihood(X)
        joint_prob_norm = logsumexp(joint_prob, axis = 1, keepdims = True)
        pred_proba = np.exp(joint_prob - joint_prob_norm)
        return pred_proba

    def _joint_log_likelihood(self, X):
        joint_prob = X * self.feature_log_prob_.T + self.class_log_prior_
        return joint_prob
 */
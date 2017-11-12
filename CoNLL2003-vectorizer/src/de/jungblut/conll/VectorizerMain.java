package de.jungblut.conll;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import info.debatty.java.stringsimilarity.QGram;
import org.apache.commons.cli.*;
import org.yaml.snakeyaml.Yaml;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;

import static com.google.common.base.Preconditions.checkArgument;

public class VectorizerMain {

    private static final Pattern SPLIT_PATTERN = Pattern.compile(" ");
    private static final int SEQUENCE_LEN = 20;
    private static final int CHAR_NGRAM_LEN = 3;
    private static final int MIN_CHAR_NGRAM_OCCURRENCE = 200;

    private static final int EMBEDDING_VECTOR_SIZE = 50;
    private static final int POS_TAG_SIZE = 47;
    private static final int SHAPE_FEATURES_SIZE = 5;

    private static final String DATA_PATH = "data/";
    private static final String GLOVE_FILE_NAME = "glove.6B.50d.txt";
    private static final String NER_TRAIN_FILE_NAME = "eng.train.txt";
    private static final String TRAIN_OUT_FILE_NAME = "vectorized";
    private static final String META_OUT_FILE_NAME = "meta.yaml";
    private static final String OUT_LABEL = "O";

    private LabelManager labelManager;
    private LabelManager posTagManager;
    private String[] qgramDict;
    private int sequenceLength;
    private int embeddingVectorSize;
    private String embeddingPath;
    private String inputFilePath;
    private String outputFolder;
    private boolean binaryOutput;

    private VectorizerMain(int sequenceLength,
                           int embeddingVectorSize,
                           String embeddingPath,
                           String inputFilePath,
                           String outputFolder,
                           boolean binaryOutput,
                           LabelManager labelManager,
                           LabelManager posTagManager,
                           String[] qgramDict) {
        this.sequenceLength = sequenceLength;
        this.embeddingVectorSize = embeddingVectorSize;
        this.embeddingPath = embeddingPath;
        this.inputFilePath = inputFilePath;
        this.outputFolder = outputFolder;
        this.binaryOutput = binaryOutput;
        this.labelManager = labelManager;
        this.posTagManager = posTagManager;
        this.qgramDict = qgramDict;
    }

    private void vectorize() throws IOException {
        System.out.println("Sequence length: " + sequenceLength);
        System.out.println("Embedding vector dimension: " + embeddingVectorSize);
        System.out.println("Embedding path: " + embeddingPath);
        System.out.println("Input path: " + inputFilePath);
        System.out.println("Binary output: " + binaryOutput);
        System.out.println("Output folder: " + outputFolder);

        // read the glove embeddings
        HashMap<String, float[]> embeddingMap = readGloveEmbeddings(embeddingPath, embeddingVectorSize);
        System.out.println("read " + embeddingMap.size()
                + " embedding vectors. Vectorizing...");

        labelManager.getOrCreate(OUT_LABEL);
        final QGram qgram = new QGram(CHAR_NGRAM_LEN);
        final String[] dict = qgramDict != null ? qgramDict : prepareNGramDictionary(qgram);
        System.out.println("qgram dictionary len: " + dict.length);

        final int singleFeatureSize = POS_TAG_SIZE + embeddingVectorSize + SHAPE_FEATURES_SIZE + dict.length;
        final int numFinalFeatures = sequenceLength * singleFeatureSize;

        System.out.println("word feature vector size: " + singleFeatureSize);

        // use an array of zeros as the default feature
        final float[] defaultVector = new float[singleFeatureSize];

        int numTotalTokens = 0;
        int numOutOfVocabTokens = 0;
        HashMultiset<String> labelHistogram = HashMultiset.create();

        try (SequenceFileWriter writer = createWriter(outputFolder, binaryOutput)) {

            Deque<float[]> vectorBuffer = new LinkedList<>();
            Deque<Integer> labelBuffer = new LinkedList<>();

            try (BufferedReader reader = new BufferedReader(new FileReader(
                    inputFilePath))) {

                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty()) {
                        continue;
                    }

                    final float[] featureVector = new float[singleFeatureSize];
                    // format is as follows: "German JJ I-NP I-MISC"
                    String[] split = SPLIT_PATTERN.split(line);
                    String tkn = cleanToken(split[0]);
                    numTotalTokens++;
                    if (embeddingMap.containsKey(tkn)) {
                        float[] embedding = embeddingMap.get(tkn);
                        System.arraycopy(embedding, 0, featureVector, 0, embedding.length);
                    } else {
                        numOutOfVocabTokens++;
                    }

                    // we add one hot encoded pos tags into the feature vector
                    String posTag = split[1].toLowerCase().trim();
                    int posTagIndex = posTagManager.getOrCreate(posTag);
                    featureVector[embeddingVectorSize + posTagIndex] = 1f;

                    // we add shape features from the non-normalized token
                    wordShape(split[0], embeddingVectorSize + POS_TAG_SIZE, featureVector);

                    // we add qgram statistics as a one-hot encoding into the feature vector
                    Map<String, Integer> profile = qgram.getProfile(tkn);
                    for (Map.Entry<String, Integer> entry : profile.entrySet()) {
                        int i = Arrays.binarySearch(dict, entry.getKey());
                        if (i >= 0) {
                            featureVector[embeddingVectorSize + POS_TAG_SIZE + SHAPE_FEATURES_SIZE + i] += entry.getValue();
                        }
                    }

                    String label = split[3].trim();
                    labelHistogram.add(label);
                    int labelIndex = labelManager.getOrCreate(label);

                    vectorBuffer.addLast(featureVector);
                    labelBuffer.addLast(labelIndex);

                    // if we reach the buffer size we can flush the next item in the queue
                    if (vectorBuffer.size() == sequenceLength) {
                        writeAndFillSequenceIfNeeded(defaultVector, writer, labelBuffer, vectorBuffer);
                    }
                }
            }

            while (!labelBuffer.isEmpty()) {
                writeAndFillSequenceIfNeeded(defaultVector, writer, labelBuffer, vectorBuffer);
            }
        }

        System.out.println(labelHistogram);

        System.out.println("oov tokens vs. total number of tokens "
                + numOutOfVocabTokens
                + " / " + numTotalTokens
                + " = " + (numOutOfVocabTokens / (double) numTotalTokens) * 100d + "%");

        // dump the label map with # features as YAML map.
        Map<String, Object> data = new HashMap<>();
        data.put("embedding_dim", embeddingVectorSize);
        data.put("seq_len", sequenceLength);
        data.put("nlabels", labelManager.getLabelMap().size());
        data.put("feature_dim", singleFeatureSize);
        data.put("total_feature_dim", numFinalFeatures);
        // inverse the map so we can do int->string lookups somewhere else
        data.put("labels", labelManager.getLabelMap().inverse());
        data.put("pos_tags", posTagManager.getLabelMap().inverse());
        data.put("ngram_dict", dict);
        Yaml yaml = new Yaml();
        Files.write(Paths.get(outputFolder + META_OUT_FILE_NAME), yaml.dump(data)
                .getBytes());

        System.out.println("Done.");
    }

    private String[] prepareNGramDictionary(QGram qgram) throws IOException {
        final HashMultiset<String> set = HashMultiset.create();
        try (BufferedReader reader = new BufferedReader(new FileReader(
                inputFilePath))) {

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }

                String[] split = SPLIT_PATTERN.split(line);
                String tkn = cleanToken(split[0]);
                Map<String, Integer> profile = qgram.getProfile(tkn);
                for (Map.Entry<String, Integer> entry : profile.entrySet()) {
                    //noinspection ResultOfMethodCallIgnored
                    set.add(entry.getKey(), entry.getValue());
                }
            }
        }

        // do some naive word statistics cut-off
        return set.entrySet()
                .stream()
                .filter(e -> e.getCount() > MIN_CHAR_NGRAM_OCCURRENCE)
                .map(Multiset.Entry::getElement)
                .sorted()
                .toArray(String[]::new);
    }

    private String cleanToken(String s) {
        return s.toLowerCase().trim();
    }

    public static void main(String[] args) throws IOException, ParseException {

        Options options = new Options();
        options
                .addOption(
                        "s",
                        "sequenceLength",
                        true,
                        "how long the sequence should be chunked onto");
        options.addOption("d", "embvecdim", true,
                "the dimensionality of the embedding vectors");
        options.addOption("b", "binary", false,
                "if supplied, outputs in binary instead of text format");
        options.addOption("i", "input", true, "the path of the dataset");
        options.addOption("o", "output", true, "the folder for the output");
        options.addOption("e", "embeddings", true, "the path of the embeddings");
        options.addOption("l", "meta", true,
                "the path of the train meta yaml to get the labels");

        if (args.length > 0 && args[0].equals("-h")) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("vectorizer", options);
            System.exit(0);
        }

        System.out.println("add -h for more options!");
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        int seqLen = Integer.parseInt(cmd.getOptionValue('s',
                SEQUENCE_LEN + ""));
        int embeddingVectorSize = Integer.parseInt(cmd.getOptionValue('d',
                EMBEDDING_VECTOR_SIZE + ""));

        String embeddingPath = cmd.getOptionValue('e', DATA_PATH + GLOVE_FILE_NAME);
        String inputFilePath = cmd.getOptionValue('i', DATA_PATH
                + NER_TRAIN_FILE_NAME);
        String outputFolderPath = cmd.getOptionValue('o', DATA_PATH);
        boolean binaryOutput = cmd.hasOption('b');

        LabelManager labelManager = new LabelManager();
        LabelManager posTagManager = new LabelManager();
        String[] qgramDict = null;
        if (cmd.hasOption('l')) {
            Yaml yaml = new Yaml();
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) yaml.load(new String(
                    Files.readAllBytes(Paths.get(cmd.getOptionValue('l')))));
            @SuppressWarnings("unchecked")
            Map<Integer, String> labels = (Map<Integer, String>) map.get("labels");
            labelManager = new LabelManager(labels);
            @SuppressWarnings("unchecked")
            Map<Integer, String> posLabels = (Map<Integer, String>) map.get("pos_tags");
            posTagManager = new LabelManager(posLabels);
            @SuppressWarnings("unchecked")
            List<String> dictList = (List<String>) map.get("ngram_dict");
            qgramDict = dictList.toArray(new String[dictList.size()]);
        }

        VectorizerMain m = new VectorizerMain(seqLen,
                embeddingVectorSize, embeddingPath, inputFilePath, outputFolderPath,
                binaryOutput, labelManager, posTagManager, qgramDict);
        m.vectorize();
    }

    private void wordShape(String s, int offset, float[] featureVector) {
        boolean digit = true;
        boolean upper = true;
        boolean lower = true;
        boolean mixed = true;
        boolean firstUpper = Character.isUpperCase(s.charAt(0));
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!Character.isDigit(c)) {
                digit = false;
            }
            if (!Character.isLowerCase(c)) {
                lower = false;
            }
            if (!Character.isUpperCase(c)) {
                upper = false;
            }
            if ((i == 0 && !Character.isUpperCase(c)) || (i >= 1 && !Character.isLowerCase(c))) {
                mixed = false;
            }
        }

        if (digit) {
            featureVector[offset] = 1f;
        }
        if (upper) {
            featureVector[offset + 1] = 1f;
        }
        if (lower) {
            featureVector[offset + 2] = 1f;
        }
        if (mixed) {
            featureVector[offset + 3] = 1f;
        }
        if (firstUpper) {
            featureVector[offset + 4] = 1f;
        }
    }

    private void writeAndFillSequenceIfNeeded(float[] defaultVector, SequenceFileWriter writer,
                                              Deque<Integer> labelBuffer, Deque<float[]> vectorBuffer) throws IOException {
        checkArgument(labelBuffer.size() == vectorBuffer.size(), "seq and feature size don't match");
        int[] labels = new int[Math.max(labelBuffer.size(), sequenceLength)];
        List<float[]> featList = new ArrayList<>();

        for (int i = 0; i < labels.length; i++) {
            if (labelBuffer.isEmpty()) {
                labels[i] = labelManager.getOrCreate("O");
                featList.add(defaultVector);
            } else {
                labels[i] = labelBuffer.pop();
                featList.add(vectorBuffer.pop());
            }
        }

        writer.write(labels, featList);
    }

    private static HashMap<String, float[]> readGloveEmbeddings(String inputFilePath, int embeddingVectorSize)
            throws IOException {
        HashMap<String, float[]> map = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] split = SPLIT_PATTERN.split(line);
                if (split.length != embeddingVectorSize + 1) {
                    throw new IllegalArgumentException(
                            "invalid embeddings used, encountered unexpected number of columns! "
                                    + split.length);
                }
                float[] vector = new float[split.length - 1];
                for (int i = 0; i < split.length - 1; i++) {
                    vector[i] = Float.parseFloat(split[i + 1]);
                }
                map.put(split[0], vector);
            }
        }
        return map;
    }

    private static SequenceFileWriter createWriter(String outputFolder,
                                                   boolean binary) throws IOException {
        if (binary) {
            return new SequenceFileWriter.BinaryWriter(outputFolder
                    + TRAIN_OUT_FILE_NAME);
        } else {
            return new SequenceFileWriter.TextWriter(outputFolder
                    + TRAIN_OUT_FILE_NAME);
        }
    }
}

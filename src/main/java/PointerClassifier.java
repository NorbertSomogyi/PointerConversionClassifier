import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.LabelLastTimeStepPreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.util.concurrent.TimeUnit;

public class PointerClassifier {
    public static boolean visualize = true;
    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 5;
        int numOutputs = 1;
        int numHiddenNodes = 4;

        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();
        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("training_set.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 5, 1);

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("validation_set.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 5, 1);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        model.fit(trainIter, nEpochs);

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }
        //An alternate way to do the above loop
        //Evaluation evalResults = model.evaluate(testIter);

        //Print the evaluation statistics
        System.out.println(eval.stats());

        System.out.println("\n****************Example finished********************");
        //Training is complete. Code that follows is for plotting the data & predictions only
        //generateVisuals(model, trainIter, testIter);
    }

    public static void generateVisuals(MultiLayerNetwork model, DataSetIterator trainIter, DataSetIterator testIter) throws Exception {
        if (visualize) {
            double xMin = 0;
            double xMax = 1.0;
            double yMin = -0.2;
            double yMax = 0.8;
            int nPointsPerAxis = 100;

            //Generate x,y points that span the whole range of features
            INDArray allXYPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);
            //Get train data and plot with predictions
            PlotUtil.plotTrainingData(model, trainIter, allXYPoints, nPointsPerAxis);
            TimeUnit.SECONDS.sleep(3);
            //Get test data, run the test data through the network to generate predictions, and plot those predictions:
            PlotUtil.plotTestData(model, testIter, allXYPoints, nPointsPerAxis);
        }
    }
}

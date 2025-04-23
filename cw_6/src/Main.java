import weka.core.Instances;
import weka.core.converters.C45Loader;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        C45Loader loader = new C45Loader();
        loader.setSource(new File("spambase/spambase.data"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        J48Classifier j48Classifier = new J48Classifier();
        j48Classifier.runJ48(data);
        NaiveBayesClassifier nbClassifier = new NaiveBayesClassifier();
        nbClassifier.runNaiveBayesVariants(data);

        CFS cfs = new CFS();
        String filename = "cfs.arff";
        cfs.newCFS(data, filename);
        DataSource source = new DataSource(filename);
        Instances data_cfs = source.getDataSet();
        data_cfs.setClassIndex(data_cfs.numAttributes() - 1);

        J48Classifier j48Classifier_cfs = new J48Classifier();
        j48Classifier.runJ48(data_cfs);
        NaiveBayesClassifier nbClassifier_cfs = new NaiveBayesClassifier();
        nbClassifier.runNaiveBayesVariants(data_cfs);

    }
}
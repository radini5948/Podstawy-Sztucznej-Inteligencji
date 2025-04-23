import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class J48Classifier {

    public void runJ48(Instances data) throws Exception {
        J48 j48 = new J48();
        j48.buildClassifier(data);

        Evaluation evalJ48 = new Evaluation(data);
        evalJ48.crossValidateModel(j48, data, 5, new Random(1));

        System.out.println("=== J48 ===");
        System.out.println(evalJ48.toSummaryString("\nWyniki:\n======\n", false));
        System.out.println("Macierz pomyłek:\n" + evalJ48.toMatrixString());
        System.out.println("Dokładność: " + (1 - evalJ48.errorRate()) * 100 + "%");
    }
}

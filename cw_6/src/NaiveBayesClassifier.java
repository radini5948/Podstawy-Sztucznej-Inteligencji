import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class NaiveBayesClassifier {

    public void runNaiveBayesVariants(Instances data) throws Exception {
        int numVariants = 3;
        NaiveBayes[] bayes = new NaiveBayes[numVariants];

        for (int i = 0; i < numVariants; i++) {
            bayes[i] = new NaiveBayes();

            if (i == 1) {
                bayes[i].setUseKernelEstimator(true);
            } else if (i == 2) {
                bayes[i].setUseSupervisedDiscretization(true);
            }

            bayes[i].buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(bayes[i], data, 5, new Random(1));

            System.out.println("=== NaiveBayes wariant " + i + " ===");
            System.out.println(eval.toSummaryString("\nWyniki:\n======\n", false));
            System.out.println("Macierz pomyłek:\n" + eval.toMatrixString());
            System.out.println("Dokładność: " + (1 - eval.errorRate()) * 100 + "%");
        }
    }
}

import weka.attributeSelection.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;

public class CFS {
    public void newCFS(Instances data, String FileName) throws Exception{
        AttributeSelection attrSelection = new AttributeSelection();
        ASEvaluation eval = new CfsSubsetEval();
        ASSearch search = new GreedyStepwise();
        attrSelection.setEvaluator(eval);
        attrSelection.setSearch(search);

        // Perform attribute selection
        attrSelection.SelectAttributes(data);
        Instances reducedData = attrSelection.reduceDimensionality(data);
        int[] selectedAttributes = attrSelection.selectedAttributes();
        System.out.println("Selected attributes: ");
        for (int i = 0; i < selectedAttributes.length; i++) {
            System.out.println((i + 1) + ": " + data.attribute(selectedAttributes[i]).name());
        }
        ArffSaver saver = new ArffSaver();
        saver.setInstances(reducedData);
        saver.setFile(new File(FileName));
        saver.writeBatch();


    }
}

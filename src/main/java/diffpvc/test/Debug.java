package diffpvc.test;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import diffpvc.C45;
import diffpvc.C45Lap;
import diffpvc.C45LeafLap;
import diffpvc.DiffPrivacyC45;
import diffpvc.Scorer.GiniScorer;

public class Debug {
public static void main(String[] args) throws Exception {
		
		int seed = 1; 
		Random random = new Random(seed);
		
		String[] paths = {
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff",
				//"E:\\Lectures\\TCloud\\dataset\\mushroom_nomissing.arff",
				//"E:\\Lectures\\TCloud\\dataset\\nursery\\nursery.arff"
		};
		
		String[] attrFile ={
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt",
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt",
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt"
		};
		
		// different budget
		String[] budgets = { "0.01","0.025", "0.05", "0.1" };
		
//		String[] classifierStrs = {"No DP", "SuLQID3","Exp", "Leaf Noisy"};
//		Classifier[] classifiers = {
//				new C45(), new SuLQID3(), new DiffPrivacyC45(), new C45_NoisyLeaf()
//		};
		
		GiniScorer gini = new GiniScorer();
		int maxDepth = 5;
		boolean unprune = true;
		int run = 3;
		boolean numCheck = false;
		
		for(int i=0; i< paths.length; i++){
			
			Instances trainData = (new DataSource(paths[i])).getDataSet();
			if (trainData.classIndex() == -1)
				trainData.setClassIndex(trainData.numAttributes() - 1);	
			
			System.out.println(
					"===========================\n"
					+"data set is "+paths[i]+"\n"
					+"data size is "+trainData.numInstances()+"\n"
					+"==========================");
			
			for(int j=0; j< budgets.length; j++){
				
				System.out.println(
						 "-------------------------------------------\n"
						+"Budget is "+budgets[j]+"\n"
						+"--------------------------------------------");
								
				
				
					C45Lap c45Lap = new C45Lap();
					c45Lap.setMaxDepth(maxDepth);
					c45Lap.setSeed(seed);
					c45Lap.setEpsilon(budgets[j]);
					c45Lap.setNumericAttributesFile(attrFile[i]);
					c45Lap.setUnpruned(unprune);
					c45Lap.setSkipNumInstancesChecks(!numCheck);
					Evaluation eval = new Evaluation(trainData);
					eval.crossValidateModel(c45Lap, trainData, 10, new Random(random.nextInt()));
					double sum = eval.pctCorrect();
				
				System.out.println("Laplace   \t"+sum);
			
				
			}
		}
		
		
	}
}

package diffpvc.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import diffpvc.C45;
import diffpvc.C45Lap;
import diffpvc.C45LeafLap;
import diffpvc.DiffPrivacyC45;
import diffpvc.Scorer.GiniScorer;

public class TestGini {
	public static void main(String[] args) throws Exception {
		
//		File log = new File("log.txt");
//		FileOutputStream logos = new FileOutputStream(log);
//		PrintStream ps = new PrintStream();
		
		
		int seed = 1; 
		Random random = new Random(seed);
		
		String[] paths = {
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff",
				"E:\\Lectures\\TCloud\\dataset\\mushroom_nomissing.arff",
				"E:\\Lectures\\TCloud\\dataset\\nursery\\nursery.arff"
		};
		
		String[] attrFile ={
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt",
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt",
				"E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt"
		};
		
		// different budget
		String[] budgets = { "0.01","0.025", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0", "2.0", "3.0" };
		
//		String[] classifierStrs = {"No DP", "SuLQID3","Exp", "Leaf Noisy"};
//		Classifier[] classifiers = {
//				new C45(), new SuLQID3(), new DiffPrivacyC45(), new C45_NoisyLeaf()
//		};
		
		GiniScorer gini = new GiniScorer();
		int maxDepth = 7;
		int run = 3;
		boolean unprune = false;
		boolean numCheck = true;
		
		System.out.println("======================================\n"
				+"Max Depth : " + maxDepth +"\n"
				+"Run times : " + run +"\n"
				+"Pruning   : " + !unprune +"\n"
				+"Num Check : " + numCheck +"\n"
				+"===============================");
		
		for(int i=0; i< paths.length; i++){
			
			Instances trainData = (new DataSource(paths[i])).getDataSet();
			if (trainData.classIndex() == -1)
				trainData.setClassIndex(trainData.numAttributes() - 1);	
			
			System.out.println(
					"===========================\n"
					+"data set is "+paths[i]+"\n"
					+"data size is "+trainData.numInstances()+"\n"
					+"==========================");
			
			String output = "";
			
			for(int j=0; j< budgets.length; j++){
				
				System.out.println(
						 "-------------------------------------------\n"
						+"Budget is "+budgets[j]+"\n"
						+"--------------------------------------------");
								
				double sum = 0;
				
				Evaluation eval = null;
				for(int k=0; k<run; k++){
				
					C45Lap c45Lap = new C45Lap();
					c45Lap.setMaxDepth(maxDepth);
					c45Lap.setSeed(random.nextInt());
					c45Lap.setEpsilon(budgets[j]);
					c45Lap.setNumericAttributesFile(attrFile[i]);
					c45Lap.setUnpruned(unprune);
					c45Lap.setSkipNumInstancesChecks(!numCheck);
					eval = new Evaluation(trainData);
					eval.crossValidateModel(c45Lap, trainData, 10, new Random(random.nextInt()));
					sum += eval.pctCorrect();
				}
				sum /= run;
				System.out.println("Laplace   \t"+sum);
			
				output += sum + " ";
				
				sum = 0;
				for(int k=0; k<run; k++){
				
					DiffPrivacyC45 diffPrivacyC45 = new DiffPrivacyC45();
					diffPrivacyC45.setMaxDepth(maxDepth);
					diffPrivacyC45.setScorer(gini);
					diffPrivacyC45.setSeed(random.nextInt());
					diffPrivacyC45.setEpsilon( budgets[j]);
					diffPrivacyC45.setNumericAttributesFile(attrFile[i]);
					diffPrivacyC45.setUnpruned(unprune);
					diffPrivacyC45.setSkipNumInstancesChecks(!numCheck);
					eval = new Evaluation(trainData);
					eval.crossValidateModel(diffPrivacyC45, trainData, 10, new Random(random.nextInt()));
					sum += eval.pctCorrect();
				}
				sum /= run;
				System.out.println("Exponential\t"+sum);
				output += sum + " ";
				
				sum = 0;
				for(int k=0; k<run; k++){
					C45LeafLap c45LeafLap = new C45LeafLap();
					c45LeafLap.setMaxDepth(maxDepth);
					c45LeafLap.setScorer(gini);
					c45LeafLap.setSeed(random.nextInt());
					c45LeafLap.setPrivacy( budgets[j] );
					c45LeafLap.setNumericAttributesFile(attrFile[i]);
					c45LeafLap.setUnpruned(unprune);
					c45LeafLap.setSkipNumInstancesChecks(!numCheck);
					eval.crossValidateModel(c45LeafLap, trainData, 10, new Random(random.nextInt()));
					sum += eval.pctCorrect();
				}
				sum /= run;
				System.out.println("Leaf Laplace\t"+sum);
				output += sum + " ";
				
				sum = 0;
				for(int k=0; k<run; k++){
					C45 c45 = new C45();
					c45.setMaxDepth(maxDepth);
					c45.setScorer(gini);
					c45.setSeed(random.nextInt());	
					c45.setNumericAttributesFile(attrFile[i]);
					c45.setUnpruned(unprune);
					c45.setSkipNumInstancesChecks(!numCheck);
					eval = new Evaluation(trainData);
					eval.crossValidateModel(c45, trainData, 10, new Random(random.nextInt()));
					sum += eval.pctCorrect();
				}
				sum /= run;
				System.out.println("No privacy\t"+sum);
				output += sum + "\n";
			}
			
			System.out.println(output);
			
			
		}
		
		
	}
}

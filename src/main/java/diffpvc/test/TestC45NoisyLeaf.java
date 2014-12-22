package diffpvc.test;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import diffpvc.AttributeScoreAlgorithm;
import diffpvc.C45;
import diffpvc.C45LeafLap;
import diffpvc.Scorer.GiniScorer;
import diffpvc.Scorer.InfoGainScorer;
import diffpvc.Scorer.MaxScorer;

public class TestC45NoisyLeaf {

	public static void main(String[] args) throws Exception {
		int seed = 1;
		Random random = new Random(seed);
		
		String trainDataPath = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff";
		String attrFile = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt";
		Instances trainData = null;
		
		//String testDataPath = "E:\\Lectures\\TCloud\\dataset\\bank-new.arff";
		//Instances testData  = null;
		
		int maxDepth = 4;
		
		String[] scorerStrs = {"Max","Gini","InfoGain"};
		AttributeScoreAlgorithm[] scorers = {new MaxScorer(), new GiniScorer(),new InfoGainScorer()};

		
		trainData = (new DataSource(trainDataPath)).getDataSet();
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);	
		

		
		for(int s=0; s< scorerStrs.length; s++){

				System.out.println(  
						 "============================================\n"
						+"Score: "+scorerStrs[s]+"\n"
						+"============================================"
				);

				//double sum = 0.0;
				for(int j=0; j<1; j++){
				
					C45LeafLap tree = new C45LeafLap();
					tree.setMaxDepth(maxDepth);
					tree.setConfidenceFactor(0.05f);
					tree.setSeed(random.nextInt());
					tree.setNumericAttributesFile(attrFile);
					tree.setScorer(scorers[s]);
					tree.setPrivacy(1.0);
					
					//tree.setDebug(true);
					//tree.setEpsilon(eStrs[i]);
					tree.buildClassifier(trainData);
					
					Evaluation eval = new Evaluation(trainData);
					eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));
					
					System.out.println("Run ["+j+"] Accuracy is "+eval.pctCorrect());
					//sum += eval.pctCorrect();
				}
//				double avg = sum/10.0;
//				System.out.println("Average Accuracy is "+avg);
			
		}

	}

}

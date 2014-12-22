package diffpvc.test;


import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import diffpvc.AttributeScoreAlgorithm;
import diffpvc.DiffPrivacyC45;
import diffpvc.DiffPrivacyID3;
import diffpvc.DiffPrivacyID3_VLDB;
import diffpvc.Scorer.GiniScorer;
import diffpvc.Scorer.InfoGainScorer;
import diffpvc.Scorer.MaxScorer;
import diffpvc.Scorer.RandomScorer;

public class TestDiffPrivacyC45 {

	
	/**
	 * @param
	 * TrainData: "adult" without missing values
	 * MaxDepth: 5
	 * Scorer: Max, Gini, InfoGain,
	 * Privacy Budget: 0.1, 0.5, 0.75, 1, 2, ... 10
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		//String[] scorers = ["InfoGain","Max","Gini"];

		int seed = 1;
		Random random = new Random(seed);
		
		String trainDataPath = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff";
		String attrFile = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt";
		Instances trainData = null;
		
		//String testDataPath = "E:\\Lectures\\TCloud\\dataset\\bank-new.arff";
		//Instances testData  = null;
		
		int maxDepth = 5;
		
		String[] scorerStrs = {"Max","Gini","InfoGain"};
		AttributeScoreAlgorithm[] scorers = {new MaxScorer(), new GiniScorer(),new InfoGainScorer()};
		String[] eStrs = {
						  "0.1","0.5",
						  "0.75","1.0","2.0",
						  "3.0","4.0","5.0",
						  "6.0","7.0","8.0","9.0","10.0"};
		
		trainData = (new DataSource(trainDataPath)).getDataSet();
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);	
		
//		testData = (new DataSource(testDataPath)).getDataSet();
//		if (testData.classIndex() == -1)
//			testData.setClassIndex(testData.numAttributes() - 1);	
		
		
		for(int s=0; s<scorers.length; s++){
			
		
			for(int i=0; i<eStrs.length; i++){
			
				System.out.println(  
						 "============================================\n"
						+"Score: "+scorerStrs[s]+"; Budget: "+eStrs[i]+"\n"
						+"============================================"
				);

				double sum = 0.0;
				for(int j=0; j<10; j++){
				
					DiffPrivacyC45 tree = new DiffPrivacyC45();
					tree.setMaxDepth(maxDepth);
					tree.setConfidenceFactor(0.25f);
					tree.setSeed(random.nextInt());
					tree.setNumericAttributesFile(attrFile);
					tree.setScorer(scorers[s]);
					tree.setEpsilon(eStrs[i]);
					//tree.buildClassifier(trainData);
					
					Evaluation eval = new Evaluation(trainData);
					eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));
					
					System.out.println("Run ["+j+"] Accuracy is "+eval.pctCorrect());
					sum += eval.pctCorrect();
				}
				double avg = sum/10.0;
				System.out.println("Average Accuracy is "+avg);
			}
		}
		
	}

}

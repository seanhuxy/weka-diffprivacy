package diffpvc.test;

import java.util.Random;

import diffpvc.RDTs.RandomForestDp;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class TestDiffPrivacyRDTsPaper3 {
	
	public static void main(String[] args) throws Exception {
	
		int seed = 1;
		Random random = new Random(seed);
		
		String trainDataPath = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff";
		//String trainDataPath = "E:\\Lectures\\TCloud\\dataset\\mushroom_nomissing.arff";
		
		String attrFile = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.arff";
		
		
		Instances trainData  = Utils4Test.getDataFromFile(trainDataPath);
		
		//double[] privacyBudgets = {0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
		double privacyBudget = 1000.0/(double)trainData.numInstances();
		
		String[] evalMethods = {"MV","TA"};
		
		int[] numTreess = {1,3,11};
		int[] maxDepths = new int[7];
		for(int i=0; i< maxDepths.length; i++){
			maxDepths[i] = i+1;
		}
		
		String csv = "";
		
		for(int i=0; i< numTreess.length; i++){
			int numTrees = numTreess[i];
			
			csv +="numTrees: "+numTrees+"\n";
			
			System.out.println(
					"---------------------------------------\n"
					+"numTrees = "+numTrees+"\n"
					+"-------------------------------------"
					);
			
			for(int j=0; j< evalMethods.length; j++){
				String evalMethod = evalMethods[j];

				for(int k=0; k< maxDepths.length; k++){
					int maxDepth = maxDepths[k];				
					System.out.println(
							 "=======================================\n"
							+"Budget: "+privacyBudget+",\n"
							+"Method: "+evalMethod+",\n"
							+"Trees : "+numTrees+",\n"
							+"Depth : "+maxDepth);
					
					double sum = 0;
					for(int l=0; l< 10; l++){
						RandomForestDp rf = new RandomForestDp();	
						rf.setPrivacy(privacyBudget);
						rf.setEvalMethod(evalMethod);
						rf.setNumTrees(numTrees);
						rf.setMaxDepth(maxDepth);
						rf.setSeed(random.nextInt());
						
						//setAttrFile(attrFile);
						
						Evaluation eval = new Evaluation(trainData);
						eval.crossValidateModel(rf, trainData, 10, new Random(random.nextInt()));						
						sum += eval.pctCorrect();
					}
					double avg = sum/10.0;
					
					System.out.println("Avg Accuracy: "+avg);				
					System.out.println("==================================");
				
					csv+= avg+",";
				}
				csv += "\n";
			}
			
		}
		System.out.println("output");
		System.out.println(csv);
		
		
	}
}

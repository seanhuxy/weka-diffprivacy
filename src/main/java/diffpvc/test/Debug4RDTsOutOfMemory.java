package diffpvc.test;

import weka.core.Instances;
import diffpvc.RDTs.RandomForestDp;

public class Debug4RDTsOutOfMemory {
	
	public static void main(String[] args) throws Exception {
		
		String trainDataPath = "E:\\Lectures\\TCloud\\dataset\\mushroom_nomissing.arff";
		Instances trainData  = Utils4Test.getDataFromFile(trainDataPath);
		
		double privacyBudget = 1000.0/(double)trainData.numInstances();
		
		int numTrees = 1;
		int maxDepth = 9;
		
		System.out.println(
				"---------------------------------------\n"
				+"Trees: "+numTrees+"\n"
				+"Depth: "+maxDepth+"\n"
				+"-------------------------------------"
				);
		
		RandomForestDp rf = new RandomForestDp();	
		rf.setPrivacy(privacyBudget);
		rf.setEvalMethod("MV");
		rf.setNumTrees(numTrees);
		rf.setMaxDepth(maxDepth);
		rf.setSeed(1);
		
		rf.buildClassifier(trainData);
	}

}

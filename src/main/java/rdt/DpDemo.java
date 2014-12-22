package rdt;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DpDemo {
	
//	protected RandomForestDp randomForest= null;
//	
//	/*
//	 * parameter
//	 */
//	
//	protected double m_Privacy = 1.0;
//	
//	protected String m_EvalMethod = null; // MV,TA,PA
//	
//	protected int m_NumTrees = 1;
//	
//	protected int m_treeHeight = 1;
	
	protected static Instances getDataFromFile(String path) throws Exception{		
		try{
			DataSource source = new DataSource(path);
			Instances data = source.getDataSet();
			
			// setting class attribute if the data format does not provide this information
			// For example, the XRFF format saves the class attribute information as well
			if (data.classIndex() == -1)
			  data.setClassIndex(data.numAttributes() - 1);	 
			return data;
		}catch(Exception e){
			System.err.println(path + " is not a legel path");
			System.exit(1);
		}
		return null;
	}
	
	/**
	 * 
	 * @param args 	
	 * 	-d: path of data set for training (*.arff)
	 * 	-D: path of data set for testing  (*.arff)
	 * 	-p: differential privacy, default = 1.0
	 * 	-m:	evaluation method, MV, PA, TA, default = MV,
	 * 	-n: number of trees, default = log2(|training data set|)
	 * 	-h: height of trees, default = the number of attribute, 
	 * 		if > the number of attribute, = the number of attribute
	 * 	-s: seed for random,	default = 1
	 * 
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		//DpDemo demo = new DpDemo();
		
		String 	trainPath = "",
				testPath  = "",
				evalMethod= "MV";
		double  privacy   = 1.0;
		int 	numTrees  = 0,	// = log(size of data set)
				height	  = 0,	// = number of attributes
				seed 	  = 1;
		
		Instances 	trainData = null,
					testData  = null;
		
		int i = 0;
		String arg;
		while( i < args.length && args[i].startsWith("-")){
			arg = args[i++];	
			
			if(arg.equals("-d")){
				if(i<args.length){
					trainPath = args[i++];
					trainData = getDataFromFile(trainPath);
				}else{
					System.err.println("-d requires a file path");
					System.exit(1);
				}
			}else if (arg.equals("-D")){
				if(i<args.length){
					testPath = args[i++];  
					testData = getDataFromFile(testPath);
				}else{
					System.err.println("-D requires a file path");
					System.exit(1);
				}
			}else if (arg.equals("-p")){
				if(i<args.length){
					try {
						privacy = Double.parseDouble(args[i++]);  
					}catch(NumberFormatException e){
						System.err.println("-p" + args[i-1] + " must be a Double.");
						System.exit(1);
					}
					if(privacy < 0){
						System.err.println("-p " + privacy + " must >= 0.");
						System.exit(1);
					}
				}else{
					System.err.println("-p requires a real number");
					System.exit(1);
				}	
			}else if (arg.equals("-m")){
				if(i<args.length){
					evalMethod = args[i++];
					if( !evalMethod.equals("MV") 
					 && !evalMethod.equals("TA")
					 && !evalMethod.equals("PA")){
						System.err.println("-m requires MV, TA or PA");
						System.exit(1);
					}
				}else{
					System.err.println("-m requires a file path");
					System.exit(1);
				}
			}else if (arg.equals("-n")){
				if(i<args.length){
					try {
						numTrees = Integer.parseInt(args[i++]);  
					}catch(NumberFormatException e){
						System.err.println("-n " + args[i-1] + " must be a Integer.");
						System.exit(1);
					}
					if(numTrees <= 0){
						System.err.println("-n " + numTrees + " must be > 0.");
						System.exit(1);
					}
				}else{
					System.err.println("-n requires an Integer");
					System.exit(1);
				}	
			}else if (arg.equals("-h")){
				if(i<args.length){
					try {
						height = Integer.parseInt(args[i++]);  
					}catch(NumberFormatException e){
						System.err.println("-h " + args[i-1] + " must be a Integer.");
						System.exit(1);
					}
					if(height <= 0){
						System.err.println("-h " + height + " must > 0.");
						System.exit(1);
					}
				}else{
					System.err.println("-h requires an Integer");
					System.exit(1);
				}	
			}else if (arg.equals("-s")){
				if(i<args.length){
					try {
						seed = Integer.parseInt(args[i++]);  
					}catch(NumberFormatException e){
						System.err.println("-s " + args[i-1] + " must be a Integer.");
						System.exit(1);
					}
				}else{
					System.err.println("-s requires an Integer");
					System.exit(1);
				}
			}
			
		}
		
		if(trainData == null){
			System.err.println("There's no training data file specified");
			System.exit(1);
		}
		if(numTrees == 0){
			numTrees = (int)(Math.log(trainData.numInstances())/Math.log(2));
		}
		if(height == 0 || height > trainData.numAttributes() ){
			height = trainData.numAttributes();
		}

		System.out.println("Differential Privacy Demo:\n"
				+"-d "+trainPath+"\n"
				+"-p "+privacy+"\n"
				+"-m "+evalMethod+"\n"
				+"-n "+numTrees+"\n"
				+"-h "+height+"\n"
				+"-s "+seed+"\n"
			);
		
		RandomForestDp rf = new RandomForestDp();
		
		rf.setPrivacy(privacy);
		rf.setEvalMethod(evalMethod);
		rf.setNumTrees(numTrees);
		rf.setMaxDepth(height);
		rf.setSeed(seed);
		
		rf.buildClassifier(trainData);
		
		Evaluation eval = new Evaluation(trainData);
		if(testData == null){
			eval.crossValidateModel(rf, trainData, 10, new Random(seed));
		}else{
			eval.evaluateModel(rf, testData);		
		}
		System.out.println(eval.toSummaryString());
	}
}

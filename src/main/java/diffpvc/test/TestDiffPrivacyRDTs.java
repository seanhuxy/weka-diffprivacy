package diffpvc.test;

import java.util.Random;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;

import diffpvc.RDTs.RandomForestDp;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestDiffPrivacyRDTs {
	
	private String DEFAULT_TRAIN_PATH ="E:\\Lectures\\TCloud\\dataset\\adult_nomissing.arff";
	private String DEFAULT_ATTR_FILE = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt";
	
	private String 	trainPath = DEFAULT_TRAIN_PATH,
					testPath  = "",
					attrFile  = DEFAULT_ATTR_FILE,
					evalMethod= "PA";
	private double  privacy   = 1.0;
	private int 	numTrees  = 0,	// = log(size of data set)
					maxDepth  = 0,	// = number of attributes/2
					seed 	  = 1;
	
	private Instances 	trainData = null,
						testData  = null;
	
	
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
	
	private void getOptions(CommandLine cmd){

		if(cmd.hasOption("t")){
			try{
				trainPath = cmd.getOptionValue("t");								
			}catch(Exception e){
				System.err.println("-t requires a file path");
				System.exit(1);
			}
		}
		if(cmd.hasOption("T")){
			try{
				testPath = cmd.getOptionValue("T");  
			}catch(Exception e){
				System.err.println("-T requires a file path");
				System.exit(1);
			}
		}
		if(cmd.hasOption("p")){
			try {
				privacy = Double.parseDouble(cmd.getOptionValue("p"));  
				if(privacy < 0){
					throw new NumberFormatException();
				}
			}catch(NumberFormatException e){
				System.err.println("-p" + cmd.getOptionValue("p") + " must be a Double.");
				System.exit(1);
			}
		}
		if(cmd.hasOption("m")){
			evalMethod = cmd.getOptionValue("m");
			if( !evalMethod.equals("MV") 
			 && !evalMethod.equals("TA")
			 && !evalMethod.equals("PA")){
				System.err.println("-m requires MV, TA or PA");
				System.exit(1);
			}
		}
		if(cmd.hasOption("n")){
			numTrees = Integer.parseInt(cmd.getOptionValue("n")); 
		}
		if(cmd.hasOption("d")){
			maxDepth = Integer.parseInt(cmd.getOptionValue("d")); 
		}
		if(cmd.hasOption("s")){
			seed = Integer.parseInt(cmd.getOptionValue("s"));  
		}
	}
	
	private void print(){
		System.out.println("Differential Privacy Demo:\n"
				+"-d "+trainPath+"\n"
				+"-p "+privacy+"\n"
				+"-m "+evalMethod+"\n"
				+"-n "+numTrees+"\n"
				+"-d "+maxDepth+"\n"
				+"-s "+seed+"\n"			
		);
	}
	
	/**
	 * 
	 * @param args 	
	 * 	-t: path of data set for training (*.arff)
	 * 	-T: path of data set for testing  (*.arff)
	 * 	-p: differential privacy, default = 1.0
	 * 	-m:	evaluation method, MV, PA, TA, default = MV,
	 * 	-n: number of trees, default = log2(|training data set|)
	 * 	-d: height of trees, default = the number of attribute, 
	 * 		if > the number of attribute, = the number of attribute
	 * 	-s: seed for random,	default = 1
	 * 	-h: get help
	 * 
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		TestDiffPrivacyRDTs test = new TestDiffPrivacyRDTs();
		
		Options options = new Options();
		options.addOption("t", "train", true, "Training Data Path")
			   .addOption("T", "test", true, "Test Data Path")
			   .addOption("p", "privacy", true, "Privacy Budget(Float)")
			   .addOption("m", "method", true, "Evaluation Method: MV, PA, TA")
			   .addOption("n", "ntrees", true, "Number of trees")
			   .addOption("d", "depth", true, "Max Depth of the decision tree(Int)")
			   .addOption("s", "seed", true, "Random Seed(Int)")
			   .addOption("h", "help", false,"Print this message");		
		CommandLineParser parser = new BasicParser();	
		CommandLine cmd = parser.parse(options, args);
		if(cmd.hasOption("h")){
			HelpFormatter helper = new HelpFormatter();
			helper.printHelp("Test DiffPrivacy Random Decision Trees", options);
			System.exit(0);
		}
	
		test.getOptions(cmd);		
		test.trainData = getDataFromFile(test.trainPath);
		//test.testData  = getDataFromFile(test.testPath);

		if(test.numTrees == 0){
			test.numTrees = (int)(Math.log(test.trainData.numInstances())/Math.log(2));
		}
		if(test.maxDepth == 0 || test.maxDepth > test.trainData.numAttributes() ){
			test.maxDepth = test.trainData.numAttributes()/2;
		}
		
		test.print();
		
		Random random = new Random(test.seed);
		
		RandomForestDp rf = new RandomForestDp();	
		rf.setPrivacy(test.privacy);
		rf.setEvalMethod(test.evalMethod);
		rf.setNumTrees(test.numTrees);
		rf.setMaxDepth(test.maxDepth);
		rf.setSeed(random.nextInt());
		
		rf.setAttrFile(test.attrFile);
		
		//rf.buildClassifier(test.trainData);
		
		Evaluation eval = new Evaluation(test.trainData);
		eval.crossValidateModel(rf, test.trainData, 10, new Random(random.nextInt()));

		System.out.println(eval.toSummaryString());
	}
}

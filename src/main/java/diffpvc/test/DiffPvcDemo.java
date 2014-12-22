package diffpvc.test;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;

import diffpvc.DiffPrivacyC45;
import diffpvc.Scorer.GiniScorer;

public class DiffPvcDemo {

	/**
	 * @param args
	 * 		- d	train data path
	 * 		- D test  data path
	 * 		
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options options = new Options();
		options.addOption("c", "classifier", true, "Classifier: ID3, C45, CART, RDT, RF")
			   .addOption("S", "scorer", true, "Scorer: InfoGain, Gini, Max")
			   .addOption("n", "noise",	true, "Mechanism to add noise: LAP, EXP")
			   .addOption("t", "train", true, "Training Data Path")
			   .addOption("T", "test", true, "Test Data Path")
			   .addOption("p", "privacy", true, "Privacy Budget(Float)")
			   .addOption("d", "depth", true, "Max Depth of the decision tree(Int)")
			   .addOption("s", "seed", true, "Random Seed(Int)")
			   .addOption("h", "help", false,"Print this message");
		
		CommandLineParser parser = new BasicParser();	
		CommandLine cmd = parser.parse(options, args);
		
		String trainPath = null;
		Instances trainData = null;
		
		String testPath = null;
		Instances testData = null;
		
		String eString = "11.0";	
		int seed = 2;
		
		if(cmd.hasOption("h")){
			HelpFormatter helper = new HelpFormatter();
			helper.printHelp("diffpvc", options);
			System.exit(0);
		}
		
		if(cmd.hasOption("t")){
			trainPath = cmd.getOptionValue("t");
			trainData = (new DataSource(trainPath)).getDataSet();
			if (trainData.classIndex() == -1)
				trainData.setClassIndex(trainData.numAttributes() - 1);	
		}else{
			
		}
		if(cmd.hasOption("T")){
			testPath  = cmd.getOptionValue("D");
			testData = (new DataSource(testPath)).getDataSet();
			if (testData.classIndex() == -1)
				testData.setClassIndex(testData.numAttributes() - 1);	
		}
		
		DiffPrivacyC45 tree = new DiffPrivacyC45();
		tree.setScorer(new GiniScorer());
		
		//tree.setDebug(true);
		tree.setEpsilon(eString);
		tree.setSeed(seed);
		
		tree.buildClassifier(trainData);
		
		Evaluation eval = new Evaluation(trainData);
		if(testData == null){
			eval.crossValidateModel(tree, trainData, 10, new Random(4));
//		}else{
//			eval.evaluateModel(sulqID3, testData);		
		}
		System.out.println(eval.toSummaryString());
	}

}

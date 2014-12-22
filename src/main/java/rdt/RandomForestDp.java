package rdt;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class RandomForestDp extends RandomForest {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected BaggingDp m_bagger = null;
	
	/**
	 * Differential privacy parameter
	 *	Input by user
	 */
	protected double m_Privacy = 0.0;
	
	protected String m_EvalMethod = "MV";

	public double getPrivacy() {
		return m_Privacy;
	}
	public void setPrivacy(double m_Privacy) {
		this.m_Privacy = m_Privacy;
	}
	public String getEvalMethod() {
		return m_EvalMethod;
	}
	public void setEvalMethod(String m_EvalMethod) {
		this.m_EvalMethod = m_EvalMethod;
	}
	
	/*
	 * input by user
	 */
	//privacy
	//m_randomSeed  = 
	//m_MaxDepth
	//m_evalMethod
	//max and min value of every numeric attribute ( from dataset)
	
	/*
	 * determined by paper
	 */
	//m_numTrees = log(size of dataset)
	
	/*
	 * useless
	 */
	//m_numFeatures = num of attributes // useless,
	//m_KValue = 0;		// not understand
	

	/**
	   * Builds a classifier for a set of instances.
	   * 
	   * @param data the instances to train the classifier with
	   * @throws Exception if something goes wrong
	   * 
	   * Dp Mode£º
	   * @Input	train data, num of trees, depth of trees, privacy level,
	   * 		random's seed
	   */
	  @Override
	  public void buildClassifier(Instances data) throws Exception {

	    // can classifier handle the data?
	    getCapabilities().testWithFail(data);

	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();

	    m_bagger = new BaggingDp();
	    RandomTreeDp rTree = new RandomTreeDp();

	    
	    // set up the random tree options
	    // Dp mode, useless, because of the fixed tree depth
//	    m_KValue = m_numFeatures;
//	    if (m_KValue < 1)
//	      m_KValue = (int) Utils.log2(data.numAttributes()) + 1;
//	    rTree.setKValue(m_KValue);
	    
	    // Max Depth of a tree
	    // In Dp mode, should be a fixed depth
	    rTree.setMaxDepth(getMaxDepth());
	    rTree.setPrivacy(getPrivacy());
	    
	    // set up the bagger and build the forest
	    m_bagger.setEvalMethod(getEvalMethod());
	    m_bagger.setClassifier(rTree);
	    m_bagger.setSeed(m_randomSeed);
	    m_bagger.setNumIterations(m_numTrees);	// Dp mode, the number of trees
	    
	    //m_bagger.setCalcOutOfBag(true);
//	    m_bagger.setCalcOutOfBag(false);	// Dp mode, useless,
	    
	    m_bagger.buildClassifier(data);
	  }
	  
	  /**
	   * Returns the class probability distribution for an instance.
	   * 
	   * @param instance the instance to be classified
	   * @return the distribution the forest generates for the instance
	   * @throws Exception if computation fails
	   */
	  @Override
	  public double[] distributionForInstance(Instance instance) throws Exception {
		  
	    return m_bagger.distributionForInstance(instance);
	  }
}

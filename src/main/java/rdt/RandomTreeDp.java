package rdt;

import java.io.Serializable;
import java.util.Random;

import weka.classifiers.trees.RandomTree;

import weka.core.Attribute;
//import weka.classifiers.trees.RandomTree.Tree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class RandomTreeDp extends RandomTree {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected TreeDp m_Tree;
	
	protected double m_Privacy;

	public double getPrivacy() {
		return m_Privacy;
	}
	public void setPrivacy(double m_Privacy) {
		this.m_Privacy = m_Privacy;
	}

	/**
	   * Builds classifier.
	   * 
	   * @param data the data to train with
	   * @throws Exception if something goes wrong or the data doesn't fit
	   */
	  @Override
	  public void buildClassifier(Instances data) throws Exception {

	    // Make sure K value is in range
//	    if (m_KValue > data.numAttributes() - 1)
//	      m_KValue = data.numAttributes() - 1;
//	    if (m_KValue < 1)
//	      m_KValue = (int) Utils.log2(data.numAttributes()) + 1;

	    // can classifier handle the data?
	    getCapabilities().testWithFail(data);

	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();

	    // only class? -> build ZeroR model
//	    if (data.numAttributes() == 1) {
//	      System.err
//	          .println("Cannot build model (only class attribute present in data!), "
//	              + "using ZeroR model instead!");
//	      m_zeroR = new weka.classifiers.rules.ZeroR();
//	      m_zeroR.buildClassifier(data);
//	      return;
//	    } else {
//	      m_zeroR = null;
//	    }

	    // Figure out appropriate datasets
	    Instances train = data;
//	    Instances train = null;
//	    Instances backfit = null;
	    Random rand = data.getRandomNumberGenerator(m_randomSeed);
//	    if (m_NumFolds <= 0) {
//	      train = data;
//	    } else {
//	      data.randomize(rand);
//	      data.stratify(m_NumFolds);
//	      train = data.trainCV(m_NumFolds, 1, rand);
//	      backfit = data.testCV(m_NumFolds, 1);
//	    }

	    // Create the attribute indices window
	    int[] attIndicesWindow = new int[data.numAttributes() - 1];
	    int j = 0;
	    for (int i = 0; i < attIndicesWindow.length; i++) {
	      if (j == data.classIndex())
	        j++; // do not include the class
	      attIndicesWindow[i] = j++;
	    }

	    //XXX for what?
	    // Compute initial class counts
//	    double[] classProbs = new double[train.numClasses()];
//	    for (int i = 0; i < train.numInstances(); i++) {
//	      Instance inst = train.instance(i);
//	      classProbs[(int) inst.classValue()] += inst.weight();
//	    }

	    // Build tree
	    m_Tree = new TreeDp();
	    m_Info = new Instances(data, 0);
	    m_Tree.buildTree(null, null, attIndicesWindow, rand, 0);
	    
	    m_Tree.trainInstances(train);
	    m_Tree.fillEmptyLeaf(rand);
	    
	    if( getPrivacy() > 0){
	    	m_Tree.addLapError(getPrivacy(),rand);
	    }
	    
	  }
	  
	  /**
	   * Computes class distribution of an instance using the tree.
	   * 
	   * @param instance the instance to compute the distribution for
	   * @return the computed class probabilities
	   * @throws Exception if computation fails
	   */
	  @Override
	  public double[] distributionForInstance(Instance instance) throws Exception {

	      return m_Tree.distributionForInstance(instance);
	    
	  }
	
	  protected class TreeDp implements Serializable {

		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		
		/** The subtrees appended to this tree. */
	    protected TreeDp[] m_Successors;
	    
	    /** The attribute to split on. */
	    protected int m_Attribute = -1;
	    
	    /** The split point. */
	    protected double m_SplitPoint = Double.NaN;

	    /** Class probabilities from the training data. */
	    protected double[] m_ClassDistribution = null;

	    
		/**
	     * Recursively generates a tree.
	     * 
	     * @param data the data to work with
	     * @param classProbs the class distribution
	     * @param attIndicesWindow the attribute window to choose attributes from
	     * @param random random number generator for choosing random attributes
	     * @param depth the current depth
	     * @throws Exception if generation fails
	     */
	    protected void buildTree(Instances data, double[] classProbs,
	        int[] attIndicesWindow, Random random, int depth) throws Exception {

	      //XXX Make leaf if there are no training instances
//	      if (data.numInstances() == 0) {
//	        m_Attribute = -1;
//	        m_ClassDistribution = null;
//	        m_Prop = null;
//	        return;
//	      }

	      // Check if node doesn't contain enough instances or is pure
	      // or maximum depth reached
	      //m_ClassDistribution = classProbs.clone();

	      //if (Utils.sum(m_ClassDistribution) < 2 * m_MinNum
	          //||Utils.eq(m_ClassDistribution[Utils.maxIndex(m_ClassDistribution)], Utils.sum(m_ClassDistribution))
	          //||((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {       
	      
	    	// Make leaf
	      if( ((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {
	    	  //System.out.println("Max Depth: "+getMaxDepth()+", height: "+depth);
	        m_Attribute = -1;
	        // m_Prop = null;
	        //m_ClassDistribution	        
	        return;
	      }

	      int attrIndex = random.nextInt(attIndicesWindow.length);	      
	      m_Attribute = attIndicesWindow[attrIndex];
	      
	      // for nominal attribute
	      if( m_Info.attribute(m_Attribute).isNominal() ){
	    	  int numAttrValues = m_Info.attribute(m_Attribute).numValues();
	    	  m_Successors = new TreeDp[numAttrValues];
	    	  for (int i = 0; i < numAttrValues; i++) {
	    		  m_Successors[i] = new TreeDp();
		          m_Successors[i].buildTree(null, null, attIndicesWindow, random, depth + 1);	    		  
	    	  }
	      }	    		  
	      
	      // for numeric attribute 
	      else{	    	  
	    	  //m_SplitPoint = m_Info.attribute(m_Attribute).random(random);
	    	  //m_SplitPoint = random.nextDouble();
	    	  Attribute attr = m_Info.attribute(m_Attribute);
	    	  
	    	  m_SplitPoint = attr.getLowerNumericBound()
	    			  +(attr.getUpperNumericBound()-attr.getLowerNumericBound())*random.nextDouble();
	    	 
	    	  m_Successors = new TreeDp[2];
	    	  for (int i = 0; i < m_Successors.length; i++) {
	    		  m_Successors[i] = new TreeDp();
		          m_Successors[i].buildTree(null, null, attIndicesWindow, random, depth + 1);	    		
	    	  }
	      }
	    }
	    
	    protected void trainInstance(Instance instance){

	    	// Node is not a leaf
	        if (m_Attribute > -1) {	          	          
	          
	        	if (instance.isMissing(m_Attribute)) {  
	        	// Value is missing
	        	  
	          } else if (m_Info.attribute(m_Attribute).isNominal()) {
	            // For nominal attributes
	        	m_Successors[(int) instance.value(m_Attribute)].trainInstance(instance);  
	        
	          } else {	
	        	// For numeric attributes  	            
	            if (instance.value(m_Attribute) < m_SplitPoint) {
	            	m_Successors[0].trainInstance(instance);
	            } else {
	            	m_Successors[1].trainInstance(instance);
	            }
	          }
	          return;
	        }

	        // Node is a leaf or successor is empty?
	        if (m_Attribute == -1) {

//	        	try{
	        	if(m_ClassDistribution == null){
	        		m_ClassDistribution = new double[m_Info.numClasses()];	          		
	        	}
	   
	        	m_ClassDistribution[(int)instance.classValue()]++;
//	        	}
//	        	catch(ArrayIndexOutOfBoundsException e){
//	        		System.err.println("instance:"+instance.classIndex()+", m_Info "+m_Info.numClasses() );
//	        	
//	        	}
	        	return;
	        }
	    }
	    
	    protected void fillEmptyLeaf(Random random){
	    	if (m_Attribute == -1) {
	    		if(m_ClassDistribution == null){
	    			m_ClassDistribution = new double[m_Info.numClasses()];
	    			m_ClassDistribution[random.nextInt(m_Info.numClasses())]++;
	    		}
	    		return;
	    	}else {
	    		for(int i=0;i<m_Successors.length;i++){
	    			m_Successors[i].fillEmptyLeaf(random);
	    		}
	    	}
	    	
	    }
	    	
	    protected void trainInstances(Instances data){
	    	
	    	for(int i=0; i< data.numInstances(); i++){
	    		Instance inst = data.instance(i);
	    		trainInstance(inst);
	    	}
	    }
	    
	    protected void addLapError(double budget, Random random){
	    	
	    	// non leaf node
	    	if( m_Attribute != -1 ){
	    		for(int i=0; i<m_Successors.length; i++){
	    			m_Successors[i].addLapError(budget, random);
	    		}
	    	}
	    	
	    	// leaf node
	    	else{
	    		for(int i=0; i<m_ClassDistribution.length; i++){
	    			m_ClassDistribution[i] += rdt.Utils.laplace(budget, random);
	    		}
	    		
	    		//XXX if any of them is less than 0, sum of them equals to 0;
	    	}
	    }
	    
	    /**
	     * Computes class distribution of an instance using the decision tree.
	     * 
	     * @param instance the instance to compute the distribution for
	     * @return the computed class distribution
	     * @throws Exception if computation fails
	     */
	    public double[] distributionForInstance(Instance instance) throws Exception {

	      double[] returnedDist = null;

	      
	      // Node is not a leaf
	      if (m_Attribute > -1) {

	        
	        if (instance.isMissing(m_Attribute)) {
	        	System.err.println("Testing phase: can't deal with missing attribute");
	        	System.exit(1);
	        	// Value is missing
//	          returnedDist = new double[m_Info.numClasses()];
//
//	          // Split instance up
//	          for (int i = 0; i < m_Successors.length; i++) {
//	            double[] help = m_Successors[i].distributionForInstance(instance);
//	            if (help != null) {
//	              for (int j = 0; j < help.length; j++) {
//	                returnedDist[j] += m_Prop[i] * help[j];
//	              }
//	            }
//	          }
	        	
	        // For nominal attributes
	        } else if (m_Info.attribute(m_Attribute).isNominal()) { 
	          returnedDist = m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
	        } else {

	          // For numeric attributes
	          if (instance.value(m_Attribute) < m_SplitPoint) {
	            returnedDist = m_Successors[0].distributionForInstance(instance);
	          } else {
	            returnedDist = m_Successors[1].distributionForInstance(instance);
	          }
	        }
	      }

	      // Node is a leaf or successor is empty?
	      if (m_Attribute == -1) {

	        // Is node empty?
	        if (m_ClassDistribution == null) {
	        	System.err.println("Testing Parse: class distribution of leaf node shouldn't be null");
	        	System.exit(1);
//	          if (getAllowUnclassifiedInstances()) {
//	            return new double[m_Info.numClasses()];
//	          } else {
//	            return null;
//	          }
	        }

	        // Else return normalized distribution
	        double[] normalizedDistribution = m_ClassDistribution.clone();
	        Utils.normalize(normalizedDistribution);
	        return normalizedDistribution;
	      } else {
	        return returnedDist;
	      }
	    }
	}
}

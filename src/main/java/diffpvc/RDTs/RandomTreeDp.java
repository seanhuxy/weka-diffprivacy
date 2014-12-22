package diffpvc.RDTs;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import diffpvc.C45Attribute;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;

public class RandomTreeDp extends Classifier implements Randomizable {

	private static final long serialVersionUID = 1L;

	protected TreeDp m_Tree;

	protected int seed = 1;

	protected Random random;

	protected double m_Privacy;

	protected int m_MaxDepth;

	/**
	 * Contains numClasses
	 */
	private Instances m_Info;

	protected List<C45Attribute> candidateAttributes;

	public void setCandidateAttributes(List<C45Attribute> candidateAttributes) {
		this.candidateAttributes = candidateAttributes;
	}
	
	/**
	 * Builds classifier.
	 * 
	 * @param data
	 *            the data to train with
	 * @throws Exception
	 *             if something goes wrong or the data doesn't fit
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		// Create the attribute indices window
		List<C45Attribute> attrWindow = new LinkedList<C45Attribute>();
		for (int i = 0; i < candidateAttributes.size(); i++) {
			attrWindow.add(candidateAttributes.get(i));
		}

		m_Info = new Instances(data, 0);
		
		// Build tree
		m_Tree = new TreeDp();
		m_Tree.buildTree(attrWindow, getMaxDepth());

		m_Tree.trainInstances(data);
		
		/*
		 * In the 2nd paper, there's no this step
		 */
		m_Tree.fillEmptyLeaf();	

		if (getPrivacy() > 0) {
			m_Tree.addLapError(getPrivacy());
		}
		
		//m_Tree.printTree("");

	}

	/**
	 * Computes class distribution of an instance using the tree.
	 * 
	 * @param instance
	 *            the instance to compute the distribution for
	 * @return the computed class probabilities
	 * @throws Exception
	 *             if computation fails
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		return m_Tree.distributionForInstance(instance);

	}
	
	public void setSeed(int seed) {
		this.seed = seed;
		random = new Random(seed);
	}

	public int getSeed() {
		return this.seed;
	}
	
	public double getPrivacy() {
		return m_Privacy;
	}

	public void setPrivacy(double m_Privacy) {
		this.m_Privacy = m_Privacy;
	}

	public int getMaxDepth() {
		return m_MaxDepth;
	}

	public void setMaxDepth(int m_MaxDepth) {
		this.m_MaxDepth = m_MaxDepth;
	}

	protected class TreeDp implements Serializable {

		private static final long serialVersionUID = 1L;

		/** The subtrees appended to this tree. */
		protected TreeDp[] m_Successors;

		/** The attribute to split on. */
		protected C45Attribute m_Attribute = null;

		/** The split point. */
		protected double m_SplitPoint = Double.NaN;

		/** Class probabilities from the training data. */
		protected double[] m_ClassDistribution = null;

		/**
		 * Recursively generates a tree.
		 * 
		 * @param attrWindow
		 *            the list of candidate attributes 
		 * @param random
		 *            random number generator for choosing random attributes
		 * @param depth
		 *            the current depth
		 * @throws Exception
		 *             if generation fails
		 */
		protected void buildTree(
				List<C45Attribute> attrWindow, int depth)
				throws Exception {

//			try{
			
			// Make a leaf
			if ( depth <= 0) {
				m_Attribute = null;
				m_ClassDistribution = new double[m_Info.numClasses()];
				return;
			}

			int attrIndex = random.nextInt(attrWindow.size());
			m_Attribute = attrWindow.get(attrIndex);

			attrWindow.remove(attrIndex);
			
			// for nominal attribute
			if (m_Attribute.WekaAttribute().isNominal()) {
				int numAttrValues = m_Attribute.numValues();
				m_Successors = new TreeDp[numAttrValues];
				for (int i = 0; i < numAttrValues; i++) {
					m_Successors[i] = new TreeDp();
					m_Successors[i].buildTree( attrWindow,
							 depth - 1);
				}
			}
			// for numeric attribute
			else {
				m_SplitPoint = m_Attribute.lowerBound()
						+ (m_Attribute.upperBound() - m_Attribute.lowerBound())
						* random.nextDouble();
				m_Successors = new TreeDp[2];
				for (int i = 0; i < m_Successors.length; i++) {
					m_Successors[i] = new TreeDp();
					m_Successors[i].buildTree( attrWindow, depth - 1);
				}
			}
			attrWindow.add(m_Attribute);
			
//			}catch( OutOfMemoryError e){
//				Runtime runtime = Runtime.getRuntime();
//				long memory = runtime.totalMemory() - runtime.freeMemory();
//				System.out.println("Total memory is megabytes: "+runtime.totalMemory()/(1024*1024));
//				System.out.println("Used memory is megabytes: "+ memory/(1024*1024));				
//			}
		}

		protected void trainInstance(Instance instance) {

			// Not a leaf
			if (m_Attribute != null) {

				// For nominal attributes
				if (m_Attribute.WekaAttribute().isNominal()) {				
					m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].trainInstance(instance);

				// For numeric attributes
				} else {	
					if (instance.value(m_Attribute.WekaAttribute()) < m_SplitPoint) {
						m_Successors[0].trainInstance(instance);
					} else {
						m_Successors[1].trainInstance(instance);
					}
				}
			}

			// Leaf
			if (m_Attribute == null) {
				if (m_ClassDistribution == null) {
					m_ClassDistribution = new double[m_Info.numClasses()];
				}

				m_ClassDistribution[(int) instance.classValue()]++;
			}
		}

		protected void trainInstances(Instances data) {
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				trainInstance(inst);
			}
		}

		protected void fillEmptyLeaf() {
			
			try{
			// leaf
			if (m_Attribute == null) {
				if (m_ClassDistribution == null) {
					m_ClassDistribution = new double[m_Info.numClasses()];
					m_ClassDistribution[random.nextInt(m_Info.numClasses())]++;
				}
				return;
			
			// inner node
			} else {
				for (int i = 0; i < m_Successors.length; i++) {
					m_Successors[i].fillEmptyLeaf();
				}
			}
			
			}catch(OutOfMemoryError e){
				Runtime runtime = Runtime.getRuntime();
				long memory = runtime.totalMemory() - runtime.freeMemory();
				System.out.println("Total memory is megabytes: "+runtime.totalMemory()/(1024*1024));
				System.out.println("Used memory is megabytes: "+ memory/(1024*1024));
			}
		}

		public String printDistribution(){
			
			String r= "[";
			for(int i=0;i<m_ClassDistribution.length;i++){
				r+=new DecimalFormat("#0.0").format(m_ClassDistribution[i])+", ";
			}
			
			r+="]";
			return r;
		}
		
		public String printTree(String pad){
			
			StringBuilder builder = new StringBuilder();
			
			if(m_Attribute == null){
				String newpad = pad + "-";
				//builder.append(newpad + printDistribution()+"\n");
				
				System.out.println(newpad + printDistribution());
			}
			
			// not leaf
			else{
				
				//builder.append(pad + m_Attribute.WekaAttribute().name()+"\n");
				
				System.out.println(pad + m_Attribute.WekaAttribute().name());
				
				String newpad = pad + " | ";
				for(int i=0; i<m_Successors.length;i++){
					builder.append(m_Successors[i].printTree(newpad));
				}
			}
			return builder.toString();
		}
		
		/**
		 * Sample a number from Laplace distribution with location parameter 0
		 * and with scale parameter beta.
		 * 
		 * @param beta
		 *            scale parameter for laplace distribution
		 * @return a value sampled from Laplace(0, beta)
		 */
		private double laplace(double beta) {
			//double beta = bigBeta.doubleValue();
			double uniform = random.nextDouble() - 0.5;
			return 0.0
					- beta
					* ((uniform > 0) ? -Math.log(1. - 2 * uniform) : Math.log(1. + 2 * uniform));
		}

		protected void addLapError(double budget) {
			// non leaf node
			if (m_Attribute != null) {
				for (int i = 0; i < m_Successors.length; i++) {
					m_Successors[i].addLapError(budget);
				}
			}
			// leaf node
			else {
				for (int i = 0; i < m_ClassDistribution.length; i++) {
					m_ClassDistribution[i] += laplace(1.0/budget);
						//	BigDecimal.ONE.divide(BigDecimal.valueOf(budget)));
					
					if( m_ClassDistribution[i] < 0){
						m_ClassDistribution[i] = 0.0;
					}
				}

				// if any of them is less than 0, sum of them equals to 0;
				if(Utils.sum(m_ClassDistribution) == 0.0){
					m_ClassDistribution[random.nextInt(m_ClassDistribution.length)] ++;
				}
			}
		}

		/**
		 * Computes class distribution of an instance using the decision tree.
		 * 
		 * @param instance
		 *            the instance to compute the distribution for
		 * @return the computed class distribution
		 * @throws Exception
		 *             if computation fails
		 */
		public double[] distributionForInstance(Instance instance)
				throws Exception {

			double[] returnedDist = null;

			// non-Leaf
			if (m_Attribute != null) {

				if (instance.isMissing(m_Attribute.WekaAttribute())) {
					System.err.println("Testing phase: can't deal with missing attribute");
					System.exit(1);

					// For nominal attributes
				} else if (m_Attribute.WekaAttribute().isNominal()) {
					returnedDist = m_Successors[(int) instance
							.value(m_Attribute.WekaAttribute())]
							.distributionForInstance(instance);
				} else {

					// For numeric attributes
					if (instance.value(m_Attribute.WekaAttribute()) < m_SplitPoint) {
						returnedDist = m_Successors[0]
								.distributionForInstance(instance);
					} else {
						returnedDist = m_Successors[1]
								.distributionForInstance(instance);
					}
				}
			}

			// Leaf
			if (m_Attribute == null) {

				// Is node empty?
				if (m_ClassDistribution == null) {
					System.err
							.println("Testing Parse: class distribution of leaf node shouldn't be null");
					System.exit(1);
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

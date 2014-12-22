package diffpvc;

import diffpvc.PrivacyAgents.CommonBigDecimal;
import diffpvc.PrivacyAgents.PrivacyAgentBudget;
import diffpvc.PrivacyAgents.PrivacyAgentPartition;
import diffpvc.Scorer.MaxScorer;
import diffpvc.Scorer.RandomScorer;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.Sourcable;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.math.BigDecimal;

import weka.classifiers.trees.j48.Stats;
import weka.core.*;

/**
 * Created by IntelliJ IDEA.
 * User: Arik Friedman
 * Date: 27/04/2009
 * DiffPrivacyC45 implements C4.5 while conforming to the privacy constraints of differential privacy.
 */

/**
 * <!-- globalinfo-start --> Class for constructing an unpruned decision tree
 * based on the C4.5 algorithm. Can deal with both nominal and numeric
 * attributes. No missing values allowed. For more information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning.
 * 1(1):81-106.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Arik Friedman
 * @version $Revision: 1.0 $
 */
public class C45 extends RandomizableClassifier{

	private static final long serialVersionUID = 1L;

	/** The node's successors. */
	protected C45[] m_Successors;

	/** Attribute used for splitting. */
	protected C45Attribute m_Attribute;

	/** Split point used for splitting with a numeric attribute */
	protected double m_SplitPoint;

	/** Class value if node is leaf. */
	protected double m_ClassValue;

	/** Class distribution if node is leaf. */
	protected double[] m_Distribution;

	/**
	 * instance (approximate) counts - the (noisy) number of instances in the
	 * subtree/leaf per class value
	 */
	protected double[] m_Counts;

	/**
	 * instance (approximate) counts and counts per class value, calibrated
	 * according to the (more accurate) number of instances computed in higher
	 * level nodes
	 */
	//protected double m_fixedNumInstances;
	//protected double[] m_fixedCounts;

	/** Class attribute of dataset. */
	protected Attribute m_ClassAttribute;

	/** The value of epsilon for each differential privacy operation */
	// protected BigDecimal m_PrivacyBudgetPerAction;

	/**
	 * Maximal number of instances allowed for the data set (used to determine
	 * sensitivity for info gain
	 */
	protected int m_maxNumInstances;

	/**
	 * Determine whether the checks for number of instances should be skipped
	 * when inducing the tree (depth of tree will be fixed)
	 */
	protected boolean m_skipNumInstancesChecks = false;

	/** The (approximate) number of instances in the current node */
	// protected double m_approxNumInstances;
	protected double m_numInstances;
	
	protected Random m_Random;

	/** Confidence level */
	protected float m_CF = DEFAULT_CONFIDENCE_FACTOR;

	/**
	 * Maximal allowed depth for the induced decision tree
	 */
	protected int m_MaxDepth = DEFAULT_MAX_DEPTH;

	/** Denote whether to post-prune the tree */
	protected boolean m_Unpruned = false;

	protected String m_numericAttributesFile = DEFAULT_NUMERIC_ATTRIBUTES_FILE;

	final private static int DEFAULT_MAX_DEPTH = 5;
	final public static String MAX_DEPTH_OPTION = "d";
	final static public String C45_SCORER_OPTION = "O";
	final static public String MAX_NUM_INSTANCES_OPTION = "I";
	final static public int DEFAULT_MAX_NUM_INSTANCES = 0;
	final static public String NUMERIC_ATTRIBUTES_FILE_OPTION = "F";
	final static public String DEFAULT_NUMERIC_ATTRIBUTES_FILE = "c:\\numericAtts.txt";
	final static public String CONFIDENCE_FACTOR_OPTION = "C";
	final static public String UNPRUNED_TREE_OPTION = "U";
	final static public float DEFAULT_CONFIDENCE_FACTOR = 0.25f;
	final static public String SKIP_NUM_INSTANCES_CHECK_OPTION = "S";

	/** C45 scorer to use (default: max scorer) */
	protected AttributeScoreAlgorithm m_Scorer = new MaxScorer();
	final private static String DEFAULT_SCORER_CLASS = "technion.dsl.datamining.scorer.MaxScorer";

	protected C45 CreateC45() {
		C45 newTree = new C45();
		newTree.m_Debug = m_Debug;
		newTree.m_Scorer = m_Scorer;
		newTree.m_MaxDepth = m_MaxDepth - 1; // not really used (other than
												// being received as a parameter
												// from the UI,
		newTree.m_CF = m_CF;
		newTree.m_Unpruned = m_Unpruned;
		newTree.m_ClassAttribute = m_ClassAttribute;
		newTree.m_skipNumInstancesChecks = m_skipNumInstancesChecks;
		return newTree;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
		// XXX numeric data?
		result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
		

		// class
		result.enable(Capabilities.Capability.NOMINAL_CLASS);
		result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Builds C45 decision tree classifier.
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// PrivacyAgent privacyAgent = new PrivacyAgentBudget(m_Epsilon);

		// remove instances with missing class
		data.deleteWithMissingClass();

		m_Random = new Random(getSeed());


		if (m_Debug)
			System.out.println("Total number of instances: "
					+ data.numInstances());
		
		if (m_maxNumInstances == 0)
			m_maxNumInstances = (int) Math.pow(2,
					Math.ceil(Utils.log2(data.numInstances())));
		m_Scorer.InitializeMaxNumInstances(m_maxNumInstances);
		
		if (m_Debug)
			System.out.println("MaxNumInstances: " + m_maxNumInstances);

		
		// Process candidate attributes
		List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
		int numNumericAtts = 0;
		@SuppressWarnings("rawtypes")
		Enumeration attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			if (att.isNumeric()) {
				candidateAttributes.add(new C45Attribute(att,m_numericAttributesFile));
				numNumericAtts++;
			} else
				candidateAttributes.add(new C45Attribute(att));
		}

		if (m_Debug) {
			System.out.println("Number of numeric attrbiutes is "+ numNumericAtts);
			System.out.println("Depth is " + m_MaxDepth);
		}
		
		// Make Trees
		makeTree(data, candidateAttributes, m_MaxDepth, m_Random);

		// Pruning
		if (!m_skipNumInstancesChecks && !m_Unpruned) {
			if (m_Debug) {
				System.out.println("\n\nTree before pruning:");
				System.out.println(toString());
			}			
			
			fixClassCountsBottomUp();
			prune();

			if (m_Debug) {
				System.out.println("\n\nTree after pruning:");
				System.out.println(toString());
			}
		}

	}
	
	/**
	 * Given noisy counts, turn them into a distribution, and make sure it
	 * "behaves", i.e., there are no negative probabilities, and that the
	 * elements sum up to 1 The fixed distribution must have the same dominant
	 * class value as in the counts
	 * 
	 * @param counts
	 *            a (possibly noisy) count of instances per class value
	 * @return a fixed version of the distribution
	 */
	protected double[] turnToDistribution(double counts[]) {
		
		double[] distribution = new double[counts.length];
		
		// in the new distribution, make sure there are no negative values
		double sum = 0;
		for (int i = 0; i < counts.length; i++) {
			distribution[i] = (counts[i] < 0) ? 0 : counts[i]; 
			sum += distribution[i];
		}

		// ensure that the distribution elements some up to 1.0
		if (sum == 0){
			for (int i = 0; i < distribution.length; i++)
				distribution[i] = 0.0;
		}
		else
			for (int i = 0; i < distribution.length; i++)
				distribution[i] /= sum;

		return distribution;
	}
	
    public double[] getDistribution(Instances data)
    {
           double[] distribution = new double[data.numClasses()];
           
           @SuppressWarnings("rawtypes")
           Enumeration instEnum = data.enumerateInstances();
           while (instEnum.hasMoreElements()) {
                  Instance inst = (Instance) instEnum.nextElement();
                  distribution[(int) inst.classValue()]++;
           }
           return distribution;
    }

	/**
	 * Turn a node into a leaf. This method chooses a class value by taking the
	 * value that maximizes the noisy count
	 *
	 * @param data
	 *            the data in the node
	 */
	protected void turnToLeaf(Instances data) {
		// calculate class distribution
		m_Counts     = getDistribution(data);
		m_Distribution = turnToDistribution(m_Counts);
		m_ClassValue = Utils.maxIndex(m_Distribution);
		m_Attribute = null;
		m_Successors = null;
		return;
	}

	/**
	 * Turn a subtree into a leaf.
	 *
	 * @param counts
	 *            the distribution of class values in the subtree
	 */
	protected void turnSubtreeToLeaf(double[] counts) {
		m_Counts = counts;
		m_Distribution = turnToDistribution(m_Counts);
		m_ClassValue = Utils.maxIndex(m_Distribution);
		m_Attribute = null;
		m_Successors = null;
		return;
	}

	protected double chooseNumericSplitPoint(Instances data,
			C45Attribute att, AttributeScoreAlgorithm scorer) {

		if (!att.isNumeric())
			throw new IllegalArgumentException(
					"Numeric split point can only be chosen for a numeric attribute.");

		// Short cut: don't go through the exponential mechanism just for random
		// selection
		if (scorer.getClass().equals(RandomScorer.class))
			return (att.lowerBound() + m_Random.nextDouble()
					* (att.upperBound() - att.lowerBound()));

		double[][] splitPoints = C45Attribute.GetSplitPoints(data, att, scorer);
		double[] scores = new double[splitPoints.length];
		double[] weights = new double[splitPoints.length];
		// Calculate the overall score for each interval
		// (= <interval size> * <score in each point>)
		for (int i = 0; i < splitPoints.length; i++) {
			scores[i]  = splitPoints[i][2];
			weights[i] = splitPoints[i][1] - splitPoints[i][0];

		}

		int maxIndex = Utils.maxIndex(scores);
		//double maxScore = scores[Utils.maxIndex(scores)];

		// Uniformly pick a point within the interval
		double splitPoint 
			= splitPoints[maxIndex][0] 
			+ (splitPoints[maxIndex][1] - splitPoints[maxIndex][0]) * m_Random.nextDouble();
		
		return splitPoint;

	}

	protected C45Attribute chooseAttribute(Instances data, AttributeScoreAlgorithm scorer,
			List<C45Attribute> attList) {

		int maxIndex;

		// shortcut: don't go through the exponential mechanism just for a
		// random selection
		if (scorer.getClass().equals(RandomScorer.class))
			maxIndex = m_Random.nextInt(attList.size());
		
		else {
			if (m_Debug)
				System.out.println("Choose Attribute, going to evaluate "+ attList.size() + " attributes");

			double[] scores = new double[attList.size()];
			
			int i = 0;
			for (C45Attribute att : attList) {
				scores[i] = scorer.Score(data, att);
				
				if (m_Debug)
					System.out.println("\tAttribute "+ i
							+ " ("
							+ att.WekaAttribute().name()
							+ ((att.isNumeric()) ? ("(split point "+ att.getSplitPoint() + ")") : "")
							+ ") Score: " + scores[i]);
				
				i++;
			}

			//index = drawFromScores(scores, scorer.GetSensitivity(), epsilon);
			maxIndex = Utils.maxIndex(scores);
		}

		if (m_Debug)
			System.out.println("Attribute " + maxIndex + " was picked");

		return attList.get(maxIndex);
	}

	protected Instances[] partition(Instances data, C45Attribute att){
		
		Instances[] splitData;
		
		if (att.isNumeric()){

			splitData = new Instances[2];
            splitData[0] = new Instances(data,data.numInstances());
            splitData[1] = new Instances(data,data.numInstances());

            double splitPoint=att.getSplitPoint();
            
            @SuppressWarnings("rawtypes")
			Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                   Instance inst = (Instance) instEnum.nextElement();
                   if (inst.value(att.WekaAttribute()) < splitPoint)
                          splitData[0].add(inst);
                   else
                          splitData[1].add(inst);                     
            }			
		}
		else{
            splitData = new Instances[att.numValues()];
            for (int j = 0; j <splitData.length; j++) {
                   splitData[j] = new Instances(data,data.numInstances());                                        
            }

            @SuppressWarnings("rawtypes")
			Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                   Instance inst = (Instance) instEnum.nextElement();
                   splitData[(int) inst.value(att.WekaAttribute())].add(inst);
            }
           
		}
		
		for (Instances aSplitData : splitData) {
             aSplitData.compactify();
		}
		
		return splitData;
	}
	
	/**
	 * Method for building a C45 tree.
	 *
	 * @param data
	 *            the training data
	 * @param candidateAttributes
	 *            the attributes to check for splitting the tree
	 * @param depth
	 *            the maximal allowed depth for the induced sub-tree
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	protected void makeTree(Instances data,
			List<C45Attribute> candidateAttributes, int depth, Random random) throws Exception {
		
		m_ClassAttribute = data.classAttribute();
		m_numInstances   = data.numInstances();
		m_Random 		 = random;

		// 1. Make leaf
		if (depth <= 0 || candidateAttributes.size() == 0 )
		{
			turnToLeaf(data); 
			return;
		}

		// If we got here, then we split the node

		// 2. for Numeric Attribute, determine split point
		for (C45Attribute att : candidateAttributes) 
		{
			if (att.isNumeric()) {
				double splitPoint = chooseNumericSplitPoint(data, att, m_Scorer);
				att.setSplitPoint(splitPoint);
			}
		}

		// 3. Choose split attribute
		m_Attribute = chooseAttribute(data, m_Scorer, candidateAttributes);
		
		if (m_Debug)
			System.out.println("Splitting with attribute "+ m_Attribute.WekaAttribute().name());

		
		// 4. Split instances by split attribute
		Instances[] splitData = partition(data, m_Attribute);
		
		candidateAttributes.remove(m_Attribute); // attribute will not be available in sub-trees										 // unless it is numeric
		if (m_Attribute.isNumeric()) {
			m_Successors = new C45[2];// numeric attributes use binary splits
			m_SplitPoint = m_Attribute.getSplitPoint();

			// Make left sub tree
			C45Attribute left = new C45Attribute(m_Attribute, m_Attribute.lowerBound(), m_SplitPoint);
			candidateAttributes.add(left);
			m_Successors[0] = CreateC45();
			m_Successors[0].makeTree(splitData[0], candidateAttributes, depth - 1, m_Random);
			candidateAttributes.remove(left);

			// Make right sub tree
			C45Attribute right = new C45Attribute(m_Attribute, m_SplitPoint, m_Attribute.upperBound());
			candidateAttributes.add(right);
			m_Successors[1] = CreateC45();
			m_Successors[1].makeTree(splitData[1], candidateAttributes, depth - 1, m_Random);
			candidateAttributes.remove(right);
			
		} else {
			m_Successors = new C45[m_Attribute.numValues()];
			for (int j = 0; j < m_Successors.length; j++) {
				m_Successors[j] = CreateC45();
				m_Successors[j].makeTree(splitData[j], candidateAttributes, depth - 1, m_Random);
			}
		}
		candidateAttributes.add(m_Attribute);// make sure that the attribute
												// will be available for next
												// successors
	}

	/**
	 * For each inner node, store a sum of class distributions in its subtrees. 
	 * The class distribution will be used for error based pruning evaluation 
	 */
	protected void fixClassCountsBottomUp() {
		if (m_Attribute == null) {
			return;
		}

		for (C45 node : m_Successors) {
			node.fixClassCountsBottomUp();
			
			if (m_Counts == null)
				m_Counts = new double[node.m_Counts.length];
			
			for (int i = 0; i < m_Counts.length; i++)
				m_Counts[i] += node.m_Counts[i];
		}
	}

	/**
	 * Prunes a tree using C4.5's pruning procedure.
	 */
	protected void prune() {
		// leaf, return
		if (m_Attribute == null) 
			return;

		// Prune all sub-trees
		for (C45 node : m_Successors)
			node.prune();

		
		double[] counts = m_Counts;
		
		// Compute error if this Tree would be leaf
		double errorsLeaf = getLeafErrors();

		// Compute error for the whole sub-tree
		double errorsTree = getSubtreeErrors();
		
		if (m_Debug)
			System.out.println("prune(): errors in leaf: " + errorsLeaf + ", errors in subtree: " + errorsTree);
		
		// Decide if leaf is best choice.
		if (Utils.smOrEq(errorsLeaf, errorsTree + 0.1)) {
			
			turnSubtreeToLeaf(counts);
			if (m_Debug)
				System.out.println("Pruning node ");
			
		} else 
			if (m_Debug)
			System.out.println("Node not pruned.");
	}


	/**
	 * Computes estimated errors for this node, when it is considered as a leaf.
	 *
	 * @param counts
	 *            the distribution of class values within the subtree
	 * @return estimated error of the subtree if it would be turned into a leaf
	 */
	private double getLeafErrors() {
		
		if (m_numInstances <= 0)
			return 0;
		
		int index = Utils.maxIndex(m_Counts);
		double error = m_numInstances - m_Counts[index];
		double numErrors = (error < 0) ? 0 : error;

		return numErrors + Stats.addErrs(m_numInstances, numErrors, m_CF);
	}

	/**
	 * Computes estimated errors for tree.
	 * 
	 * @return estimated errors for tree
	 */
	private double getSubtreeErrors() {
		if (m_Attribute == null)
			return getLeafErrors();

		if (m_numInstances <= 0)
			return 0;

		double errors = 0;
		for (C45 node : m_Successors)
			errors += node.getSubtreeErrors();

		return errors;
	}

	/**
	 * Classifies a given test instance using the decision tree.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return the classification
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double classifyInstance(Instance instance)
			throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException(
					"C4.5: no missing values, " + "please.");
		}
		if (m_Attribute == null) {
			return m_ClassValue;
		} else {
			// treat non-numeric attribute
			if (!m_Attribute.isNumeric())
				return m_Successors[(int) instance.value(m_Attribute
						.WekaAttribute())].classifyInstance(instance);

			// treat numeric attribute
			if (instance.value(m_Attribute.WekaAttribute()) < m_SplitPoint)
				return m_Successors[0].classifyInstance(instance);
			else
				return m_Successors[1].classifyInstance(instance);
		}
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 *
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double[] distributionForInstance(Instance instance)
			throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException(
					"C4.5: no missing values, " + "please.");
		}
		if (m_Attribute == null) {
			return m_Distribution;
		} else {
			// treat non-numeric attribute
			if (!m_Attribute.isNumeric())
				return m_Successors[(int) instance.value(m_Attribute
						.WekaAttribute())].distributionForInstance(instance);

			// treat numeric attribute
			if (instance.value(m_Attribute.WekaAttribute()) < m_SplitPoint)
				return m_Successors[0].distributionForInstance(instance);
			else
				return m_Successors[1].distributionForInstance(instance);
		}
	}

	/**
	 * Prints the decision tree using the private toString method from below.
	 * Function altered with respect to original in C4.5 - epsilon parameter is
	 * output as well
	 *
	 * @return a textual description of the classifier
	 */
	public String toString() {

		if ((m_Distribution == null) && (m_Successors == null)) {
			return //m_Epsilon.toString()+ 
					"-Differential Privacy C4.5: No model built yet.\n\n";
		}
		return //m_Epsilon.toString() + 
				"-Differential Privacy C4.5\n\n" + "\n\n"
				+ toString(0);
	}

	/**
	 * Outputs a tree at a certain level.
	 *
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	protected String toString(int level) {

		StringBuffer text = new StringBuffer();

		if (m_Attribute == null) {
			if (Instance.isMissingValue(m_ClassValue)) {
				text.append(": null");
			} else {
				text.append("  [" + m_numInstances + "]");
				text.append(": ").append(
						m_ClassAttribute.value((int) m_ClassValue));
				text.append("   Counts  " + distributionToString(m_Counts)
						+ "   Distribution "
						+ distributionToString(m_Distribution));
			}
		} else {
			text.append("  [" + m_numInstances + "]");
			for (int j = 0; j < m_Successors.length; j++) {
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				if (m_Attribute.isNumeric())
					text.append(m_Attribute.WekaAttribute().name())
							.append(j == 0 ? " < " : " >= ")
							.append(m_SplitPoint);
				else
					text.append(m_Attribute.WekaAttribute().name())
							.append(" = ")
							.append(m_Attribute.WekaAttribute().value(j));
				text.append(m_Successors[j].toString(level + 1));
			}
		}
		return text.toString();
	}

	private String distributionToString(double[] distribution) {
		StringBuffer text = new StringBuffer();
		text.append("[");
		for (double d : distribution)
			text.append(d + "; ");
		text.append("]");
		return text.toString();
	}

	/**
	 * Adds this tree recursively to the buffer.
	 *
	 * @param id
	 *            the unqiue id for the method
	 * @param buffer
	 *            the buffer to add the source code to
	 * @return the last ID being used
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected int toSource(int id, StringBuffer buffer) throws Exception {
		int result;
		int i;
		int newID;
		StringBuffer[] subBuffers;

		buffer.append("\n");
		buffer.append("  protected static double node").append(id)
				.append("(Object[] i) {\n");

		// leaf?
		if (m_Attribute == null) {
			result = id;
			if (Double.isNaN(m_ClassValue))
				buffer.append("    return Double.NaN;");
			else
				buffer.append("    return ").append(m_ClassValue).append(";");
			if (m_ClassAttribute != null)
				buffer.append(" // ").append(
						m_ClassAttribute.value((int) m_ClassValue));
			buffer.append("\n");
			buffer.append("  }\n");
		} else {
			buffer.append("    // ").append(m_Attribute.WekaAttribute().name())
					.append("\n");

			// subtree calls
			subBuffers = new StringBuffer[m_Successors.length];
			newID = id;
			for (i = 0; i < m_Successors.length; i++) {
				newID++;

				buffer.append("    ");
				if (i > 0)
					buffer.append("else ");
				if (m_Attribute.isNumeric())
					buffer.append("if (((Double) i[")
							.append(m_Attribute.WekaAttribute().index())
							.append("]).doubleValue()")
							.append(i == 0 ? " < " : " >= ")
							.append(m_SplitPoint).append(")\n");
				else
					buffer.append("if (((String) i[")
							.append(m_Attribute.WekaAttribute().index())
							.append("]).equals(\"")
							.append(m_Attribute.WekaAttribute().value(i))
							.append("\"))\n");
				buffer.append("      return node");
				buffer.append(newID);
				buffer.append("(i);\n");

				subBuffers[i] = new StringBuffer();
				newID = m_Successors[i].toSource(newID, subBuffers[i]);
			}
			buffer.append("    else\n");
			buffer.append(
					"      throw new IllegalArgumentException(\"Value '\" + i[")
					.append(m_Attribute.WekaAttribute().index())
					.append("] + \"' is not allowed!\");\n");
			buffer.append("  }\n");

			// output subtree code
			for (i = 0; i < m_Attribute.numValues(); i++) {
				buffer.append(subBuffers[i].toString());
			}
			// noinspection UnusedAssignment
			subBuffers = null;

			result = newID;
		}

		return result;
	}

	/**
	 * Returns a string that describes the classifier as source. The classifier
	 * will be contained in a class with the given name (there may be auxiliary
	 * classes), and will contain a method with the signature:
	 * 
	 * <pre>
	 * <code>
	 * public static double classify(Object[] i);
	 * </code>
	 * </pre>
	 * 
	 * where the array <code>i</code> contains elements that are either Double,
	 * String, with missing values represented as null. The generated code is
	 * public domain and comes with no warranty. <br/>
	 * Note: works only if class attribute is the last attribute in the dataset.
	 *
	 * @param className
	 *            the name that should be given to the source class.
	 * @return the object source described by a string
	 * @throws Exception
	 *             if the souce can't be computed
	 */
	public String toSource(String className) throws Exception {
		StringBuffer result;
		int id;

		result = new StringBuffer();

		result.append("class ").append(className).append(" {\n");
		result.append("  public static double classify(Object[] i) {\n");
		id = 0;
		result.append("    return node").append(id).append("(i);\n");
		result.append("  }\n");
		toSource(id, result);
		result.append("}\n");

		return result.toString();
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.0 $");
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maxDepthTipText() {
		return "The maximal depth allowed for the induced decision tree.";
	}

	/**
	 * Set the privacy policy
	 *
	 * @param depth
	 *            the maximal depth allowed for the induced decision tree
	 */
	public void setMaxDepth(int depth) {
		m_MaxDepth = depth;
	}

	/**
	 * Get the used privacy policy
	 *
	 * @return the maximal depth allowed for the induced decision tree
	 */
	public int getMaxDepth() {
		return m_MaxDepth;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String scorerTipText() {
		return "The scorer to use to score attributes when spliting nodes for decision tree induction.";
	}

	/**
	 * Set the attribute scorer
	 *
	 * @param newScorer
	 *            the new scorer to use.
	 */
	public void setScorer(AttributeScoreAlgorithm newScorer) {
		m_Scorer = newScorer;
	}

	/**
	 * Get the used scorer
	 *
	 * @return the used attribute scorer
	 */
	public AttributeScoreAlgorithm getScorer() {
		return m_Scorer;
	}

	public void setMaxNumInstances(int num) {
		m_maxNumInstances = num;
	}

	public int getMaxNumInstances() {
		return m_maxNumInstances;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maxNumInstancesTipText() {
		return "The maximal number of instances that the training set can get (0 to round up to nearest log2 from above).";
	}

	public void setNumericAttributesFile(String file) {
		m_numericAttributesFile = file;
	}

	public String getNumericAttributesFile() {
		if (m_numericAttributesFile == null
				|| m_numericAttributesFile.length() == 0)
			return DEFAULT_NUMERIC_ATTRIBUTES_FILE;

		return m_numericAttributesFile;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numericAttributesFileTipText() {
		return "The path and name for a text file containing upper and lower bounds of numeric attributes";
	}

	/**
	 * Get the value of unpruned.
	 *
	 * @return Value of unpruned.
	 */
	public boolean getUnpruned() {

		return m_Unpruned;
	}

	/**
	 * Set the value of unpruned. Turns reduced-error pruning off if set.
	 * 
	 * @param v
	 *            Value to assign to unpruned.
	 */
	public void setUnpruned(boolean v) {
		m_Unpruned = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String unprunedTipText() {
		return "Whether to prune the tree";
	}

	/**
	 * Get the value of CF.
	 *
	 * @return Value of CF.
	 */
	public float getConfidenceFactor() {

		return m_CF;
	}

	/**
	 * Set the value of CF.
	 *
	 * @param v
	 *            Value to assign to CF.
	 */
	public void setConfidenceFactor(float v) {

		m_CF = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String confidenceFactorTipText() {
		return "The confidence factor used for pruning (smaller values incur "
				+ "more pruning).";
	}

	public void setSkipNumInstancesChecks(boolean setDepth) {
		m_skipNumInstancesChecks = setDepth;
	}

	public boolean getSkipNumInstancesChecks() {
		return m_skipNumInstancesChecks;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String skipNumInstancesChecksTipText() {
		return "Determines whether the number of instances checks for tree depth should be skipped when inducing the tree. Setting this to true disables pruning.";
	}

	/**
	 * Returns an enumeration of all the available options..
	 *
	 * @return an enumeration of all available options.
	 */
	@SuppressWarnings("rawtypes")
	public Enumeration listOptions() {
		Vector<Option> newVector = new Vector<Option>(7);
		newVector.addElement(new Option(
				"\tMaximal allowed depth for the induced decision tree (default: "
						+ DEFAULT_MAX_DEPTH + ").", MAX_DEPTH_OPTION, 1, "-"
						+ MAX_DEPTH_OPTION));

		newVector.addElement(new Option(
				"\tFull class name of attribute scorer.\n" + "\t(default: "
						+ DEFAULT_SCORER_CLASS + ")", C45_SCORER_OPTION, 1, "-"
						+ C45_SCORER_OPTION));

		newVector.addElement(new Option(
				"\tPath and name of file with upper and lower bounds for numeric attributes.\n"
						+ "\t(default: " + DEFAULT_NUMERIC_ATTRIBUTES_FILE
						+ ")", NUMERIC_ATTRIBUTES_FILE_OPTION, 1, "-"
						+ NUMERIC_ATTRIBUTES_FILE_OPTION));

		newVector
				.addElement(new Option(
						"\tMaximal number of allowed training instances, 0 will automatically round up to nearest log 2 based on given training set.\n"
								+ "\t(default: "
								+ DEFAULT_MAX_NUM_INSTANCES
								+ ")", MAX_NUM_INSTANCES_OPTION, 1, "-"
								+ MAX_NUM_INSTANCES_OPTION));

		newVector.addElement(new Option("\tWhether pruning is performed",
				UNPRUNED_TREE_OPTION, 0, "-" + UNPRUNED_TREE_OPTION));

		newVector.addElement(new Option(
				"\tSet confidence threshold for pruning.\n"
						+ "\t(default 0.25)", CONFIDENCE_FACTOR_OPTION, 1, "-"
						+ CONFIDENCE_FACTOR_OPTION + " <pruning confidence>"));

		newVector.addElement(new Option(
				"\tWhether number of instances checks are skipped",
				SKIP_NUM_INSTANCES_CHECK_OPTION, 0, "-"
						+ SKIP_NUM_INSTANCES_CHECK_OPTION));

		return newVector.elements();
	}

	/**
	 * Sets the OptionHandler's options using the given list. All options will
	 * be set (or reset) during this call (i.e. incremental setting of options
	 * is not possible).
	 *
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	// @ requires options != null;
	// @ requires \nonnullelements(options);
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		String paramString = Utils.getOption(MAX_DEPTH_OPTION, options);
		if (paramString.length() != 0)
			setMaxDepth(Integer.parseInt(paramString));
		else
			setMaxDepth(DEFAULT_MAX_DEPTH);

		String scorerString = Utils.getOption(C45_SCORER_OPTION, options);

		if (scorerString.length() > 0) {
			setScorer((AttributeScoreAlgorithm) Utils.forName(
					AttributeScoreAlgorithm.class, scorerString, null));
		} else {
			setScorer((AttributeScoreAlgorithm) Utils.forName(
					AttributeScoreAlgorithm.class, DEFAULT_SCORER_CLASS, null));
		}

		String maxNumInstancesString = Utils.getOption(
				MAX_NUM_INSTANCES_OPTION, options);
		if (maxNumInstancesString.length() > 0)
			setMaxNumInstances(Integer.parseInt(maxNumInstancesString));
		else
			setMaxNumInstances(DEFAULT_MAX_NUM_INSTANCES);

		paramString = Utils.getOption(NUMERIC_ATTRIBUTES_FILE_OPTION, options);
		if (paramString.length() > 0)
			setNumericAttributesFile(paramString);
		else
			setNumericAttributesFile(DEFAULT_NUMERIC_ATTRIBUTES_FILE);

		// Pruning option
		m_Unpruned = Utils.getFlag(UNPRUNED_TREE_OPTION, options);

		m_skipNumInstancesChecks = Utils.getFlag(
				SKIP_NUM_INSTANCES_CHECK_OPTION, options);

		// Confidence factor for pruning
		paramString = Utils.getOption(CONFIDENCE_FACTOR_OPTION, options);
		if (paramString.length() != 0) {
			if (m_Unpruned)
				throw new IllegalArgumentException(
						"Doesn't make sense to change confidence for unpruned "
								+ "tree!");
			else {
				setConfidenceFactor(Float.parseFloat(paramString));
				if ((m_CF <= 0) || (m_CF >= 1))
					throw new IllegalArgumentException(
							"Confidence has to be greater than zero and smaller "
									+ "than one!");
			}
		} else
			setConfidenceFactor(DEFAULT_CONFIDENCE_FACTOR);

	}

	/**
	 * Gets the current option settings for the OptionHandler.
	 *
	 * @return the list of current option settings as an array of strings
	 */
	// @ ensures \result != null;
	// @ ensures \nonnullelements(\result);
	/* @pure@ */
	public String[] getOptions() {
		String[] superOptions = super.getOptions();

		String[] options = new String[10 + superOptions.length];
		int current = 0;

		options[current++] = "-" + MAX_DEPTH_OPTION;
		options[current++] = "" + m_MaxDepth;

		options[current++] = "-" + C45_SCORER_OPTION;
		options[current++] = getScorer().getClass().getName();

		options[current++] = "-" + NUMERIC_ATTRIBUTES_FILE_OPTION;
		options[current++] = getNumericAttributesFile();

		options[current++] = "-" + CONFIDENCE_FACTOR_OPTION;
		options[current++] = Float.toString(getConfidenceFactor());

		if (m_Unpruned)
			options[current++] = "-" + UNPRUNED_TREE_OPTION;

		if (m_skipNumInstancesChecks)
			options[current++] = "-" + SKIP_NUM_INSTANCES_CHECK_OPTION;

		for (String superOption : superOptions)
			options[current++] = superOption;
		while (current < options.length) {
			options[current++] = "";
		}

		return options;
	}

}

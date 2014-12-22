package diffpvc.RDTs;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import diffpvc.C45Attribute;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class RandomForestDp extends Classifier {

	private static final long serialVersionUID = 1L;

	protected BaggingDp m_bagger = null;

	/**
	 * Differential privacy parameter Input by user
	 */
	protected double m_Privacy = 0.0;

	protected String m_EvalMethod = "MV";

	protected int m_numTrees;

	protected int m_maxDepth;

	private int m_randomSeed;

	protected String m_attrFile;

	public void setAttrFile(String attrFile) {
		m_attrFile = attrFile;
	}

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

	/**
	 * Builds a classifier for a set of instances.
	 * 
	 * @param data
	 *            the instances to train the classifier with
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		double budgetPerTree = getPrivacy() / m_numTrees;
		//System.out.println("Budget per tree is " + budgetPerTree);

		// Candidate Attributes
		List<C45Attribute> candidateAttributes = new ArrayList<C45Attribute>();

		@SuppressWarnings("rawtypes")
		Enumeration attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			if (att.isNumeric())
				candidateAttributes.add(new C45Attribute(att, m_attrFile));
			else
				candidateAttributes.add(new C45Attribute(att));
		}

		m_bagger = new BaggingDp();
		RandomTreeDp rTree = new RandomTreeDp();

		// Max Depth of a tree
		rTree.setMaxDepth(getMaxDepth());
		rTree.setPrivacy(budgetPerTree);
		rTree.setCandidateAttributes(candidateAttributes);

		// set up the bagger and build the forest
		m_bagger.setEvalMethod(getEvalMethod());
		m_bagger.setClassifier(rTree);
		m_bagger.setNumIterations(m_numTrees); // Dp mode, the number of trees
		m_bagger.setSeed(m_randomSeed);

		m_bagger.buildClassifier(data);
	}

	public int getSeed() {
		return m_randomSeed;
	}

	public void setSeed(int m_randomSeed) {
		this.m_randomSeed = m_randomSeed;
	}

	public int getNumTrees() {
		return m_numTrees;
	}

	public void setNumTrees(int m_numTrees) {
		this.m_numTrees = m_numTrees;
	}

	public int getMaxDepth() {
		return m_maxDepth;
	}

	public void setMaxDepth(int m_maxDepth) {
		this.m_maxDepth = m_maxDepth;
	}

	/**
	 * Returns the class probability distribution for an instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return the distribution the forest generates for the instance
	 * @throws Exception
	 *             if computation fails
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return m_bagger.distributionForInstance(instance);
	}
}

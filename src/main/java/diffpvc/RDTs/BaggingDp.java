package diffpvc.RDTs;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;

public class BaggingDp extends Classifier {

	private static final long serialVersionUID = 1L;

	protected Classifier m_Classifier;

	/** Array for storing the generated base classifiers. */
	protected Classifier[] m_Classifiers;

	/** Number of basic Classifier */
	private int m_NumIterations;

	/** Evaluation Methods, includes Major Voting(MV), Threshold Averaging(TA), and Probabilistic Averaging(PA) */
	protected String evalMethod = null; // MV,TA,PA
	
	protected int seed = 1;
	
	protected Random random;
	
	/**
	 * Bagging method.
	 * 
	 * @param data
	 *            the training data to be used for generating the bagged
	 *            classifier.
	 * @throws Exception
	 *             if the classifier could not be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		// getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		if (m_Classifier == null) {
			throw new Exception("A base classifier has not been specified!");
		}
		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

		for (int j = 0; j < m_Classifiers.length; j++) {

			if (m_Classifier instanceof Randomizable) {
				((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());
			}

			// build the classifier
			m_Classifiers[j].buildClassifier(data);
		}
	}

	/**
	 * For debug
	 * @param arr
	 * @return
	 */
	@SuppressWarnings("unused")
	private static String printArray(double[] arr) {
		String out = "[";
		for (int i = 0; i < arr.length; i++) {
			out += arr[i] + ",";
		}
		out += "]";
		return out;
	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return preedicted class probability distribution
	 * @throws Exception
	 *             if distribution can't be computed successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		double[] sums = new double[instance.numClasses()], newProbs;
		
		if (evalMethod.equals("MV")) { // Major Voting

			for (int i = 0; i < m_NumIterations; i++) {
				newProbs = m_Classifiers[i].distributionForInstance(instance);
				int maxIndex = Utils.maxIndex(newProbs);
				sums[maxIndex] += 1;
			}
		} else if (evalMethod.equals("TA") || evalMethod.equals("PA")) { // Threshold
			for (int i = 0; i < m_NumIterations; i++) {
				newProbs = m_Classifiers[i]
						.distributionForInstance(instance);
				for (int j = 0; j < newProbs.length; j++)
					sums[j] += newProbs[j];
			}
		} else {
			System.err.println("Runtime error: evaluation method " + evalMethod
					+ " is invalid");
			System.exit(1);
		}

		if (Utils.eq(Utils.sum(sums), 0)) {
			// System.out.println("Debug: output sums=0 is "+printArray(sums));
			return sums;
		} else {
			// System.out.println("Debug: output sums is "+printArray(sums));
			Utils.normalize(sums);
			return sums;
		}
	}

	public void setClassifier(Classifier classifier) {
		this.m_Classifier = classifier;
	}

	public void setNumIterations(int num) {
		this.m_NumIterations = num;

	}
	public void setSeed(int seed) {
		this.seed = seed;
		this.random = new Random(seed);
	}

	public int getSeed() {
		return this.seed;
	}

	public String getEvalMethod() {
		return evalMethod;
	}

	public void setEvalMethod(String e) {
		this.evalMethod = e;
	}
}

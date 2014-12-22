package rdt;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;

public class BaggingDp extends Bagging {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected String evalMethod = null;	// MV,TA,PA
	
	public String getEvalMethod() {
		return evalMethod;
	}
	public void setEvalMethod(String e) {
		this.evalMethod = e;
	}

	/**
	   * Bagging method.
	   * 
	   * @param data the training data to be used for generating the bagged
	   *          classifier.
	   * @throws Exception if the classifier could not be built successfully
	   */
	  @Override
	  public void buildClassifier(Instances data) throws Exception {

	    // can classifier handle the data?
	    getCapabilities().testWithFail(data);

	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();

	    if (m_Classifier == null) {
	        throw new Exception("A base classifier has not been specified!");
	    }
	    m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

	    Random random = new Random(m_Seed);

	    for (int j = 0; j < m_Classifiers.length; j++) {
	    	Instances bagData = data;

	    	if (m_Classifier instanceof Randomizable) {
	    		((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());
	    	}

		    // build the classifier
		    m_Classifiers[j].buildClassifier(bagData);
	    }
	  }

	  /**
	   * Calculates the class membership probabilities for the given test instance.
	   * 
	   * @param instance the instance to be classified
	   * @return preedicted class probability distribution
	   * @throws Exception if distribution can't be computed successfully
	   */
	  @Override
	  public double[] distributionForInstance(Instance instance) throws Exception {
	
	    double[] sums = new double[instance.numClasses()], newProbs;
	
	    if(instance.classAttribute().isNumeric() == true){
	    	for (int i = 0; i < m_NumIterations; i++) {
	    		sums[0] += m_Classifiers[i].classifyInstance(instance);    		
	    	}
	    	
	    	sums[0] /= m_NumIterations;
	    	return sums;
	    }
	    
	    else{
	    	if(evalMethod.equals("MV")){ //Major Voting
	    	
		    	for (int i = 0; i < m_NumIterations; i++) {
		    		newProbs = m_Classifiers[i].distributionForInstance(instance);
		    		int maxIndex = Utils.maxIndex(newProbs);
		    		sums[maxIndex] += 1; 
		    	}
	    	}
	    	
	    	else if(evalMethod.equals("TA") || evalMethod.equals("PA") 	){ // Threshold Averaging
	    	
		    	for (int i = 0; i < m_NumIterations; i++) {
		    		newProbs = m_Classifiers[i].distributionForInstance(instance);
		    		for (int j = 0; j < newProbs.length; j++)
		    			sums[j] += newProbs[j];
		    	}
    		}
	    	
	    	if (Utils.eq(Utils.sum(sums), 0)) {
		    	return sums;
		    } else {
		    	Utils.normalize(sums);
		    	return sums;
		    }
	    }
	  }
	  
	  
	 
}

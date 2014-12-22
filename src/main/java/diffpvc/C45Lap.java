package diffpvc;

import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import diffpvc.PrivacyAgents.PrivacyAgentBudget;
import diffpvc.Scorer.GiniScorer;
import diffpvc.Scorer.RandomScorer;

public class C45Lap extends DiffPrivacyC45 {

	private static final long serialVersionUID = 1L;

    protected C45Lap CreateC45Lap()
    {
    	   C45Lap newTree = new C45Lap();
           newTree.m_Debug=m_Debug;
           newTree.m_Scorer = m_Scorer;
           newTree.m_PrivacyBudgetPerAction=m_PrivacyBudgetPerAction;
           newTree.m_MaxDepth = m_MaxDepth -1; // not really used (other than being received as a parameter from the UI,
           //newTree.m_CF=m_CF;
           newTree.m_Unpruned=m_Unpruned;
           newTree.m_ClassAttribute=m_ClassAttribute;
           newTree.m_skipNumInstancesChecks=m_skipNumInstancesChecks;
           return newTree;
    }
	
	
	protected double chooseNumericSplitPoint(PrivateInstances data,
			C45Attribute att, AttributeScoreAlgorithm scorer) {

		if (!att.isNumeric())
			throw new IllegalArgumentException(
					"Numeric split point can only be chosen for a numeric attribute.");

		// Short cut: don't go through the exponential mechanism just for random
		// selection
		if (scorer.getClass().equals(RandomScorer.class))
			return (att.lowerBound() + data.m_Random.nextDouble()
					* (att.upperBound() - att.lowerBound()));

		double[][] splitPoints = C45Attribute.GetSplitPoints(data.getInstances(), att, scorer);
		double[] scores = new double[splitPoints.length];
		double[] weights = new double[splitPoints.length];
		// Calculate the overall score for each interval
		// (= <interval size> * <score in each point>)
		for (int i = 0; i < splitPoints.length; i++) {
			scores[i] = splitPoints[i][2];
			weights[i] = splitPoints[i][1] - splitPoints[i][0];

		}

		int maxIndex = Utils.maxIndex(scores);
		// double maxScore = scores[Utils.maxIndex(scores)];

		// Uniformly pick a point within the interval
		double splitPoint = splitPoints[maxIndex][0]
				+ (splitPoints[maxIndex][1] - splitPoints[maxIndex][0])
				* data.m_Random.nextDouble();
		return splitPoint;

	}
	
	protected C45Attribute chooseAttribute(PrivateInstances data, AttributeScoreAlgorithm scorer,
			List<C45Attribute> candidateAttributes, BigDecimal m_PrivacyBudgetPerAction)
			throws PrivacyBudgetExhaustedException {

		double[] attScores = new double[candidateAttributes.size()];
		BigDecimal epsilonPerAction = m_PrivacyBudgetPerAction.divide(		
				BigDecimal.valueOf(2 * candidateAttributes.size()),
				MATH_CONTEXT);

		if (m_Debug)
			System.out.println("Checking " + candidateAttributes.size()
					+ " attributes, epsilon per action is " + epsilonPerAction);

		for (int attNum = 0; attNum < candidateAttributes.size(); attNum++) {
			PrivateInstances[] splitData = data
					.PartitionByAttribute(candidateAttributes.get(attNum));
			if (m_Debug)
				System.out.println("\tChecking attribute " + attNum);

			for (PrivateInstances attSplit : splitData) {

				double partitionSize = attSplit
						.NoisyNumInstances(epsilonPerAction); // N_{j}^{A}  // budget using 1
				if (partitionSize <= 0)
					continue;

				double[] distribution = attSplit
						.getNoisyDistribution(epsilonPerAction); // N_{j,c}^{A}  // budget using 2

				double scoreShift = 0;
				for (double classCount : distribution) {
					if (classCount <= 0)
						continue;
					if (classCount > partitionSize)
						classCount = partitionSize; 

					scoreShift += Math.pow(classCount, 2) / partitionSize;

				}
				attScores[attNum] += scoreShift;
			}
		}

		if (m_Debug)
			for (double score : attScores)
				System.out.println("Attribute score: " + score);

		return candidateAttributes.get(Utils.maxIndex(attScores));
	}
	
	 /**
     * Method for building a C45 tree.
     *
     * @param data the training data
     * @param candidateAttributes the attributes to check for splitting the tree
     * @param depth the maximal allowed depth for the induced sub-tree
     * @exception Exception
     * if decision tree can't be built successfully
     */
    protected void makeTree(PrivateInstances data,List<C45Attribute> candidateAttributes,int depth) throws Exception
    {
           int maxNumAttributeValues=maxNumValues(candidateAttributes);
           int numClassValues=data.classAttribute().numValues();
           m_ClassAttribute = data.classAttribute();

           m_approxNumInstances=0;
           if (!m_skipNumInstancesChecks)
                  m_approxNumInstances = data.NoisyNumInstances(m_PrivacyBudgetPerAction);                   // potential budget cost 1
           if (m_approxNumInstances<0)
                  m_approxNumInstances=0;
           
           // Check whether there are no more attributes available for splits,
           // whether no further splits are allowed, or whether there are enough instances to split the node,
           if (depth<=0 || candidateAttributes.size()==0 || 
        		   (!m_skipNumInstancesChecks && !EnoughInstancesToSplit(m_approxNumInstances, maxNumAttributeValues, numClassValues)))
           {
                  turnToLeaf(data);                  // privacy budget use 2a
                  return;
           }

           //  If we got here, then we split the node

           // Determine split point for numeric attributes              
           for (C45Attribute att:candidateAttributes)                        // budget cost 2 (x number of numeric attributes)
           {
                  if (att.isNumeric())
                  {
                         double splitPoint = chooseNumericSplitPoint( data, att ,m_Scorer);
                         att.setSplitPoint(splitPoint);
                  }
           }

           // Choose attribute with maximum score              
           m_Attribute = chooseAttribute(data, m_Scorer, candidateAttributes, m_PrivacyBudgetPerAction); // budget cost 3
          
           
           if (m_Debug)
                  System.out.println("Splitting with attribute " + m_Attribute.WekaAttribute().name());

           PrivateInstances[] splitData = data.PartitionByAttribute(m_Attribute);
           candidateAttributes.remove(m_Attribute); // attribute will not be available in sub-trees unless it is numeric

           if (m_Attribute.isNumeric())
           {
                  m_Successors = new DiffPrivacyC45[2];// numeric attributes use binary splits
                  m_SplitPoint=m_Attribute.getSplitPoint();                      

                  C45Attribute left=new C45Attribute(m_Attribute,m_Attribute.lowerBound(),m_SplitPoint);
                  candidateAttributes.add(left);
                  m_Successors[0] = CreateC45Lap();
                  m_Successors[0].makeTree(splitData[0],candidateAttributes,depth-1);
                  candidateAttributes.remove(left);

                  C45Attribute right=new C45Attribute(m_Attribute,m_SplitPoint,m_Attribute.upperBound());
                  candidateAttributes.add(right);
                  m_Successors[1] = CreateC45Lap();
                  m_Successors[1].makeTree(splitData[1],candidateAttributes,depth-1);
                  candidateAttributes.remove(right);
           }
           else {
                  m_Successors = new DiffPrivacyC45[m_Attribute.numValues()];
                  for (int j = 0; j < m_Successors.length; j++) {
                         m_Successors[j] = CreateC45Lap();
                         m_Successors[j].makeTree(splitData[j],candidateAttributes,depth-1);
                  }
           }
           candidateAttributes.add(m_Attribute);// make sure that the attribute will be available for next successors
    }
    
    
    /**
     * Builds C45 decision tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
           // can classifier handle the data?
           getCapabilities().testWithFail(data);
                         
           PrivacyAgent privacyAgent = new PrivacyAgentBudget(m_Epsilon);

           // remove instances with missing class
           data.deleteWithMissingClass();

           PrivateInstances privateData= new PrivateInstances(privacyAgent,data);
           privateData.setDebugMode(m_Debug);
           privateData.setSeed(getSeed());

           if (m_Debug)
                  System.out.println("Total number of instances: " + data.numInstances());
           if (m_maxNumInstances==0)
                  m_maxNumInstances =  (int) Math.pow(2,Math.ceil(Utils.log2(data.numInstances())));
           m_Scorer.InitializeMaxNumInstances(m_maxNumInstances);
           if (m_Debug)
                  System.out.println("MaxNumInstances: " + m_maxNumInstances);

           List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
           int numNumericAtts=0;
           Enumeration attEnum = data.enumerateAttributes();
           while (attEnum.hasMoreElements())
           {
                  Attribute att=(Attribute)attEnum.nextElement();
                  if (att.isNumeric())
                  {
                         candidateAttributes.add(new C45Attribute(att,m_numericAttributesFile));
                         numNumericAtts++;
                  }
                  else
                         candidateAttributes.add(new C45Attribute(att));
           }
           // adapt the differential privacy parameter from a global
           // parameter to a per-operation parameter
           // Due to partition operation, a quota should be given per level of depth
           // for each level of depth (node), we make three operations:
           // 1. noisy count on num instances (to decide whether to turn to leaf or split
           // 2. Determine split points for numeric attributes
           // 3. choose class (for leaf)   or choose splitting attribute (for node)
           int budgetForAttributeSelection=(m_Scorer.GetSensitivity()>0)? 1 : 0;
           int budgetForNumInstancesChecks=m_skipNumInstancesChecks?0:1;
           
           if(m_Debug){
           System.out.println("Epsilon is " + m_Epsilon);
           System.out.println("budget for attribute selection is " + budgetForAttributeSelection);
           System.out.println("Number of numeric attrbiutes is " + numNumericAtts);
           System.out.println("budget for checking number of instances " + budgetForNumInstancesChecks);
           System.out.println("Depth is " + m_MaxDepth);
           System.out.println("Divisor is " + ((budgetForNumInstancesChecks+budgetForAttributeSelection+numNumericAtts)*(m_MaxDepth)+budgetForNumInstancesChecks+1));
           }
           m_PrivacyBudgetPerAction =m_Epsilon.divide(BigDecimal.valueOf((budgetForNumInstancesChecks+budgetForAttributeSelection)*(m_MaxDepth)+budgetForNumInstancesChecks+1),DiffPrivacyClassifier.MATH_CONTEXT); // +2 accounts for leaf's budget              
           
           if (m_Debug)
                  System.out.println("epsilon per action is " + m_PrivacyBudgetPerAction);
           
           makeTree(privateData,candidateAttributes, m_MaxDepth);

            // Calibrate noisy counts in the tree to match each other
           fixNumInstancesTopDown(m_approxNumInstances);
           fixClassCountsBottomUp();

           if (!m_skipNumInstancesChecks && !m_Unpruned)
           {
                  if (m_Debug)
                  {
                         System.out.println("\n\nTree before pruning:");
                         System.out.println(toString());
                  }

                  prune();

                  if (m_Debug)
                  {
                         System.out.println("\n\nTree after pruning:");
                         System.out.println(toString());
                  }
           }
           
           //printTree("");

    }
    
	private String printDistribution(){
		
		String r= "[";
		for(int i=0;i<m_Distribution.length;i++){
			r+=new DecimalFormat("#0.00").format(m_Distribution[i])+", ";
		}
		
		r+="]";
		return r;
	}
    
	public String printTree(String pad){
		
		StringBuilder builder = new StringBuilder();
		
		if(m_Attribute == null){
			String newpad = pad + "-";
			
			System.out.println(newpad + printDistribution());
		}
		
		// not leaf
		else{
			
			System.out.println(pad + m_Attribute.WekaAttribute().name());
			
			String newpad = pad + " | ";
			for(int i=0; i<m_Successors.length;i++){
				builder.append(((C45Lap)m_Successors[i]).printTree(newpad));
			}
		}
		return builder.toString();
	}
}

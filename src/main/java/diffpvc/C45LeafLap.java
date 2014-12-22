package diffpvc;

import java.math.BigDecimal;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

public class C45LeafLap extends C45 {

	private static final long serialVersionUID = 1L;

	protected double m_Privacy = 0.1;	
	
	protected C45LeafLap CreateC45LeafLap() {
		C45LeafLap newTree = new C45LeafLap();
		newTree.m_Debug = m_Debug;
		newTree.m_Scorer = m_Scorer;
		newTree.m_Privacy = m_Privacy;
		newTree.m_MaxDepth = m_MaxDepth - 1; // not really used (other than
												// being received as a parameter
												// from the UI,
		newTree.m_CF = m_CF;
		newTree.m_Unpruned = m_Unpruned;
		newTree.m_ClassAttribute = m_ClassAttribute;
		newTree.m_skipNumInstancesChecks = m_skipNumInstancesChecks;
		return newTree;
	}
	
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);

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
		
		
		makeTree(data, candidateAttributes, m_MaxDepth, m_Random);

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
		
		fillEmptyLeaf(data);	

		if (getPrivacy() > 0) {
			addLapError(getPrivacy());
		}
	}
	
	protected void makeTree(Instances data,List<C45Attribute> candidateAttributes, int depth, Random random) 
			throws Exception {
		
		m_ClassAttribute = data.classAttribute();
		m_numInstances = data.numInstances();		
		m_Random = random;

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
		candidateAttributes.remove(m_Attribute); // attribute will not be
													// available in sub-trees
													// unless it is numeric

		if (m_Attribute.isNumeric()) {
			m_Successors = new C45LeafLap[2];// numeric attributes use binary splits
			m_SplitPoint = m_Attribute.getSplitPoint();

			C45Attribute left = new C45Attribute(m_Attribute,m_Attribute.lowerBound(), m_SplitPoint);
			candidateAttributes.add(left);
			m_Successors[0] = CreateC45LeafLap();
			m_Successors[0].makeTree(splitData[0], candidateAttributes,depth - 1, m_Random);
			candidateAttributes.remove(left);

			C45Attribute right = new C45Attribute(m_Attribute, m_SplitPoint,m_Attribute.upperBound());
			candidateAttributes.add(right);
			m_Successors[1] = CreateC45LeafLap();
			m_Successors[1].makeTree(splitData[1], candidateAttributes,depth - 1, m_Random);
			candidateAttributes.remove(right);
			
		} else {
			m_Successors = new C45LeafLap[m_Attribute.numValues()];
			for (int j = 0; j < m_Successors.length; j++) {
				m_Successors[j] = CreateC45LeafLap();
				m_Successors[j].makeTree(splitData[j], candidateAttributes,depth - 1, m_Random);
			}
		}
		candidateAttributes.add(m_Attribute);// make sure that the attribute
												// will be available for next
												// successors
	}
	
	public double getPrivacy() {
		return m_Privacy;
	}
	public void setPrivacy(String eStr){
		if (eStr!=null && eStr.length()!=0)
			m_Privacy = Double.parseDouble(eStr);
	}

	private double laplace(double beta) {
		//double beta = bigBeta.doubleValue();
		double uniform = m_Random.nextDouble() - 0.5;
		return 0.0
				- beta
				* ((uniform > 0) ? -Math.log(1. - 2 * uniform) : Math.log(1. + 2 * uniform));
	}

	protected void addLapError(double budget) {
		
		// non leaf node
		if (m_Attribute != null) {
			for (int i = 0; i < m_Successors.length; i++) {
				((C45LeafLap)m_Successors[i]).addLapError(budget);
			}
		}
		// leaf node
		else {
			for (int i = 0; i < m_Counts.length; i++) {
				m_Counts[i] += laplace(1.0/budget);
					//	BigDecimal.ONE.divide(BigDecimal.valueOf(budget)));
				
				if( m_Counts[i] < 0){
					m_Counts[i] = 0.0;
				}
			}

			// if any of them is less than 0, sum of them equals to 0;
			if(Utils.sum(m_Counts) == 0.0){
				m_Counts[m_Random.nextInt(m_Counts.length)] ++;
			}
			
			m_ClassValue = Utils.maxIndex(m_Counts);
			m_Distribution = turnToDistribution(m_Counts);
		}

	}
	
	protected void fillEmptyLeaf(Instances info) {
		
		// leaf
		if (m_Attribute == null) {
			if (m_Counts == null) {
				m_Counts = new double[info.numClasses()];
				m_Counts[m_Random.nextInt(info.numClasses())]++;
				
				m_ClassValue = Utils.maxIndex(m_Counts);
				m_Distribution = turnToDistribution(m_Counts);
			}
			return;
		
		// inner node
		} else {
			for (int i = 0; i < m_Successors.length; i++) {
				((C45LeafLap)m_Successors[i]).fillEmptyLeaf(info);
			}
		}

	}

}

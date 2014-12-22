package diffpvc.test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Utils4Test {
	
	public static Instances getDataFromFile(String path) throws Exception{		
		try{
			DataSource source = new DataSource(path);
			Instances data = source.getDataSet();
			
			// setting class attribute if the data format does not provide this information
			// For example, the XRFF format saves the class attribute information as well
			if (data.classIndex() == -1)
			  data.setClassIndex(data.numAttributes() - 1);	 
			return data;
		}catch(Exception e){
			System.err.println(path + " is not a legel path");
			System.exit(1);
		}
		return null;
	}
}

package rdt;

import java.util.Random;

public class Utils {
	
	static public double laplace(double b, Random random){

		if( b <= 0 ){			
			b = 0;
			System.err.println("privacy budget shouldn't be less than 0");
		}
		
		double rand = 10.0*random.nextDouble();
		return (b/2.0)*Math.exp(-rand*b);

	}

}

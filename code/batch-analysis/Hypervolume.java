
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

//import ga.Solution;

public class Hypervolume {


  /**
  * Constructor
  * Creates a new instance of MultiDelta
  */
  public Hypervolume() {

  } // Hypervolume

  /*
   returns true if 'point1' dominates 'points2' with respect to the
   to the first 'noObjectives' objectives
   */
  static boolean  dominates(double  point1[], double  point2[], int  noObjectives) {
    int  i;
    int  betterInAnyObjective;

    betterInAnyObjective = 0;
    for (i = 0; i < noObjectives && point1[i] >= point2[i]; i++)
      if (point1[i] > point2[i])
      	betterInAnyObjective = 1;

    return ((i >= noObjectives) && (betterInAnyObjective>0));
  } //Dominates

  static void  swap(double [][] front, int  i, int  j){
    double  [] temp;

    temp = front[i];
    front[i] = front[j];
    front[j] = temp;
  } // Swap


  /* all nondominated points regarding the first 'noObjectives' dimensions
  are collected; the points referenced by 'front[0..noPoints-1]' are
  considered; 'front' is resorted, such that 'front[0..n-1]' contains
  the nondominated points; n is returned */
  static int  filterNondominatedSet(double [][] front, int  noPoints, int  noObjectives){
    int  i, j;
    int  n;

    n = noPoints;
    i = 0;
    while (i < n) {
      j = i + 1;
      while (j < n) {
        if (dominates(front[i], front[j], noObjectives)) {
	/* remove point 'j' */
	  n--;
	  swap(front, j, n);
        } else if (dominates(front[j], front[i], noObjectives)) {
	/* remove point 'i'; ensure that the point copied to index 'i'
	   is considered in the next outer loop (thus, decrement i) */
	  n--;
	  swap(front, i, n);
	  i--;
	  break;
        } else
	  j++;
      }
      i++;
    }
    return n;
  } // FilterNondominatedSet


  /* calculate next value regarding dimension 'objective'; consider
     points referenced in 'front[0..noPoints-1]' */
  static double surfaceUnchangedTo(double [][] front, int  noPoints, int  objective) {
    int     i;
    double  minValue, value;

    if (noPoints < 1)
      System.err.println("run-time error");

    minValue = front[0][objective];
    for (i = 1; i < noPoints; i++) {
      value = front[i][objective];
      if (value < minValue)
        minValue = value;
    }
    return minValue;
  } // SurfaceUnchangedTo

  /* remove all points which have a value <= 'threshold' regarding the
     dimension 'objective'; the points referenced by
     'front[0..noPoints-1]' are considered; 'front' is resorted, such that
     'front[0..n-1]' contains the remaining points; 'n' is returned */
  static int  reduceNondominatedSet(double [][] front, int  noPoints, int  objective,
			   double  threshold){
    int  n;
    int  i;

    n = noPoints;
    for (i = 0; i < n; i++)
      if (front[i][objective] <= threshold) {
        n--;
        swap(front, i, n);
      }

    return n;
  } // ReduceNondominatedSet


  public static double calculateHypervolume(double [][] front, int  noPoints,int  noObjectives){
    int     n;
    double  volume, distance;

    volume = 0;
    distance = 0;
    n = noPoints;
    while (n > 0) {
      int     noNondominatedPoints;
      double  tempVolume, tempDistance;

      noNondominatedPoints = filterNondominatedSet(front, n, noObjectives - 1);
      //noNondominatedPoints = front.length;
      if (noObjectives < 3) {
        if (noNondominatedPoints < 1)
          System.err.println("run-time error");

        tempVolume = front[0][0];
      } else
        tempVolume = calculateHypervolume(front,
                                          noNondominatedPoints,
                                          noObjectives - 1);

      tempDistance = surfaceUnchangedTo(front, n, noObjectives - 1);
      volume += tempVolume * (tempDistance - distance);
      distance = tempDistance;
      n = reduceNondominatedSet(front, n, noObjectives - 1, distance);
    }
    return volume;
  } // CalculateHypervolume


  /* merge two fronts */
  static double [][] mergeFronts(double [][] front1, int  sizeFront1,
		 double [][] front2, int  sizeFront2, int  noObjectives)
  {
    int  i, j;
    int  noPoints;
    double [][] frontPtr;

    /* allocate memory */
    noPoints = sizeFront1 + sizeFront2;
    frontPtr = new double[noPoints][noObjectives];
    /* copy points */
    noPoints = 0;
    for (i = 0; i < sizeFront1; i++) {
      for (j = 0; j < noObjectives; j++)
        frontPtr[noPoints][j] = front1[i][j];
      noPoints++;
    }
    for (i = 0; i < sizeFront2; i++) {
      for (j = 0; j < noObjectives; j++)
        frontPtr[noPoints][j] = front2[i][j];
      noPoints++;
    }

    return frontPtr;
  } // MergeFronts



  /***************************************************************************************
   * COPIED FROM MetricUtils by JRU
   */

  /**
   * This method receives a pareto front and two points, one whit maximum values
   * and the other with minimum values allowed, and returns a the normalized
   * Pareto front.
   * @param front A pareto front.
   * @param maximumValue The maximum values allowed
   * @param minimumValue The minimum values allowed
   * @return the normalized pareto front
   **/
  protected static double [][] getNormalizedFront(double [][] front,
                                        double [] maximumValue,
                                        double [] minimumValue) {

    double [][] normalizedFront = new double[front.length][];

    for (int i = 0; i < front.length;i++) {
      normalizedFront[i] = new double[front[i].length];
      for (int j = 0; j < front[i].length; j++) {
        normalizedFront[i][j] = (front[i][j] - minimumValue[j]) /
                                (maximumValue[j] - minimumValue[j]);
      }
    }
    return normalizedFront;
  } // getNormalizedFront


  /**
   * This method receives a pareto front and two points, one whit maximum values
   * and the other with minimum values allowed, and returns a the normalized
   * Pareto front.
   * @param front A pareto front.
   * @param //maximumValue The maximum values allowed
   * @param //minimumValue The minimum values allowed
   * @return the normalized pareto front
   **/
  protected static double [][] getRescaledFront(double [][] front,
                                        double scalingFactor) {

    double [][] rescaledFront = new double[front.length][];

    for (int i = 0; i < front.length;i++) {
      rescaledFront[i] = new double[front[i].length];
      for (int j = 0; j < front[i].length; j++) {
        rescaledFront[i][j] = ((front[i][j]) / scalingFactor) ;
      }
    }
    return rescaledFront;
  } // getNormalizedFront

  /** Gets the maximum values for each objectives in a given pareto
   *  front
   *  @param front The pareto front
   *  @param noObjectives Number of objectives in the pareto front
   *  @return double [] An array of noOjectives values whit the maximun values
   *  for each objective
   **/
  protected static  double [] getMaximumValues(double [][] front, int noObjectives) {
    double [] maximumValue = new double[noObjectives];
    for (int i = 0; i < noObjectives; i++)
      maximumValue[i] =  Double.NEGATIVE_INFINITY;
    for (double[] aFront : front) {
      for (int j = 0; j < aFront.length; j++) {
        if (aFront[j] > maximumValue[j])
          maximumValue[j] = aFront[j];
      }
    }
    return maximumValue;
  } // getMaximumValues


  /** Gets the minimum values for each objectives in a given pareto
   *  front
   *  @param front The pareto front
   *  @param noObjectives Number of objectives in the pareto front
   *  @return double [] An array of noOjectives values whit the minimum values
   *  for each objective
   **/
  protected static double [] getMinimumValues(double [][] front, int noObjectives) {
    double [] minimumValue = new double[noObjectives];
    for (int i = 0; i < noObjectives; i++)
      minimumValue[i] = Double.MAX_VALUE;

    for (double[] aFront : front) {
      for (int j = 0; j < aFront.length; j++) {
        if (aFront[j] < minimumValue[j])
          minimumValue[j] = aFront[j];
      }
    }
    return minimumValue;
  } // getMinimumValues


  /**
   * This method receives a normalized pareto front and return the inverted one.
   * This operation needed for minimization problems
   * @param front The pareto front to inverse
   * @return The inverted pareto front
   **/
  protected static  double[][] invertedFront(double [][] front) {
    double [][] invertedFront = new double[front.length][];

    for (int i = 0; i < front.length; i++) {
      invertedFront[i] = new double[front[i].length];
      for (int j = 0; j < front[i].length; j++) {
        if (front[i][j] <= 1.0 && front[i][j]>= 0.0) {
          invertedFront[i][j] = 1.0 - front[i][j];
        } else if (front[i][j] > 1.0) {
          invertedFront[i][j] = 0.0;
        } else if (front[i][j] < 0.0) {
          invertedFront[i][j] = 1.0;
        }
      }
    }
    return invertedFront;
  } // invertedFront


  public static void main(String[] args) throws IOException {
      //String path = "/home/marharyta/Desktop/cloud-myfiles/RESEARCH/Jean Ruppert - Statistics - Pareto/genetic algos/knapsack/nsgaiii-master/deep-tests/";
      String path = "";
      String file_name = path + "fitness_vals.txt";
      ArrayList<double[]> resList = new ArrayList<>();
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file_name));
      String line;
      while ((line=bufferedReader.readLine()) != null){
          String[] tmp = line.split(" ");
          double[] vals = new double[tmp.length];
          for (int i=0; i< tmp.length; i++){
              vals[i] = Double.parseDouble(tmp[i]);
          }
          resList.add(vals);
      }
      bufferedReader.close();
      double[][] front = new double[resList.size()][resList.get(0).length];
      for (int i=0; i<resList.size(); i++){
          front[i] = resList.get(i);
      }
      long time = System.nanoTime();
      int numTimes = Integer.parseInt(args[0]);
      int hypervolumeVal = computeHypervolumeMonteCarloInt(front, new Random(), numTimes);
      double seconds = (System.nanoTime() - time) / 1000000000.;
      //System.out.println("Time: " + seconds);
      //System.out.println(hypervolumeVal);
      PrintWriter pw = new PrintWriter(path + "hypervolume.txt");
      pw.write(String.valueOf(hypervolumeVal) + '\n');
      pw.write("Time: " + seconds);
      pw.close();
  }

    public static void min(double[] accumulator, long[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.min(accumulator[i], other[i]);
        }
    }

    public static void min(double[] accumulator, double[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.min(accumulator[i], other[i]);
        }
    }

    public static void min(long[] accumulator, long[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.min(accumulator[i], other[i]);
        }
    }

    public static void max(double[] accumulator, long[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.max(accumulator[i], other[i]);
        }
    }

    public static void max(double[] accumulator, double[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.max(accumulator[i], other[i]);
        }
    }

    public static void max(long[] accumulator, long[] other) {
        for (int i = 0; i< accumulator.length; i++) {
            accumulator[i]=Math.max(accumulator[i], other[i]);
        }
    }

    public static int computeHypervolumeMonteCarloInt(double[][] firstFront, Random random, int n) {
        int nK;
        if (firstFront.length==0) {
            return 0;
        }
        int firstFrontSize= firstFront.length;
        nK = firstFront[0].length;
        //int n =10000000;
        double [] maxProfits = new double[nK];
        Arrays.fill(maxProfits, Double.MIN_VALUE);
        double [] minProfits = new double[nK];
        Arrays.fill(minProfits, Double.MAX_VALUE);
        for(int s=0; s < firstFrontSize; s++) {
            max(maxProfits, firstFront[s]);
            min(minProfits, firstFront[s]);
        }
        // normalize first front
        double[][] firstFrontRenormalized = new double[firstFrontSize][];
        for(int s=0; s < firstFrontSize; s++) {
            firstFrontRenormalized[s] = new double[nK];
            for (int iK=0; iK<nK; iK++) {
                firstFrontRenormalized[s][iK] = firstFront[s][iK]/maxProfits[iK];
            }
        }
        int nWeaklyDominated =0;
        for (int i=0; i< n; i++) {
            nWeaklyDominated += weaklyDominatedIndicatorFunction(firstFrontRenormalized, randomSolution(nK, random));
        }
        return nWeaklyDominated;
    }

  public static double computeHypervolumeMonteCarlo(double[][] firstFront, Random random) {
	  int nK;
	  if (firstFront.length==0) {
		  return 0;
	  }
	  int firstFrontSize= firstFront.length;
	  nK = firstFront[0].length;
	  int n =10000000;
	  double [] maxProfits = new double[nK];
	  Arrays.fill(maxProfits, Double.MIN_VALUE);
	  double [] minProfits = new double[nK];
	  Arrays.fill(minProfits, Double.MAX_VALUE);
	  for(int s=0; s < firstFrontSize; s++) {
		   max(maxProfits, firstFront[s]);
		   min(minProfits, firstFront[s]);
	  }
	  // normalize first front
	  double[][] firstFrontRenormalized = new double[firstFrontSize][];
	  for(int s=0; s < firstFrontSize; s++) {
		  firstFrontRenormalized[s] = new double[nK];
		  for (int iK=0; iK<nK; iK++) {
			  firstFrontRenormalized[s][iK] = firstFront[s][iK]/maxProfits[iK];
		  }
	  }
	  long nWeaklyDominated =0;
	  for (int i=0; i< n; i++) {
		  nWeaklyDominated += weaklyDominatedIndicatorFunction(firstFrontRenormalized, randomSolution(nK, random));
	  }
	  return ((double) nWeaklyDominated / (double) n);
  }

  public static double hypervolumeMonteCarloScalingFactor(double[][] firstFront) {
	  int nK;
	  if (firstFront.length==0) {
		  return 0;
	  }
	  int firstFrontSize= firstFront.length;
	  nK = firstFront[0].length;
	  double [] maxProfits = new double[nK];
	  Arrays.fill(maxProfits, Long.MIN_VALUE);
	  for(int s=0; s < firstFrontSize; s++) {
		  max(maxProfits, firstFront[s]);
	  }
	  double sum_ln_maxProfit = 0;
	  for (int iK=0; iK<nK; iK++) {
		  sum_ln_maxProfit +=Math.log(maxProfits[iK]);
	  }
	  return Math.exp(sum_ln_maxProfit/nK);
  }

  static final long s1StrictlyDominatesS2 = 0;
  static final long s2WeaklyDominatesS1 = 1;
  static final long noDomination = 2;
  private static final long allEqualUntilNow = -1;
  private static final long s2StrictlyDominatesS1 = 1;

  /**
   * indicates wether s1 strictly dominates s2, s2 weakly dominates s1, or no domination
   * <p> USAGE: Monte-Carlo integration of hypervolume (hypervolume is the volume weakly dominated by the first front))
   *
   * @param s1 coordinates of first solution
   * @param s2 coordinates of second solution
   * @return <ul>
   * <li> 0 if s1 strictly dominates s2 (s1StrictlyDominatesS2)
   * <li> 1 if s2 weakly dominates s1 (s2WeaklyDominatesS1 )
   * <li> 2 otherwise (noDomination)
   * </ul>
   */
  static long domination(double[] s1, double[] s2) {
	  int nK = s1.length;
	  long previous = allEqualUntilNow ; // -1 means all equal until now
	  int i=0;
	  int nKminus1 = nK-1;
	  while (i < nKminus1) {
		  if (previous ==allEqualUntilNow ) {
			  if (s1[i] > s2[i]) {
				  previous = s1StrictlyDominatesS2;
			  } else if(s1[i] < s2[i]) {
				  previous = s2StrictlyDominatesS1; // 1 now means strictly
			  } else {
				  previous =allEqualUntilNow ;
			  }
		  } else if(previous ==s1StrictlyDominatesS2) {
			  if (s1[i] >= s2[i]) {
				  previous = s1StrictlyDominatesS2;
			  } else {
				  return noDomination ;
			  }
		  } else if(previous ==s2StrictlyDominatesS1) {
			  if (s2[i] >= s1[i]) {
				  previous = s2StrictlyDominatesS1;
			  } else {
				  return noDomination ;
			  }
		  }
		  i++;
	  }
	  if (previous ==-1) {
		  if (s1[i] > s2[i]) {
			  return s1StrictlyDominatesS2;
		  } else {
			  return  s2WeaklyDominatesS1 ; // 1 now means weakly
		  }
	  } else if(previous ==0) {
		  if (s1[i] >= s2[i]) {
			  return s1StrictlyDominatesS2;
		  } else {
			  return noDomination ;
		  }
	  } else if(previous ==1) {
		  if (s2[i] >= s1[i]) {
			  return s2WeaklyDominatesS1  ;
		  } else {
			  return noDomination ;
		  }
	  }
	  throw new Error("s2WeaklyDominatesS1: programming error unforeseen case");
  }

  /**
   *
   * @param front
   * @param solution
   * @return 1 if solution is weakly dominated by an element of the front
   *         0 otherwise
   */
  static long weaklyDominatedIndicatorFunction(double[][] front, double[] solution) {
	  int frontLength = front.length;
	  int iFront =0;
	  while (iFront < frontLength) {
	  		long d =  domination(solution, front[iFront]);
	  		if(d !=2) {
	  			return d;
	  		}
	  		iFront ++;
	  }
	  return 0;
  }

  private static double[] randomSolution(int nK, Random random) {
	  double[] result = new double[nK];
	  for (int i=0; i<nK; i++) {
		  result[i]=random.nextDouble();
	  }
	  return result;
  }


}

package com.github.trinity.supermds;

import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.PointValuePair;

/**
 * A convergence checker that considers both:
 * 1. The relative/absolute change in function value, and
 * 2. The relative/absolute change in parameter vector.
 *
 * @author Sean Phillips
 */
public class CombinedConvergenceChecker implements ConvergenceChecker<PointValuePair> {

    private final double relTolValue;
    private final double absTolValue;
    private final double relTolParam;
    private final double absTolParam;

    /**
     * @param relTolValue Relative tolerance for function value change
     * @param absTolValue Absolute tolerance for function value change
     * @param relTolParam Relative tolerance for parameter vector change
     * @param absTolParam Absolute tolerance for parameter vector change
     */
    public CombinedConvergenceChecker(double relTolValue, double absTolValue,
                                      double relTolParam, double absTolParam) {
        this.relTolValue = relTolValue;
        this.absTolValue = absTolValue;
        this.relTolParam = relTolParam;
        this.absTolParam = absTolParam;
    }

    @Override
    public boolean converged(int iteration, PointValuePair previous, PointValuePair current) {
        // Check function value change
        double prevValue = previous.getValue();
        double currValue = current.getValue();
        double diffValue = Math.abs(prevValue - currValue);
        double maxValue = Math.max(Math.abs(prevValue), Math.abs(currValue));
        boolean valueConverged = diffValue <= Math.max(relTolValue * maxValue, absTolValue);

        // Check parameter vector change
        double[] prevPoint = previous.getPoint();
        double[] currPoint = current.getPoint();
        double maxParamDiff = 0.0;
        for (int i = 0; i < prevPoint.length; i++) {
            double diff = Math.abs(prevPoint[i] - currPoint[i]);
            double maxAbs = Math.max(Math.abs(prevPoint[i]), Math.abs(currPoint[i]));
            double tol = Math.max(relTolParam * maxAbs, absTolParam);
            if (diff > tol) {
                maxParamDiff = diff;
                break;  // early exit if any component fails
            }
        }
        boolean paramConverged = maxParamDiff == 0.0;

        return valueConverged && paramConverged;
    }
}

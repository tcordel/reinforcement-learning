package fr.tcordel.rl.neural;

import java.util.Arrays;

public abstract class NeuralUtils {

	public static int[] oneHot(double[] values) {
		assert values != null && values.length > 0;
		int[] hot = new int[values.length];
		Arrays.fill(hot, 0, hot.length, 0);
		double max = Double.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > max) {
				index = i;
				max = values[i];
			}
		}
		hot[index] = 1;
		return hot;
	}

	public static double mse(double[] expected, double[] actual, int[] oneHot) {
		double result = 0;
		int n = 0;
		for (int i = 0; i < expected.length; i++) {
			if (oneHot[i] == 1) {
				result += Math.pow(expected[i] - actual[i], 2);
				n++;
			}
		}
		return result / n;
	}

}

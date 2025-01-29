package fr.tcordel.basic;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
class CodinGame2 {

	public static void main(String args[]) {
		Scanner in = new Scanner(System.in);
		int tests = in.nextInt();
		int trainingSets = in.nextInt();
		List<double[]> testList = new ArrayList<>();
		List<double[]> trainingInputList = new ArrayList<>();
		List<double[]> trainingOutputList = new ArrayList<>();

		NeuralNetwork nn = new NeuralNetwork(8, 16, 8);
		for (int i = 0; i < tests; i++) {
			testList.add(CodinGame2.getCharArray(in));
		}
		for (int i = 0; i < trainingSets; i++) {
			trainingInputList.add(CodinGame2.getCharArray(in));
			trainingOutputList.add(CodinGame2.getCharArray(in));
		}

		List<Integer> indexes = IntStream.range(0, trainingSets)
				.mapToObj(i -> i)
				.collect(Collectors.toList());

		// for (int j = 0; j < trainingSets; j++) {
		// double cumulativeMse = 0d;
		boolean hasError = true;
		while (hasError) {

			hasError = false;
			Collections.shuffle(indexes);
			double cumulativeMse = 0d;
			for (int i : indexes) {
				double[] out = trainingOutputList.get(i);
				String expected = doublesToString(out);
				double[] o = nn.train(trainingInputList.get(i), out);
				hasError |= !expected.equals(doublesToString(o));

				double mse = 0d;
				for (int index = 0; index < out.length; index++) {
					mse += Math.pow(out[index] - o[index], 2);
				}
				mse *= 1d / o.length;
				System.err.println("data %d mse %f".formatted(i, mse));

				cumulativeMse += mse;
			}
			System.err.println("cumulativeMse %f".formatted(cumulativeMse));
		}
		for (int j = 0; j < 50; j++) {
			for (int i = 0; i < trainingSets; i++) {
				double[] out = trainingOutputList.get(i);
				double[] o = nn.train(trainingInputList.get(i), out);
			}
		}

		for (int i = 0; i < tests; i++) {
			double[] predict = nn.predict(testList.get(i));
			String answer = doublesToString(predict);
			System.out.println(answer);
		}
	}

	static String doublesToString(double[] predict) {
		String answer = Arrays.stream(predict)
				.map(Math::round)
				.mapToInt(d -> (int) d)
				.mapToObj(String::valueOf)
				.collect(Collectors.joining(""));
		return answer;
	}

	private static double[] getCharArray(Scanner in) {
		String next = in.next();
		System.err.println(next);
		char[] charArray = next.toCharArray();
		return IntStream.range(0, charArray.length)
				.mapToDouble(c -> Double.parseDouble(String.valueOf(charArray[c]))).toArray();
	}
}

class NeuralNetwork {

	private final int[] layers;
	private final double eta = 0.3d;
	double[][] o;
	double[][] deltas;
	// layer, output, input
	double[][][] weights;
	double[][] thetas;
	BigDecimal initValue = null;

	private static final BigDecimal MOD = new BigDecimal("2147483648");
	private static final BigDecimal DIVISOR = new BigDecimal("2147483647");

	public NeuralNetwork(int... layers) {
		if (layers.length < 2) {
			throw new IllegalStateException("Requires at least an input and outputs");
		}
		this.layers = layers;
		weights = new double[layers.length - 1][][];
		thetas = new double[layers.length - 1][];
		o = new double[layers.length][];
		deltas = new double[layers.length - 1][];
		o[0] = new double[layers[0]];
		int maxSize = -1;
		for (int i = 1; i < layers.length; i++) {
			int layerSize = layers[i];
			int previousLayerSize = layers[i - 1];
			maxSize = Math.max(maxSize, Math.max(layerSize, previousLayerSize));
			thetas[i - 1] = new double[layerSize];
			o[i] = new double[layerSize];
			deltas[i - 1] = new double[layerSize];
			weights[i - 1] = new double[previousLayerSize][];
			for (int j = 0; j < previousLayerSize; j++) {
				weights[i - 1][j] = new double[layerSize];
			}
		}

		for (int layer = 1; layer < layers.length; layer++) {
			int layerSize = layers[layer];
			int previousLayerSize = layers[layer - 1];
			for (int j = 0; j < layerSize; j++) {
				for (int i = 0; i < previousLayerSize; i++) {
					// System.err.println("layer %d : w[i%d,o%d]".formatted(layer, i,j));
					weights[layer - 1][i][j] = getWeightInitValue();
				}
				// System.err.println("layer %d : t[o%d]".formatted(layer, j));
				thetas[layer - 1][j] = getWeightInitValue();
			}
		}
	}

	double getWeightInitValue() {
		if (initValue == null) {
			initValue = new BigDecimal("1103527590");
		} else {
			BigDecimal d = initValue.multiply(new BigDecimal("1103515245")).add(new BigDecimal("12345"));
			initValue = d.remainder(MOD);
		}

		double weight = initValue.divide(DIVISOR, MathContext.DECIMAL128).doubleValue();
		return weight;
	}

	private double sigmoid(double value) {
		return 1d / (1 + Math.exp(-value));
	}

	private void frontward(double[] ins) {
		System.arraycopy(ins, 0, o[0], 0, ins.length);
		for (int i = 1; i < layers.length; i++) {
			double[] previousLayer = o[i - 1];
			double[] layer = o[i];
			Arrays.fill(layer, 0, layer.length, 0);
			for (int k = 0; k < layer.length; k++) {
				for (int j = 0; j < previousLayer.length; j++) {
					layer[k] += (previousLayer[j] * weights[i - 1][j][k]);
				}
				layer[k] += thetas[i - 1][k];
				layer[k] = sigmoid(layer[k]);
			}
		}
	}

	public double[] train(double[] ins, double[] out) {
		frontward(ins);

		for (int index = o.length - 1; index > 0; index--) {
			int layer = index - 1;
			for (int nodeIndex = 0; nodeIndex < deltas[layer].length; nodeIndex++) {
				// String debug = "layer " + layer + "o[j] = o[j]*(1-o[j])";
				deltas[layer][nodeIndex] = o[index][nodeIndex]
						* (1 - o[index][nodeIndex]);
				if (index == o.length - 1) {
					// debug += "*(o[j] - tj)";
					deltas[layer][nodeIndex] *= (o[index][nodeIndex] - out[nodeIndex]);
				} else {
					double nextLayerWeight = 0;
					// debug += "(";
					for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < deltas[index].length; nextLayerNodeIndex++) {
						// debug += "+d[k]*w[j,k]";
						if (nodeIndex == 5) {
							int a = 1;

						}
						nextLayerWeight += deltas[index][nextLayerNodeIndex]
								* weights[index][nodeIndex][nextLayerNodeIndex];
					}
					// debug += ")";
					deltas[layer][nodeIndex] *= nextLayerWeight;
				}
				// System.err.println(debug);
			}
		}

		for (int layer = 0; layer < weights.length; layer++) {
			for (int j = 0; j < weights[layer].length; j++) {
				for (int k = 0; k < weights[layer][j].length; k++) {
					weights[layer][j][k] += -eta * deltas[layer][k] * o[layer][j];
				}
			}
			for (int ds = 0; ds < thetas[layer].length; ds++) {
				thetas[layer][ds] += -eta * deltas[layer][ds];
			}
		}
		return o[o.length - 1];
	}

	public double[] predict(double[] ins) {
		frontward(ins);
		return o[o.length - 1];
	}

}

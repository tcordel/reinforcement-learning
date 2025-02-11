package fr.tcordel.rl.neural;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

public class NeuralNetwork {

	private final int[] layers;
	// private final double eta = 0.0005d;
	private final double eta = 0.001d;
	public double[][] o;
	double[][] deltas;
	// layer, output, input
	public double[][][] weights;
	public double[][] biases;
	BigDecimal initValue = null;
	ActivationFonction[] activationFonctions;

	public NeuralNetwork(NeuralNetwork of) {
		this.activationFonctions = Arrays.copyOf(of.activationFonctions, of.activationFonctions.length);
		this.layers = new int[of.layers.length];
		System.arraycopy(of.layers, 0, layers, 0, of.layers.length);
		this.o = init(of.o);
		this.deltas = init(of.deltas);
		this.biases = init(of.biases);
		this.weights = new double[of.weights.length][][];
		for (int i = 0; i < of.weights.length; i++) {
			this.weights[i] = init(of.weights[i]);
		}
		load(of);
	}

	public void load(NeuralNetwork of) {
		copy(of.o, o);
		copy(of.deltas, deltas);
		copy(of.biases, biases);
		for (int i = 0; i < of.weights.length; i++) {
			copy(of.weights[i], weights[i]);
		}
	}

	public static double[][] init(double[][] from) {
		double[][] init = new double[from.length][];
		for (int i = 0; i < from.length; i++) {
			init[i] = new double[from[i].length];
		}
		return init;
	}

	private static void copy(double[][] from, double[][] to) {
		for (int i = 0; i < from.length; i++) {
			System.arraycopy(from[i], 0, to[i], 0, from[i].length);
		}
	}

	public NeuralNetwork(DoubleSupplier weightInitialisor, int... layers) {
		if (layers.length < 2) {
			throw new IllegalStateException("Requires at least an input and outputs");
		}
		this.layers = layers;
		weights = new double[layers.length - 1][][];
		biases = new double[layers.length - 1][];
		o = new double[layers.length][];
		deltas = new double[layers.length - 1][];
		o[0] = new double[layers[0]];
		int maxSize = -1;
		for (int i = 1; i < layers.length; i++) {
			int layerSize = layers[i];
			int previousLayerSize = layers[i - 1];
			maxSize = Math.max(maxSize, Math.max(layerSize, previousLayerSize));
			biases[i - 1] = new double[layerSize];
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
					weights[layer - 1][i][j] = weightInitialisor.getAsDouble();
				}

				biases[layer - 1][j] = weightInitialisor.getAsDouble();
			}
		}
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
				layer[k] += biases[i - 1][k];
				layer[k] = activationFonctions[i - 1].forward(layer[k]);
			}
		}
	}

	public double[] train(double[] ins, double[] out) {
		return trainMse(List.of(ins), List.of(out),
				List.of(IntStream.range(0, out.length).map(i -> 1).toArray()));
	}

	public double[] trainMse(List<double[]> ins, List<double[]> outs, List<int[]> oneHots) {


		List<double[][]> deltaList = new ArrayList<>();
		List<double[][]> oList = new ArrayList<>();

		for (int i = 0; i < ins.size(); i++) {
			for (int j = 0; j < deltas.length; j++) {
				Arrays.fill(deltas[j], 0, deltas[j].length, 0d);
			}
			double[] in = ins.get(i);
			double[] out = outs.get(i);
			int[] oneHot = oneHots.get(i);

			frontward(in);

			for (int index = o.length - 1; index > 0; index--) {
				int layer = index - 1;
				for (int nodeIndex = 0; nodeIndex < deltas[layer].length; nodeIndex++) {
					deltas[layer][nodeIndex] = activationFonctions[index - 1].backward(o[index][nodeIndex]);
					if (index == o.length - 1) {
						deltas[layer][nodeIndex] *= (2 / ins.size()) * (o[index][nodeIndex] - out[nodeIndex]) * oneHot[nodeIndex];
					} else {
						double nextLayerWeight = 0;
						for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < deltas[index].length; nextLayerNodeIndex++) {
							nextLayerWeight += deltas[index][nextLayerNodeIndex]
									* weights[index][nodeIndex][nextLayerNodeIndex];
						}
						deltas[layer][nodeIndex] *= nextLayerWeight;
					}
				}
			}
			oList.add(clone(o));
			deltaList.add(clone(deltas));
		}

		for (int layer = 0; layer < weights.length; layer++) {
			for (int j = 0; j < weights[layer].length; j++) {
				for (int k = 0; k < weights[layer][j].length; k++) {
					for (int l = 0; l < oList.size(); l++) {
						double dW = oList.get(l)[layer][j] * deltaList.get(l)[layer][k];
						weights[layer][j][k] += -eta * dW / ins.size();
					}
				}
			}
			for (int ds = 0; ds < biases[layer].length; ds++) {
					double dB = 0d;
					for (int l = 0; l < oList.size(); l++) {
						dB += deltaList.get(l)[layer][ds];
					}
				biases[layer][ds] += -eta * dB / ins.size();
			}
		}
		return o[o.length - 1];
	}

	public static double[][] clone(double[][] ds) {
		double[][] clone = init(ds);
		copy(ds, clone);
		return clone;
	}

	public double[] predict(double[] ins) {
		frontward(ins);
		return Arrays.copyOf(o[o.length - 1], o[o.length - 1].length);
	}

	public List<double[]> predict(List<double[]> ins) {
		return ins
				.stream()
				.map(this::predict)
				.toList();
	}

	record Layer(int size, ActivationFonction activationFonction) {
	};

	record Network(int in, List<Layer> hidden, Layer out) {
	}

	public void setActivationFonctions(ActivationFonction... activationFonctions) {
		this.activationFonctions = activationFonctions;
	}
}

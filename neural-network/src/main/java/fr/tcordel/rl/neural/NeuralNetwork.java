package fr.tcordel.rl.neural;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

public class NeuralNetwork {

	private final int[] layers;
	private final double eta = 0.5d;
	double[][] o;
	double[][] deltas;
	// layer, output, input
	double[][][] weights;
	double[][] thetas;
	BigDecimal initValue = null;
	ActivationFonction[] activationFonctions;

	public NeuralNetwork(NeuralNetwork of) {
		this.activationFonctions = Arrays.copyOf(of.activationFonctions, of.activationFonctions.length);
		this.layers = new int[of.layers.length];
		System.arraycopy(of.layers, 0, layers, 0, of.layers.length);
		this.o = init(of.o);
		this.deltas = init(of.deltas);
		this.thetas = init(of.thetas);
		this.weights = new double[of.weights.length][][];
		for (int i = 0; i < of.weights.length; i++) {
			this.weights[i] = init(of.weights[i]);
		}
		load(of);
	}

	public void load(NeuralNetwork of) {
		copy(of.o, o);
		copy(of.deltas, deltas);
		copy(of.thetas, thetas);
		for (int i = 0; i < of.weights.length; i++) {
			copy(of.weights[i], weights[i]);
		}
	}

	private double[][] init(double[][] from) {
		double[][] init = new double[from.length][];
		for (int i = 0; i < from.length; i++) {
			init[i] = new double[from[i].length];
		}
		return init;
	}

	private void copy(double[][] from, double[][] to) {
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
					weights[layer - 1][i][j] = weightInitialisor.getAsDouble();
				}
				thetas[layer - 1][j] = weightInitialisor.getAsDouble();
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
				layer[k] += thetas[i - 1][k];
				layer[k] = activationFonctions[i - 1].forward(layer[k]);
			}
		}
	}

	public double[] train(double[] ins, double[] out) {
		return train(ins, out, IntStream.range(0, out.length).map(i -> 1).toArray());
	}

	public double[] train(double[] ins, double[] out, int[] oneHot) {
		frontward(ins);

		for (int i = 0; i < deltas.length; i++) {
			Arrays.fill(deltas[i], 0, deltas[i].length, 0d);
		}

		for (int index = o.length - 1; index > 0; index--) {
			int layer = index - 1;
			for (int nodeIndex = 0; nodeIndex < deltas[layer].length; nodeIndex++) {
				deltas[layer][nodeIndex] = o[index][nodeIndex]
						* activationFonctions[index - 1].backward(o[index][nodeIndex]);
				if (index == o.length - 1) {
					deltas[layer][nodeIndex] *= (o[index][nodeIndex] - out[nodeIndex]) * oneHot[nodeIndex];
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

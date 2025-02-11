package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.rl.neural.ActivationFonction;
import fr.tcordel.rl.neural.NeuralNetwork;
import fr.tcordel.rl.neural.NeuralUtils;
import fr.tcordel.rl.neural.WeightInitializor;
import fr.tcordel.utils.Matplot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class DQN {
	double gamma = 0.99;
	double alpha = 0.1;
	double epsilon = 1;
	double epsilonFinal = 0.1;
	int numEpisodes = 15000;
	double epsilonDecay = epsilon / (3000);
	int batchSize = 64;	
	int optimizationIteration = 30;
	int samplingSize = batchSize * optimizationIteration;

	private final NeuralNetwork qModel;
	private final NeuralNetwork qTargetModel;
	private final Random random = new Random();
	private final Memory memory = new Memory();

	public static void main(String[] args) {
		new DQN().train();

	}

	DQN() {
		qModel = new NeuralNetwork(WeightInitializor.ZERO, 4, 64, 2);
		qModel.setActivationFonctions(ActivationFonction.RELU, ActivationFonction.NONE);
		qTargetModel = new NeuralNetwork(qModel);
		initAdam();
	}

	class Memory {
		private final int size = 10000;
		private final List<Dump> memories = new ArrayList<>(size);

		void add(Dump state) {
			if (memories.size() == size) {
				memories.remove(0);
			}
			memories.add(state);
		}

		List<Dump> sample(int batchSize) {
			List<Integer> index = new ArrayList<>();
			IntStream.range(0, memories.size())
					.forEach(index::add);
			Collections.shuffle(index);
			List<Dump> batch = new ArrayList<>(batchSize);
			for (int i = 0; i < batchSize; i++) {
				batch.add(memories.get(index.get(i)));
			}
			return batch;
		}

		public int size() {
			return memories.size();
		}

		public void clear() {
			memories.clear();
		}
	}

	int pickSample(double[] state, double epsilon) {
		if (random.nextDouble() > epsilon) {
			double[] predict = qModel.predict(state);
			double maxValue = predict[0];
			int index = 0;
			for (int i = 1; i < predict.length; i++) {
				if (predict[i] > maxValue) {
					maxValue = predict[i];
					index = i;
				}
			}
			return index;
		} else {
			return random.nextInt(0, 2);
		}
	}

	private void train() {
		CartPole cartPole = new CartPole();
		List<Double> rewards = new ArrayList<>();
		for (int i = 0; i < numEpisodes; i++) {
			double cumReward = 0;
			boolean done = true;
			double[] state = null;
			for (int unused = 0; unused < 500; unused++) {
				if (done) {
					state = cartPole.reset();
					done = false;
					cumReward = 0;
				}

				int action = pickSample(state, epsilon);
				StepResult stepResult = cartPole.step(action);
				double[] newState = stepResult.state();
				done = stepResult.truncated() || stepResult.terminated();
				// memory
				double reward = stepResult.reward();
				memory.add(new Dump(state, reward, action, newState, stepResult.terminated() ? 1d : 0d));
				state = newState;
				cumReward += reward;
			}

			if (memory.size() < 2000) {
				continue;
			}

			List<Dump> samples = memory.sample(samplingSize);
			for (int ib = 0; ib < optimizationIteration; ib++) {
				optimize(samples.subList(ib * optimizationIteration, (ib + 1) *
						optimizationIteration));
				// optimizeAdam(samples.subList(ib * optimizationIteration, (ib + 1) *
				// optimizationIteration));
			}

			rewards.add(cumReward);
			System.out.println(
					"Run iteration %d rewards %f epsilon %f".formatted(
							rewards.size(), evaluate(cartPole), epsilon));
			if (rewards.size() % 100 == 0) {
				qTargetModel.load(qModel);
			}

			// rewards.add((double) totalReward);
			// System.err.println("ep %d, totalReward %d".formatted(i, totalReward));
			if (epsilon - epsilonDecay >= epsilonFinal) {
				epsilon -= epsilonDecay;
			}
			if (rewards.size() > 200 && rewards.subList(rewards.size() - 200, rewards.size())
					.stream().mapToDouble(Double::doubleValue).average().orElse(0) >= 495) {
				break;
			}
		}

		Matplot.print(rewards);
	}

	private void optimize(List<Dump> subList) {
		List<double[]> ins = subList.stream()
				.map(Dump::state)
				.toList();
		List<double[]> outs = subList.stream()
				.map(dump -> Arrays.stream(qTargetModel.predict(dump.newState()))
						.map(q -> dump.reward() + gamma * (1 - dump.done()) * q).toArray())
				.toList();
		List<int[]> oneHots = subList.stream()
				.map(dump -> {
					int[] action = new int[2];
					action[dump.action()] = 1;
					return action;
				}).toList();
		qModel.trainMse(ins, outs, oneHots);
	}
	//
	// private void optimizeSGD(List<Dump> subList) {
	// 	double[][] w1 = qModel.weights[0];
	// 	double[][] w2 = qModel.weights[1];
	// 	double[][] gradW1 = NeuralNetwork.init(w1);
	// 	double[][] gradW2 = NeuralNetwork.init(w2);
	// 	double[] hidden = qModel.o[1];
	// 	double[] gradHidden = new double[hidden.length];
	// 	double deltaTotal = 0;
	// 	for (Dump dump : subList) {
	//
	// 		double[] qValues = qModel.predict(dump.state());
	// 		double qValueSelected = qValues[dump.action()];
	//
	// 		double target = dump.reward()
	// 				+ gamma * (1 - dump.done()) * Arrays.stream(qTargetModel.predict(dump.newState())).max().orElse(0d);
	//
	// 		double delta = target - qValueSelected;
	// 		deltaTotal += delta;
	//
	// 		double[] hidden1 = qModel.o[1];
	// 		double[] gradHidden1 = new double[hidden1.length];
	//
	// 		for (int i = 0; i < gradW2.length; i++) {
	// 			gradW2[i][dump.action()] += delta * hidden1[i];
	// 		}
	//
	// 		for (int i = 0; i < gradHidden1.length; i++) {
	// 			gradHidden1[i] = delta * w2[i][dump.action()] * ActivationFonction.RELU.backward(hidden1[i]);
	// 			// gradHidden[i] += gradHidden1[i];
	// 		}
	//
	// 		for (int i = 0; i < gradW1.length; i++) {
	// 			for (int j1 = 0; j1 < gradW1[i].length; j1++) {
	// 				gradW1[i][j1] += dump.state()[i] * gradHidden1[j1];
	// 			}
	// 		}
	//
	// 	}
	//
	// 	for (int i = 0; i < gradW2.length; i++) {
	// 		for (int j = 0; j < gradW2[i].length; j++) {
	// 			w2[i][j] += alpha * gradW2[i][j] / subList.size();
	// 		}
	// 	}
	//
	// 	for (int i = 0; i < gradW1.length; i++) {
	// 		for (int j = 0; j < gradW1[i].length; j++) {
	// 			w1[i][j] += alpha * gradW1[i][j] / subList.size();
	// 		}
	// 	}
	// 	for (int i = 0; i < gradHidden.length; i++) {
	// 		qModel.biases[0][i] += alpha * gradHidden[i] / subList.size();
	// 	}
	// 	for (int i = 0; i < qModel.biases[1].length; i++) {
	// 		qModel.biases[1][i] += alpha * deltaTotal / subList.size();
	// 	}
	//
	// }

	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilonAdam = 1e-8;
	double weightDecay = 0.01;
	private double[][] mW1, vW1, mW2, vW2;
	private double[] mb1, vb1, mb2, vb2;
	private int t = 0;

	private void initAdam() {
		mW1 = NeuralNetwork.init(qModel.weights[0]);
		vW1 = NeuralNetwork.init(qModel.weights[0]);
		mW2 = NeuralNetwork.init(qModel.weights[1]);
		vW2 = NeuralNetwork.init(qModel.weights[1]);
		mb1 = new double[qModel.biases[0].length];
		vb1 = new double[qModel.biases[0].length];
		mb2 = new double[qModel.biases[1].length];
		vb2 = new double[qModel.biases[1].length];
	}

	private void optimizeAdam(List<Dump> subList) {
		t++;
		double[][] w1 = qModel.weights[0];
		double[][] w2 = qModel.weights[1];
		double[][] gradW1 = NeuralNetwork.init(w1);
		double[][] gradW2 = NeuralNetwork.init(w2);
		double[] gradB1 = new double[mb1.length];
		double[] gradB2 = new double[mb2.length];
		for (Dump dump : subList) {
			double[] qValues = qModel.predict(dump.state());
			double qValueSelected = qValues[dump.action()];
			double target = dump.reward()
					+ gamma * (1 - dump.done()) * Arrays.stream(qTargetModel.predict(dump.newState())).max().orElse(0d);
			double delta = target - qValueSelected;
			double[] hidden = qModel.o[1];
			double[] gradHidden = new double[hidden.length];

			for (int i = 0; i < gradW2.length; i++) {
				gradW2[i][dump.action()] = delta * hidden[i];
			}

			for (int i = 0; i < gradHidden.length; i++) {
				gradHidden[i] = delta * w2[i][dump.action()] * ActivationFonction.RELU.backward(hidden[i]);
			}

			for (int i = 0; i < gradW1.length; i++) {
				for (int j = 0; j < gradW1[i].length; j++) {
					gradW1[i][j] = dump.state()[i] * gradHidden[j];
				}
			}

			for (int i = 0; i < gradB1.length; i++) {
				gradB1[i] = gradHidden[i];
			}
			for (int i = 0; i < gradB2.length; i++) {
				gradB2[i] = delta;
			}

		}

		for (int i = 0; i < gradW1.length; i++) {
			for (int j = 0; j < gradW1[i].length; j++) {
				gradW1[i][j] = gradW1[i][j] / subList.size();
			}
		}
		for (int i = 0; i < gradW2.length; i++) {
			for (int j = 0; j < gradW2[i].length; j++) {
				gradW2[i][j] = gradW2[i][j] / subList.size();
			}
		}
		for (int i = 0; i < gradB1.length; i++) {
			gradB1[i] = gradB1[i] / subList.size();
		}
		for (int i = 0; i < gradB2.length; i++) {
			gradB2[i] = gradB2[i] / subList.size();
		}
		updateWeightsAdamW(w1, gradW1, mW1, vW1);
		updateWeightsAdamW(w2, gradW2, mW2, vW2);
		updateBiasesAdamW(qModel.biases[0], gradB1, mb1, vb1);
		updateBiasesAdamW(qModel.biases[1], gradB2, mb2, vb2);
	}

	private void updateWeightsAdamW(double[][] weights, double[][] grads, double[][] m, double[][] v) {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[i][j];
				v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j];
				double mHat = m[i][j] / (1 - Math.pow(beta1, t));
				double vHat = v[i][j] / (1 - Math.pow(beta2, t));
				weights[i][j] -= alpha * (mHat / (Math.sqrt(vHat) + epsilonAdam) + weightDecay * weights[i][j]);
			}
		}
	}

	private void updateBiasesAdamW(double[] biases, double[] grads, double[] m, double[] v) {
		for (int i = 0; i < biases.length; i++) {
			m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
			v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
			double mHat = m[i] / (1 - Math.pow(beta1, t));
			double vHat = v[i] / (1 - Math.pow(beta2, t));
			biases[i] -= alpha * (mHat / (Math.sqrt(vHat) + epsilonAdam));
		}
	}

	private double evaluate(CartPole cartPole) {

		double cumReward = 0;
		boolean done = false;
		double[] state = cartPole.reset();
		while (!done) {
			int action = pickSample(state, epsilon);
			StepResult step = cartPole.step(action);
			done = step.truncated() || step.terminated();
			cumReward += step.reward();
			state = step.state();
		}
		return cumReward;
	}

	record Dump(double[] state, double reward, int action, double[] newState, double done) {
	}
}

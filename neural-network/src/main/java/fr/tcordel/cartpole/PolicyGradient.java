package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.rl.neural.ActivationFonction;
import fr.tcordel.rl.neural.NeuralNetwork;
import fr.tcordel.rl.neural.WeightInitializor;
import fr.tcordel.utils.Matplot;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class PolicyGradient {
	double gamma = 0.99;
	double alpha = 0.001;
	int numEpisodes = 20000;
	int batchSize = 64;

	private final NeuralNetwork qModel;
	private final Random random = new Random();
	private final Memory memory = new Memory();

	public static void main(String[] args) {
		new PolicyGradient().train();

	}

	PolicyGradient() {
		qModel = new NeuralNetwork(WeightInitializor.RANDOM, 4, 64, 2);
		qModel.setActivationFonctions(ActivationFonction.RELU, ActivationFonction.NONE);
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

	double[] probs = new double[2];

	int pickSample(double[] state) {
		double[] predict = qModel.predict(state);
		double total;
		softmax(predict);
		double rand = random.nextDouble();
		total = 0;
		for (int i = 0; i < predict.length; i++) {
			total += probs[i];
			if (rand <= total) {
				return i;
			}
		}

		throw new IllegalStateException("rand not found %f for total %f".formatted(rand, total));
	}

	private void softmax(double[] predict) {
		double total = 0;
		for (int i = 0; i < predict.length; i++) {
			double exp = Math.exp(predict[i]);
			probs[i] = exp;
			total += exp;
		}
		for (int i = 0; i < predict.length; i++) {
			probs[i] = probs[i] / total;
		}
	}

	private void train() {
		CartPole cartPole = new CartPole();
		List<Double> rewards = new ArrayList<>();
		for (int i = 0; i < numEpisodes; i++) {
			memory.clear();

			double cumReward = 0;
			boolean done = false;
			double[] state = cartPole.reset();
			while (!done) {

				int action = pickSample(state);
				StepResult stepResult = cartPole.step(action);
				double[] newState = stepResult.state();
				done = stepResult.truncated() || stepResult.terminated();
				// memory
				double reward = stepResult.reward();
				memory.add(new Dump(state, reward, action, newState, stepResult.terminated() ? 1d : 0d));
				state = newState;
				cumReward += reward;
			}

			List<Dump> samples = memory.memories;
			for (int ib = samples.size() - 2; ib >= 0; ib--) {
				samples.get(ib).reward += samples.get(ib + 1).reward * gamma;

			}

			optimizeSGD(samples);
			// optimizeAdam(samples.subList(ib * optimizationIteration, (ib + 1) *
			// optimizationIteration));

			rewards.add(cumReward);
			System.out.println(
					"Run episode %d rewards %f ".formatted(
							rewards.size(), evaluate(cartPole)));

		}

		Matplot.print(rewards);
	}

	static double crossEntropy(double[] ins, double[] out) {

		return 0d;
	}

	private void optimizeSGD(List<Dump> subList) {
		double[][] w1 = qModel.weights[0];
		double[][] w2 = qModel.weights[1];
		double[][] gradW1 = NeuralNetwork.init(w1);
		double[][] gradW2 = NeuralNetwork.init(w2);
		double[] hidden = qModel.o[1];
		double[] gradHidden = new double[hidden.length];
		double deltaTotal = 0;
		for (Dump dump : subList) {

		double[] grad = new double[2];
			double[] logits = qModel.predict(dump.state);
			softmax(logits);
			double logProb = Math.log(probs[dump.action]);

			for (int i = 0; i < 2; i++) {
				grad[i] = (i == dump.action ? 1 : 0) - probs[i];
				grad[i] *= dump.reward * logProb;
			}

			double[] hidden1 = qModel.o[1];
			double[] gradHidden1 = new double[hidden1.length];

			for (int i = 0; i < gradW2.length; i++) {
				for (int j = 0; j < gradW2[i].length; j++) {
					gradW2[i][j] += grad[j] * hidden1[i];
				}
			}

			for (int i = 0; i < gradHidden1.length; i++) {
				for (int j = 0; j < grad.length; j++) {
					gradHidden1[i] += grad[j] * w2[i][j] * ActivationFonction.RELU.backward(hidden1[i]);
				}
			}

			for (int i = 0; i < gradW1.length; i++) {
				for (int j1 = 0; j1 < gradW1[i].length; j1++) {
					gradW1[i][j1] += dump.state[i] * gradHidden1[j1];
				}
			}

		}

		for (int i = 0; i < gradW2.length; i++) {
			for (int j = 0; j < gradW2[i].length; j++) {
				w2[i][j] += alpha * gradW2[i][j] / subList.size();
			}
		}

		for (int i = 0; i < gradW1.length; i++) {
			for (int j = 0; j < gradW1[i].length; j++) {
				w1[i][j] += alpha * gradW1[i][j] / subList.size();
			}
		}
		for (int i = 0; i < gradHidden.length; i++) {
			qModel.biases[0][i] += alpha * gradHidden[i] / subList.size();
		}
		for (int i = 0; i < qModel.biases[1].length; i++) {
			qModel.biases[1][i] += alpha * deltaTotal / subList.size();
		}

	}

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
			int action = pickSample(state);
			StepResult step = cartPole.step(action);
			done = step.truncated() || step.terminated();
			cumReward += step.reward();
			state = step.state();
		}
		return cumReward;
	}

	class Dump {
		double[] state;
		double reward;
		int action;
		double[] newState;
		double done;

		public Dump(double[] state, double reward, int action, double[] newState, double done) {
			this.state = state;
			this.reward = reward;
			this.action = action;
			this.newState = newState;
			this.done = done;
		}
	}
}

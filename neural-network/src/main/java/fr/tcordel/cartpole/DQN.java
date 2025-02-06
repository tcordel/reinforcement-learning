package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.rl.neural.ActivationFonction;
import fr.tcordel.rl.neural.NeuralNetwork;
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
	double epsilonDecay = epsilon / 3000;
	double epsilonFinal = 0.1;
	int numEpisodes = 15000;
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
		qModel = new NeuralNetwork(WeightInitializor.RANDOM, true, 4, 64, 2);
		qModel.setActivationFonctions(ActivationFonction.RELU, ActivationFonction.NONE);
		qTargetModel = new NeuralNetwork(qModel);
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
			// for (int j = 0; j < optimizationIteration; j++) {
			// optimize(samples);
			// }
			for (int ib = 0; ib < optimizationIteration; ib++) {
				optimize(samples.subList(ib * optimizationIteration, (ib + 1) * optimizationIteration));
			}

			rewards.add(cumReward);
			System.out.println(
					"Run iteration %d rewards %f epsilon %f".formatted(
							rewards.size(), evaluate(cartPole), epsilon));
			if (rewards.size() % 50 == 0) {
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
		double[][] w1 = qModel.weights[0];
		double[][] w2 = qModel.weights[1];
		double[][] gradW1 = NeuralNetwork.init(w1);
		double[][] gradW2 = NeuralNetwork.init(w2);
		double[] hidden = qModel.o[1];
		double[] gradHidden = new double[hidden.length];
		double deltaTotal = 0;
		for (Dump dump : subList) {

			double[] qValues = qModel.predict(dump.state());
			double qValueSelected = qValues[dump.action()];

			double target = dump.reward()
					+ gamma * (1 - dump.done()) * Arrays.stream(qTargetModel.predict(dump.newState())).max().orElse(0d);

			double delta = target - qValueSelected;
			deltaTotal += delta;

			double[] hidden1 = qModel.o[1];
			double[] gradHidden1 = new double[hidden1.length];

			for (int i3 = 0; i3 < gradW2.length; i3++) {
				gradW2[i3][dump.action()] += delta * hidden1[i3];
			}

			for (int i1 = 0; i1 < gradHidden1.length; i1++) {
				gradHidden1[i1] = delta * w2[i1][dump.action()] * ActivationFonction.RELU.backward(hidden1[i1]);
			}

			for (int i2 = 0; i2 < gradW1.length; i2++) {
				for (int j1 = 0; j1 < gradW1[i2].length; j1++) {
					gradW1[i2][j1] += dump.state()[i2] * gradHidden1[j1];
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
			qModel.thetas[0][i] += alpha * gradHidden[i] / subList.size();
		}
		for (int i = 0; i < qModel.thetas[1].length; i++) {
			qModel.thetas[1][i] += alpha * deltaTotal / subList.size();
		}

		// // compute Q(s_{t+1}) : size=[batchSize, 2]
		// List<double[]> targetVals =
		// qTargetModel.predict(subList.stream().map(Dump::newState).toList());
		// // compute max Q(s_{t+1}) : size=[batchSize]
		// List<int[]> oneHots = subList.stream()
		// .map(dump -> IntStream.range(0, 2).map(i -> i == dump.action() ? 1 :
		// 0).toArray())
		// .toList();
		// double[] maxTarget = targetVals.stream().mapToDouble(values ->
		// Arrays.stream(values).max().orElseThrow())
		// .toArray();
		// // compute r_t + gamma * (1 - d_t) * max Q(s_{t+1}) : size=[batchSize]
		// List<double[]> qVals1 = IntStream.range(0, subList.size())
		// .mapToObj(i -> {
		// Dump dump = subList.get(i);
		// double newQ = dump.reward() + gamma * (1 - dump.done()) * maxTarget[i];
		// return IntStream.range(0, 2).mapToDouble(index -> index == dump.action() ?
		// newQ : 0d).toArray();
		// }).toList();
		//
		// for (int i = 0; i < subList.size(); i++) {
		// double mse = 1;
		// while (mse > 0.1) {
		// double[] out = qVals1.get(i);
		// int[] oneHot = oneHots.get(i);
		// double mse1 = NeuralUtils.mse(qModel.predict(subList.get(i).state()), out,
		// oneHot);
		// double[] train = qModel.train(subList.get(i).state(), out, oneHot,
		// subList.size());
		// mse = NeuralUtils.mse(train, out, oneHot);
		// double x = mse1 - mse;
		// System.err.println(x);
		// }
		// }
		// double[] qVals2 = IntStream.range(0, subList.size())
		// .mapToDouble(i -> {
		// Dump dump = subList.get(i);
		// return qModel.predict(dump.state())[dump.action()];
		// }).toArray();

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

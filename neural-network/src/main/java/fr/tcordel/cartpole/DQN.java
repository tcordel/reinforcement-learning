package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.rl.neural.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DQN {
	double gamma = 0.99;
	double alpha = 0.1;
	double epsilon = 1;
	double epsilonDecay = epsilon / 3000;
	double epsilonFinal = 0.1;
	int numEpisodes = 15000;
	double samplingSize = 64 * 30;
	double batchSize = 64;

	private final NeuralNetwork qModel;
	private final NeuralNetwork qTargetModel;
	private final Random random = new Random();
	private final List<Memory> memories = new ArrayList<>(10000);

	public static void main(String[] args) {
		new DQN().train();

	}

	DQN() {
		qModel = new NeuralNetwork(4, 64, 2);
		qTargetModel = new NeuralNetwork(qModel);
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
				memories.add(new Memory(state, reward, action, newState));
				state = newState;
				cumReward += reward;
			}

			if (memories.size() < 2000) {
				continue;
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

	record Memory(double[] state, double reward, int action, double[] newState) {
	}
}

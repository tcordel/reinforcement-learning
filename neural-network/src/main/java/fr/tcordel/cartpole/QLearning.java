package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

public class QLearning {

	private static final int DISCRETISATION = 20;
	double gamma = 0.99;
	double alpha = 0.1;
	double epsilon = 1;
	double epsilonDecay = epsilon / 4000;
	int numEpisodes = 6000;
	private final Random random = new Random();
	Map<String, Map<Double, Double>> qTable = new HashMap<>();

	public static void main(String[] args) {
		QLearning qLearning = new QLearning();
		qLearning.train();
	}

	private void train() {
		CartPole cartPole = new CartPole();
		for (int i = 0; i < numEpisodes; i++) {
			String state = discretese(cartPole.reset());
			boolean done = false;
			int totalReward = 0;
			while (!done) {
				initQTable(state);
				int action = pickSample(state, epsilon);
				StepResult stepResult = cartPole.step(action);
				String newState = discretese(stepResult.state());
				initQTable(newState);
				done = stepResult.truncated() || stepResult.terminated();
				updateQTable(state, action, newState, stepResult.reward());
				totalReward += stepResult.reward();

				state = newState;
			}
			if (epsilon - epsilonDecay >= 0) {
				epsilon -= epsilonDecay;
			}
			System.err.println("ep %d, totalReward %d".formatted(i, totalReward));
		}
	}

	private void updateQTable(String state, double action, String newState, double reward) {
		double nextMaxQ = qTable.get(newState)
				.values()
				.stream()
				.mapToDouble(Double::doubleValue)
				.max()
				.orElse(0d);

		double newValue = qTable.get(state).get(action)
				+ alpha * (reward + gamma * nextMaxQ - qTable.get(state).get(action));
		qTable.get(state).put(action, newValue);

	}

	int pickSample(String state, double epsilon) {
		if (random.nextDouble() > epsilon) {
			return 0;
		} else {
			return random.nextInt(0, 2);
		}
	}

	private String discretese(double[] state) {
		int[] discretised = new int[4];
		discretised[0] = discretese(state[0], CartPole.X_THRESHOLD);
		discretised[1] = discretese(state[1], 4);
		discretised[2] = discretese(state[2], CartPole.THETA_THRESHOLD);
		discretised[3] = discretese(state[3], 4);
		return Arrays.stream(discretised)
				.mapToObj(String::valueOf)
				.collect(Collectors.joining(","));
	}

	private int discretese(double state, double boundary) {
		double window = 2 * boundary / DISCRETISATION;
		return (int) ((state + boundary) / window);
	}

	private void initQTable(String state) {
		if (!qTable.containsKey(state)) {
			HashMap<Double, Double> hashMap = new HashMap<>();
			hashMap.put(0d, 0d);
			hashMap.put(1d, 0d);
			qTable.put(state, hashMap);
		}
	}
}

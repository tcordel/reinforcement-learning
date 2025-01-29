package fr.tcordel.cartpole;

import java.util.ArrayList;
import java.util.List;

public class DQN {
	double gamma = 0.99;
	double alpha = 0.1;
	double epsilon = 1;
	double epsilonDecay = epsilon / 3000;
	double epsilonFinal = 0.1;
	int numEpisodes = 15000;
	double samplingSize = 64 * 30;
	double batchSize = 64;

	public static void main(String[] args) {
		new DQN().train();

	}

	private void train() {
		CartPole cartPole = new CartPole();
		List<Double> rewards = new ArrayList<>();
		for (int i = 0; i < numEpisodes; i++) {
			// rewards.add((double) totalReward);
			System.err.println("ep %d, totalReward %d".formatted(i, totalReward));
			if (epsilon - epsilonDecay >= epsilonFinal) {
				epsilon -= epsilonDecay;
			}
			if (rewards.size() > 200 && rewards.subList(rewards.size() - 200, rewards.size())
					.stream().mapToDouble(Double::doubleValue).average().orElse(0) >= 495) {
				break;
			}
		}
	}
}

package fr.tcordel.rl.openai;

import fr.tcordel.cartpole.CartPole;
import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.cartpole.CartPole.StepResult;

import java.util.*;

// Neural network implementation
class NeuralNetwork {
	private final int inputSize, hiddenSize, outputSize;
	private double[][] weights1, weights2;
	private double[] bias1, bias2;
	private final Random rand = new Random();

	public NeuralNetwork(int input, int hidden, int output) {
		this.inputSize = input;
		this.hiddenSize = hidden;
		this.outputSize = output;
		initializeWeights();
	}

	private void initializeWeights() {
		weights1 = new double[inputSize][hiddenSize];
		weights2 = new double[hiddenSize][outputSize];
		bias1 = new double[hiddenSize];
		bias2 = new double[outputSize];
		for (int i = 0; i < inputSize; i++)
			for (int j = 0; j < hiddenSize; j++)
				weights1[i][j] = rand.nextDouble() * 2 - 1;
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < outputSize; j++)
				weights2[i][j] = rand.nextDouble() * 2 - 1;
			bias1[i] = rand.nextDouble() * 2 - 1;
		}
		Arrays.fill(bias2, rand.nextDouble() * 2 - 1);
	}

	public double[] predict(double[] input) {
		double[] hidden = new double[hiddenSize];
		double[] output = new double[outputSize];

		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < inputSize; j++)
				hidden[i] += input[j] * weights1[j][i];
			hidden[i] = Math.max(0, hidden[i] + bias1[i]); // ReLU Activation
		}

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize; j++)
				output[i] += hidden[j] * weights2[j][i];
			output[i] += bias2[i];
		}
		return output;
	}

	public void train(double[] state, double target, int action, double alpha) {
		double[] hidden = new double[hiddenSize];
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < inputSize; j++)
				hidden[i] += state[j] * weights1[j][i];
			hidden[i] = Math.max(0, hidden[i] + bias1[i]);
		}
		double[] output = predict(state);
		double error = target - output[action];

		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < outputSize; j++) {
				weights2[i][j] += alpha * error * hidden[i];
			}
		}
		for (int i = 0; i < inputSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				weights1[i][j] += alpha * error * state[i];
			}
		}
	}
}

// DQN Agent with Experience Replay
class DQNAgent {
	private final NeuralNetwork model;
	private final List<double[]> memory;
	private final double gamma = 0.99;
	private final double alpha = 0.01;
	private final Random rand = new Random();

	public DQNAgent(int inputSize, int hiddenSize, int outputSize) {
		model = new NeuralNetwork(inputSize, hiddenSize, outputSize);
		memory = new ArrayList<>();
	}

	public int selectAction(double[] state, double epsilon) {
		if (rand.nextDouble() < epsilon)
			return rand.nextInt(2);
		double[] qValues = model.predict(state);
		return (qValues[0] > qValues[1]) ? 0 : 1;
	}

	public void store(double[] state, double reward, int action, double[] newState) {
		memory.add(new double[] { state[0], state[1], state[2], state[3], reward, action, newState[0], newState[1],
				newState[2], newState[3] });
		if (memory.size() > 10000)
			memory.remove(0);
	}

	public void train() {
		if (memory.size() < 64)
			return;
		for (int i = 0; i < 64; i++) {
			double[] sample = memory.get(rand.nextInt(memory.size()));
			double target = sample[4] + gamma * Arrays
					.stream(model.predict(new double[] { sample[6], sample[7], sample[8], sample[9] })).max().orElse(0);
			model.train(new double[] { sample[0], sample[1], sample[2], sample[3] }, target, (int) sample[5], alpha);
		}
	}
}

// Simulated CartPole environment

public class CartPoleSolver {
	public static void main(String[] args) {
		DQNAgent agent = new DQNAgent(4, 16, 2);
		CartPole env = new CartPole();
		int episodes = 10000;
		double epsilon = 1;
		double epsilonDecay = 0.995;

		for (int e = 0; e < episodes; e++) {
			double[] state = env.reset();
			double totalReward = 0;
			for (int t = 0; t < 500; t++) {
				int action = agent.selectAction(state, epsilon);
				StepResult step = env.step(action);
				agent.store(state, step.reward(), action,step.state());
				agent.train();
				state = step.state();
				totalReward += step.reward();
				if (step.truncated() || step.terminated()) {
					break;
				}
				if (totalReward >= 200)
					break;
			}
			epsilon *= epsilonDecay;
			System.out.println("Episode " + e + " Reward: " + totalReward);
		}
	}
}

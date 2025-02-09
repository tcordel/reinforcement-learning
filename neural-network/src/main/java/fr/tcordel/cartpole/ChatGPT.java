package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.utils.Matplot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ChatGPT {

	private double[][] weightsInputHidden; // Poids entre entrée et couche cachée
	private double[] biasesHidden; // Biais de la couche cachée

	private double[][] weightsHiddenOutput; // Poids entre couche cachée et sortie
	private double[] biasesOutput; // Biais de la couche de sortie

	private double learningRate;
	private final List<Dump> memory = new ArrayList<>();

	double gamma = 0.99d;
	private final Random random = new Random();

	public ChatGPT(int inputSize, int hiddenSize, int outputSize, double learningRate) {
		this.learningRate = learningRate;

		// Initialisation des poids et biais
		this.weightsInputHidden = new double[hiddenSize][inputSize];
		this.biasesHidden = new double[hiddenSize];

		this.weightsHiddenOutput = new double[outputSize][hiddenSize];
		this.biasesOutput = new double[outputSize];

		// Initialisation aléatoire des poids
		initializeWeights(weightsInputHidden);
		initializeWeights(weightsHiddenOutput);
	}

	private void initializeWeights(double[][] weights) {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = Math.random() * 0.01;
			}
		}
	}

	// ReLU activation function
	public static double[] relu(double[] x) {
		double[] output = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			output[i] = Math.max(0, x[i]); // ReLU
		}
		return output;
	}

	// Dérivée de ReLU pour backpropagation
	public static double[] reluDerivative(double[] x) {
		double[] output = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			output[i] = (x[i] > 0) ? 1 : 0; // ReLU'
		}
		return output;
	}

	// Softmax function
	public static double[] softmax(double[] logits) {
		double max = Arrays.stream(logits).max().orElse(0);
		double sum = 0.0;
		double[] expValues = new double[logits.length];

		for (int i = 0; i < logits.length; i++) {
			expValues[i] = Math.exp(logits[i] - max);
			sum += expValues[i];
		}

		for (int i = 0; i < logits.length; i++) {
			expValues[i] /= sum;
		}

		return expValues;
	}

	// Cross-entropy gradient
	public static double crossEntropyGradient(double[] logits, int target) {
		double[] probs = softmax(logits);
		// probs[target] -= 1; // Gradient softmax + cross-entropy
		return -Math.log(probs[target]);
	}

	// Forward pass
	public double[][] forward(double[] input) {
		double[] hiddenLayer = new double[biasesHidden.length];
		double[] outputLayer = new double[biasesOutput.length];

		// Couche cachée : Z = W*X + b, puis ReLU
		for (int i = 0; i < biasesHidden.length; i++) {
			hiddenLayer[i] = biasesHidden[i];
			for (int j = 0; j < input.length; j++) {
				hiddenLayer[i] += weightsInputHidden[i][j] * input[j];
			}
		}
		hiddenLayer = relu(hiddenLayer);

		// Couche de sortie : Z = W*H + b, puis softmax
		for (int i = 0; i < biasesOutput.length; i++) {
			outputLayer[i] = biasesOutput[i];
			for (int j = 0; j < hiddenLayer.length; j++) {
				outputLayer[i] += weightsHiddenOutput[i][j] * hiddenLayer[j];
			}
		}

		return new double[][] { hiddenLayer, outputLayer }; // Retourne valeurs cachées et logits
	}

	// Mise à jour des poids avec moyenne des gradients
	public void updateWeightsBatch(double[][] inputs, int[] targets, double[] rewards) {
		int batchSize = inputs.length;
		double[][] weightGradientsHidden = new double[weightsInputHidden.length][weightsInputHidden[0].length];
		double[] biasGradientsHidden = new double[biasesHidden.length];

		double[][] weightGradientsOutput = new double[weightsHiddenOutput.length][weightsHiddenOutput[0].length];
		double[] biasGradientsOutput = new double[biasesOutput.length];

		// Accumuler les gradients sur le batch
		for (int i = 0; i < batchSize; i++) {
			double[][] forwardPass = forward(inputs[i]);
			double[] hiddenLayer = forwardPass[0];
			double[] logits = forwardPass[1];
			double[] outputGradient = new double[2];

			// Calcul des gradients
			double logProbs = - crossEntropyGradient(logits, targets[i]);
			double loss = -logProbs * rewards[i];
			outputGradient[targets[i]] = loss;

			// Gradient couche cachée
			double[] hiddenGradient = new double[hiddenLayer.length];
			double[] reluDeriv = reluDerivative(hiddenLayer);

			for (int j = 0; j < hiddenGradient.length; j++) {
				for (int k = 0; k < 2; k++) {
					hiddenGradient[j] += outputGradient[k] * weightsHiddenOutput[k][j];
				}
				hiddenGradient[j] *= reluDeriv[j]; // Application de ReLU'
			}

			// Accumulation des gradients de sortie
			for (int j = 0; j < weightsHiddenOutput.length; j++) {
				for (int k = 0; k < weightsHiddenOutput[j].length; k++) {
					weightGradientsOutput[j][k] += outputGradient[j] * hiddenLayer[k];
				}
				biasGradientsOutput[j] += outputGradient[j];
			}

			// Accumulation des gradients de la couche cachée
			for (int j = 0; j < weightsInputHidden.length; j++) {
				for (int k = 0; k < weightsInputHidden[j].length; k++) {
					weightGradientsHidden[j][k] += hiddenGradient[j] * inputs[i][k];
				}
				biasGradientsHidden[j] += hiddenGradient[j];
			}
		}

		// Moyenne et mise à jour des poids
		for (int i = 0; i < weightsHiddenOutput.length; i++) {
			for (int j = 0; j < weightsHiddenOutput[i].length; j++) {
				weightsHiddenOutput[i][j] -= learningRate * (weightGradientsOutput[i][j] / batchSize);
			}
			biasesOutput[i] -= learningRate * (biasGradientsOutput[i] / batchSize);
		}

		for (int i = 0; i < weightsInputHidden.length; i++) {
			for (int j = 0; j < weightsInputHidden[i].length; j++) {
				weightsInputHidden[i][j] -= learningRate * (weightGradientsHidden[i][j] / batchSize);
			}
			biasesHidden[i] -= learningRate * (biasGradientsHidden[i] / batchSize);
		}
	}

	// Entraînement
	public void trainBatch(double[][] inputs, int[] targets) {
		// updateWeightsBatch(inputs, targets);
	}

	public static void main(String[] args) {
		ChatGPT nn = new ChatGPT(4, 64, 2, 0.001);
		nn.train();
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

	private void train() {
		CartPole cartPole = new CartPole();
		List<Double> history = new ArrayList<>();
		for (int i = 0; i < 20000; i++) {
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

			List<Dump> samples = memory;
			for (int ib = samples.size() - 2; ib >= 0; ib--) {
				samples.get(ib).reward += samples.get(ib + 1).reward * gamma;
			}

			double[][] inputs = samples.stream()
				.map(d -> d.state)
				.toArray(size -> new double[size][]);

			int[] targets = samples.stream()
				.mapToInt(d -> d.action)
				.toArray();

			double[] rewards = samples.stream()
				.mapToDouble(d -> d.reward)
				.toArray();
			// optimizeAdam(samples.subList(ib * optimizationIteration, (ib + 1) *
			// optimizationIteration));
			updateWeightsBatch(inputs, targets, rewards);

			history.add(cumReward);
			System.out.println(
					"Run episode %d rewards %f ".formatted(
							history.size(), evaluate(cartPole)));

		}

		Matplot.print(history);
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

	int pickSample(double[] state) {
		double[] predict = forward(state)[1];
		double total;
		double[] probs = softmax(predict);
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
}

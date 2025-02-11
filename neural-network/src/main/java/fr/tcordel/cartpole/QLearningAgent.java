package fr.tcordel.cartpole;

import fr.tcordel.cartpole.CartPole.StepResult;
import fr.tcordel.utils.Matplot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;



class AdamOptimizer {
	private double lr, beta1, beta2, eps;
	private int t;
	private Map<String, double[][]> params, m, v;

	public AdamOptimizer(Map<String, double[][]> params, double lr) {
		this.lr = lr;
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		this.eps = 1e-8;
		this.t = 0;
		this.params = params;

		// Initialisation des accumulateurs de moments (m et v)
		this.m = new HashMap<>();
		this.v = new HashMap<>();
		for (String key : params.keySet()) {
			m.put(key, new double[params.get(key).length][params.get(key)[0].length]);
			v.put(key, new double[params.get(key).length][params.get(key)[0].length]);
		}
	}

	public void step(Map<String, double[][]> grads) {
		t++; // Incrémentation du compteur de mises à jour

		for (String key : params.keySet()) {
			double[][] param = params.get(key);
			double[][] grad = grads.get(key);
			double[][] mHat = new double[param.length][param[0].length];
			double[][] vHat = new double[param.length][param[0].length];

			// Mise à jour des moments m et v
			for (int i = 0; i < param.length; i++) {
				for (int j = 0; j < param[0].length; j++) {
					// Mise à jour de la moyenne des gradients (moment de premier ordre)
					m.get(key)[i][j] = beta1 * m.get(key)[i][j] + (1 - beta1) * grad[i][j];
					// Mise à jour de la variance des gradients (moment de second ordre)
					v.get(key)[i][j] = beta2 * v.get(key)[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];

					// Biais corrigé pour m et v
					mHat[i][j] = m.get(key)[i][j] / (1 - Math.pow(beta1, t));
					vHat[i][j] = v.get(key)[i][j] / (1 - Math.pow(beta2, t));

					// Mise à jour des paramètres avec Adam
					param[i][j] -= lr * mHat[i][j] / (Math.sqrt(vHat[i][j]) + eps);
				}
			}
		}
	}
}

class QNet {
	int inputDim, hiddenDim, outputDim;
	double[][] weights1, weights2, bias1, bias2;
	AdamOptimizer optimizer;

	public QNet(int inputDim, int hiddenDim, int outputDim, double lr) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;

		// Initialisation des poids avec Glorot/Xavier Normalization
		this.weights1 = randomMatrix(inputDim, hiddenDim, Math.sqrt(2.0 / inputDim));
		this.bias1 = new double[1][hiddenDim]; // (1, 64)
		this.weights2 = randomMatrix(hiddenDim, outputDim, Math.sqrt(2.0 / hiddenDim));
		this.bias2 = new double[1][outputDim]; // (1, 2)

		// Optimiseur Adam
		this.optimizer = new AdamOptimizer(getParams(), lr);
	}

	public double[][] forward(double[][] stateBatch) {
		// Calcul de la couche cachée : ReLU(W1 * state + b1)
		double[][] hidden = relu(
				matrixAdd(matrixDot(stateBatch, weights1, stateBatch.length, hiddenDim, inputDim), bias1));

		// Calcul des valeurs Q : W2 * hidden + b2
		double[][] output = matrixAdd(matrixDot(hidden, weights2, hidden.length, outputDim, hiddenDim), bias2);

		return output; // (batchSize, outputDim)
	}

	public Map<String, double[][]> getParams() {
		return Map.of("w1", weights1, "b1", bias1, "w2", weights2, "b2", bias2);
	}

	public void update(Map<String, double[][]> grads) {
		optimizer.step(grads);
	}

	public void copyFrom(QNet other) {
		this.weights1 = deepCopy(other.weights1);
		this.bias1 = deepCopy(other.bias1);
		this.weights2 = deepCopy(other.weights2);
		this.bias2 = deepCopy(other.bias2);
	}

	// Fonction pour créer une matrice aléatoire avec distribution normalisée
	private double[][] randomMatrix(int rows, int cols, double scale) {
		Random rand = new Random();
		double[][] matrix = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = rand.nextGaussian() * scale;
			}
		}
		return matrix;
	}

	// Fonction pour effectuer un produit matriciel : A (m x n) * B (n x p) = C (m x
	// p)
	private double[][] matrixDot(double[][] a, double[][] b, int rowsA, int colsB, int commonDim) {
		double[][] result = new double[rowsA][colsB];
		for (int i = 0; i < rowsA; i++) {
			for (int j = 0; j < colsB; j++) {
				for (int k = 0; k < commonDim; k++) {
					result[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return result;
	}

	// Ajout d'un biais à chaque ligne d'une matrice
	static double[][] matrixAdd(double[][] mat, double[][] bias) {
		double[][] result = new double[mat.length][mat[0].length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				result[i][j] = mat[i][j] + bias[0][j]; // Ajouter la même ligne de biais à chaque ligne
			}
		}
		return result;
	}

	// Appliquer la fonction ReLU
	private double[][] relu(double[][] matrix) {
		for (double[] row : matrix) {
			for (int i = 0; i < row.length; i++) {
				row[i] = Math.max(0, row[i]);
			}
		}
		return matrix;
	}

	// Deep copy pour éviter la modification des références
	private double[][] deepCopy(double[][] matrix) {
		double[][] copy = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			System.arraycopy(matrix[i], 0, copy[i], 0, matrix[i].length);
		}
		return copy;
	}
}

class ReplayMemory {
	private final int bufferSize;
	private final LinkedList<double[][]> buffer;

	public ReplayMemory(int bufferSize) {
		this.bufferSize = bufferSize;
		this.buffer = new LinkedList<>();
	}

	public void add(double[] state, int action, double reward, double[] nextState, boolean done) {
		if (buffer.size() == bufferSize) {
			buffer.poll();
		}
		buffer.add(new double[][] { state, { action }, { reward }, nextState, { done ? 1.0 : 0.0 } });
	}

	public List<double[][]> sample(int sampleSize) {
		Random rand = new Random();
		List<double[][]> sample = new ArrayList<>();
		for (int i = 0; i < sampleSize; i++) {
			sample.add(buffer.get(rand.nextInt(buffer.size())));
		}
		return sample;
	}

	public int length() {
		return buffer.size();
	}
}

public class QLearningAgent {
	private static final double GAMMA = 0.99;
	private static final int BATCH_SIZE = 64;
	private static final int SAMPLING_SIZE = BATCH_SIZE * 30;
	private static final double EPSILON_START = 1.0;
	private static final double EPSILON_DECAY = EPSILON_START / 3000;
	private static final double EPSILON_FINAL = 0.1;

	private QNet qModel;
	private QNet qTargetModel;
	private ReplayMemory memory;
	private double epsilon;

	public QLearningAgent() {
		qModel = new QNet(4, 64, 2, 0.0005);
		qTargetModel = new QNet(4, 64, 2, 0.0005);
		qTargetModel.copyFrom(qModel);
		memory = new ReplayMemory(10000);
		epsilon = EPSILON_START;
	}
	public static void main(String[] args) {
		new QLearningAgent().train(new CartPole(), 15000);

	}

	private Map<String, double[][]> computeGradients(QNet model, List<double[][]> batch) {
		int batchSize = batch.size();
		double[][] states = new double[batchSize][4];
		int[] actions = new int[batchSize];
		double[] rewards = new double[batchSize];
		double[][] nextStates = new double[batchSize][4];
		double[] dones = new double[batchSize];

		for (int i = 0; i < batchSize; i++) {
			states[i] = batch.get(i)[0];
			actions[i] = (int) batch.get(i)[1][0];
			rewards[i] = batch.get(i)[2][0];
			nextStates[i] = batch.get(i)[3];
			dones[i] = batch.get(i)[4][0];
		}

		double[][] hidden = relu(QNet.matrixAdd(matrixDot(states, model.weights1, batchSize, 64, 4), model.bias1));
		double[][] qVals = QNet.matrixAdd(matrixDot(hidden, model.weights2, batchSize, 2, 64), model.bias2);

		double[][] nextHidden = relu(
				QNet.matrixAdd(matrixDot(nextStates, model.weights1, batchSize, 64, 4), model.bias1));
		double[][] nextQVals = QNet.matrixAdd(matrixDot(nextHidden, model.weights2, batchSize, 2, 64), model.bias2);

		double[] targetQVals = new double[batchSize];
		for (int i = 0; i < batchSize; i++) {
			targetQVals[i] = rewards[i] + GAMMA * (1 - dones[i]) * Arrays.stream(nextQVals[i]).max().getAsDouble();
		}

		double[][] lossGrad = new double[batchSize][2];
		for (int i = 0; i < batchSize; i++) {
			lossGrad[i][actions[i]] = 2 * (qVals[i][actions[i]] - targetQVals[i]) / batchSize;
		}

		double[][] gradW2 = matrixDot(transpose(hidden, batchSize, 64), lossGrad, 64, 2, batchSize);
		double[] gradB2 = sumRows(lossGrad);

		double[][] hiddenGrad = matrixDot(lossGrad, transpose(model.weights2, 64, 2), batchSize, 64, 2);
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < 64; j++) {
				hiddenGrad[i][j] *= (hidden[i][j] > 0 ? 1 : 0);
			}
		}

		double[][] gradW1 = matrixDot(transpose(states, batchSize, 4), hiddenGrad, 4, 64, batchSize);
		double[] gradB1 = sumRows(hiddenGrad);

		return Map.of("w1", gradW1, "b1", new double[][] { gradB1 }, "w2", gradW2, "b2", new double[][] { gradB2 });
	}

	private void optimize() {
		List<double[][]> batch = memory.sample(SAMPLING_SIZE);
		Map<String, double[][]> grads = computeGradients(qModel, batch);
		qModel.update(grads);
	}

	private int pickAction(double[] state) {
		if (Math.random() > epsilon) {
			double[] qValues = qModel.forward(new double[][] {state})[0];
			return maxIndex(qValues);
		} else {
			return new Random().nextInt(2);
		}
	}

	public void train(CartPole env, int iterations) {
		List<Integer> rewardRecords = new ArrayList<>();

		for (int iter = 0; iter < iterations; iter++) {
			double[] state = env.reset();
			int totalReward = 0;

			for (int step = 0; step < 500; step++) {
				int action = pickAction(state);
				double[] nextState;
				double reward;
				boolean terminal;

				// Simulation step
				StepResult result = env.step(action);
				nextState = Arrays.copyOfRange(result.state(), 0, 4);
				reward = result.reward();
				terminal = result.truncated() || result.terminated();
				memory.add(state, action, reward, nextState, terminal);
				totalReward += reward;
				state = nextState;

				if (terminal)
					break;
			}

			if (memory.length() >= 2000) {
				optimize();
			}

			rewardRecords.add(totalReward);
			System.out.printf("Iteration %d - Reward: %d - Epsilon: %.5f%n", iter + 1, totalReward, epsilon);

			if ((iter + 1) % 50 == 0) {
				qTargetModel.copyFrom(qModel);
			}

			if (epsilon > EPSILON_FINAL) {
				epsilon -= EPSILON_DECAY;
			}
			if (rewardRecords.size() > 200 && rewardRecords.subList(rewardRecords.size() - 200, rewardRecords.size())
					.stream().mapToDouble(Integer::doubleValue).average().orElse(0) >= 495) {
				break;
			}
		}

		Matplot.print(rewardRecords.stream().mapToDouble(Integer::doubleValue).boxed().toList());
	}

	private int maxIndex(double[] array) {
		int maxIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > array[maxIndex])
				maxIndex = i;
		}
		return maxIndex;
	}

	private double[][] matrixDot(double[][] a, double[][] b, int rowsA, int colsB, int commonDim) {
		double[][] result = new double[rowsA][colsB];
		for (int i = 0; i < rowsA; i++) {
			for (int j = 0; j < colsB; j++) {
				for (int k = 0; k < commonDim; k++) {
					result[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return result;
	}

	private double[][] transpose(double[][] matrix, int rows, int cols) {
		double[][] result = new double[cols][rows];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[j][i] = matrix[i][j];
			}
		}
		return result;
	}

	private double[][] relu(double[][] matrix) {
		for (double[] row : matrix) {
			for (int i = 0; i < row.length; i++) {
				row[i] = Math.max(0, row[i]);
			}
		}
		return matrix;
	}

	private double[] sumRows(double[][] matrix) {
		double[] result = new double[matrix[0].length];
		for (double[] row : matrix) {
			for (int i = 0; i < row.length; i++) {
				result[i] += row[i];
			}
		}
		return result;
	}
}

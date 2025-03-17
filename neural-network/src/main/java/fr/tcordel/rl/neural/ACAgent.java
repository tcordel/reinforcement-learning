package fr.tcordel.rl.neural;

import fr.tcordel.game.Result;
import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.dummy.RandomAgent;
import fr.tcordel.rl.qlearning.QAgent;
import fr.tcordel.utils.Matplot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.function.BiFunction;

public class ACAgent {
	private static final int STATE_SIZE = 18;
	private static final int OUTPUT_SIZE = 9;
	private static final double GAMMA = 0.99;
	private final Random random = new Random();

	int colsB = 64;
	private QNet actor;
	private QNet critic;
	private ReplayMemory memory;

	public ACAgent() {
		actor = new QNet(STATE_SIZE, colsB, OUTPUT_SIZE, 0.001);
		critic = new QNet(STATE_SIZE, colsB, OUTPUT_SIZE, 0.001);
		memory = new ReplayMemory(10000);
	}

	public static void main(String[] args) {
		TicTacToe game = new TicTacToe();
		ACAgent agent = new ACAgent();
		agent.train(game, 3000);
		// agent.train(game, 1);

		Scanner in = new Scanner(System.in);

		while (true) {
			System.err.println("new game");
			game.resetBoard();
			Result result = game.getResult();

			char player = TicTacToe.O;

			while (Result.PENDING.equals(result)) {
				if (player == TicTacToe.O) {
					int action = agent.pickAction(game, game.boardToState());
					System.out.println("IA : %d %d".formatted(action / 3, action % 3));
					game.play(TicTacToe.O, action / 3, action % 3);
				} else {
					System.out.println("Choose action :");
					System.out.println(game.toString());
					String action = in.nextLine();
					game.play(player,
							Integer.parseInt(String.valueOf(action.charAt(0))),
							Integer.parseInt(String.valueOf(action.charAt(1))));
				}

				player = player == TicTacToe.X ? TicTacToe.O : TicTacToe.X;
				result = game.getResult();
			}
			System.out.println("win " + result);
		}

	}

	private Map<String, double[][]> computeCriticGradients(QNet model, List<double[][]> batch) {
		int batchSize = batch.size();
		double[][] states = new double[batchSize][STATE_SIZE];
		int[] actions = new int[batchSize];
		double[] rewards = new double[batchSize];
		double[][] nextStates = new double[batchSize][STATE_SIZE];
		double[] dones = new double[batchSize];

		for (int i = 0; i < batchSize; i++) {
			states[i] = batch.get(i)[0];
			actions[i] = (int) batch.get(i)[1][0];
			rewards[i] = batch.get(i)[2][0];
			nextStates[i] = batch.get(i)[3];
			dones[i] = batch.get(i)[4][0];
		}

		double[][] hidden = relu(
				QNet.matrixAddForBias(matrixDot(states, model.weights1, batchSize, colsB, STATE_SIZE), model.bias1));
		double[][] qVals = QNet.matrixAddForBias(matrixDot(hidden, model.weights2, batchSize, OUTPUT_SIZE, colsB),
				model.bias2);

		double[] targetQVals = new double[batchSize];
		for (int i = 0; i < batchSize; i++) {
			targetQVals[i] = rewards[i];
		}

		double[][] lossGrad = new double[batchSize][OUTPUT_SIZE];
		for (int i = 0; i < batchSize; i++) {
			lossGrad[i][actions[i]] = 2 * (qVals[i][actions[i]] - targetQVals[i]) / batchSize;
		}

		double[][] gradW2 = matrixDot(transpose(hidden, batchSize, colsB), lossGrad, colsB, OUTPUT_SIZE, batchSize);
		double[] gradB2 = sumRows(lossGrad);

		double[][] hiddenGrad = matrixDot(lossGrad, transpose(model.weights2, colsB, OUTPUT_SIZE), batchSize, colsB,
				OUTPUT_SIZE);
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < colsB; j++) {
				hiddenGrad[i][j] *= (hidden[i][j] > 0 ? 1 : 0);
			}
		}

		double[][] gradW1 = matrixDot(transpose(states, batchSize, STATE_SIZE), hiddenGrad, STATE_SIZE, colsB,
				batchSize);
		double[] gradB1 = sumRows(hiddenGrad);

		return Map.of("w1", gradW1, "b1", new double[][] { gradB1 }, "w2", gradW2, "b2", new double[][] { gradB2 });
	}

	private Map<String, double[][]> computeActorGradients(QNet model, List<double[][]> batch) {
		int batchSize = batch.size();
		double[][] states = new double[batchSize][4];
		int[] actions = new int[batchSize];
		double[][] rewards = new double[batchSize][];
		double[][] nextStates = new double[batchSize][4];
		double[] dones = new double[batchSize];

		for (int i = 0; i < batchSize; i++) {
			states[i] = batch.get(i)[0];
			actions[i] = (int) batch.get(i)[1][0];
			rewards[i] = new double[OUTPUT_SIZE];
			Arrays.fill(rewards[i], 0, OUTPUT_SIZE, batch.get(i)[2][0]);
			nextStates[i] = batch.get(i)[3];
			dones[i] = batch.get(i)[4][0];
		}

		double[][] hidden = relu(
				QNet.matrixAddForBias(matrixDot(states, model.weights1, batchSize, colsB, STATE_SIZE), model.bias1));
		double[][] logits = QNet.matrixAddForBias(matrixDot(hidden, model.weights2, batchSize, OUTPUT_SIZE, colsB),
				model.bias2);
		double[][] probs = softmax(logits);

		double[][] valuesHidden = relu(
				QNet.matrixAddForBias(matrixDot(states, critic.weights1, batchSize, colsB, STATE_SIZE), critic.bias1));
		double[][] values = QNet.matrixAddForBias(
				matrixDot(valuesHidden, critic.weights2, batchSize, OUTPUT_SIZE, colsB),
				critic.bias2);

		double[][] advantages = QNet.matrixOperation(rewards, values, (r, v) -> r - v);

		double[][] dLogit = new double[batchSize][OUTPUT_SIZE];
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < OUTPUT_SIZE; j++) {
				dLogit[i][j] = probs[i][j];
				if (actions[i] == j) {
					dLogit[i][j] -= 1;
				}
				dLogit[i][j] *= advantages[i][j];
			}
		}

		double[][] gradW2 = matrixDot(transpose(hidden, batchSize, colsB), dLogit, colsB, OUTPUT_SIZE, batchSize);
		double[] gradB2 = sumRows(dLogit);

		double[][] hiddenGrad = matrixDot(dLogit, transpose(model.weights2, colsB, OUTPUT_SIZE), batchSize, colsB,
				OUTPUT_SIZE);
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < colsB; j++) {
				hiddenGrad[i][j] *= (hidden[i][j] > 0 ? 1 : 0);
			}
		}

		double[][] gradW1 = matrixDot(transpose(states, batchSize, STATE_SIZE), hiddenGrad, STATE_SIZE, colsB,
				batchSize);
		double[] gradB1 = sumRows(hiddenGrad);

		return Map.of("w1", gradW1, "b1", new double[][] { gradB1 }, "w2", gradW2, "b2", new double[][] { gradB2 });
	}

	public static double[] crossEntropyLoss(double[][] logits, int[] targets) {
		int batchSize = logits.length;
		int numClasses = logits[0].length;

		// Appliquer log-softmax
		double[][] logSoftmax = logSoftmax(logits, batchSize, numClasses);

		// Calcul de la perte NLL pour chaque exemple du batch
		double[] losses = new double[batchSize];
		for (int i = 0; i < batchSize; i++) {
			losses[i] = -logSoftmax[i][targets[i]]; // Prendre la valeur du log-softmax pour la classe cible
		}

		return losses; // Retourner un tableau avec la perte de chaque échantillon
	}

	private static double[][] logSoftmax(double[][] logits, int batchSize, int numClasses) {
		double[][] logSoftmax = new double[batchSize][numClasses];

		for (int i = 0; i < batchSize; i++) {
			double maxLogit = Arrays.stream(logits[i]).max().getAsDouble(); // Stabilité numérique

			// Exponentielle normalisée
			double sumExp = 0.0;
			for (int j = 0; j < numClasses; j++) {
				sumExp += Math.exp(logits[i][j] - maxLogit);
			}

			double logSumExp = Math.log(sumExp);
			for (int j = 0; j < numClasses; j++) {
				logSoftmax[i][j] = logits[i][j] - maxLogit - logSumExp;
			}
		}

		return logSoftmax;
	}

	private void optimize() {
		List<double[][]> frames = memory.getAll();
		for (int ib = frames.size() - 2; ib >= 0; ib--) {
			frames.get(ib)[2][0] += frames.get(ib + 1)[2][0] * GAMMA;
		}
		Map<String, double[][]> criticGrads = computeCriticGradients(critic, frames);
		critic.update(criticGrads);
		Map<String, double[][]> actorGrads = computeActorGradients(actor, frames);
		actor.update(actorGrads);
	}

	private int pickAction(TicTacToe game, double[] state) {
		double[] predict = actor.forward(new double[][] { state })[0];
		double[] actionsHotspot = game.getActionsHotspot();
		// for (int i = 0; i < predict.length; i++) {
		// predict[i] *= actionsHotspot[i];
		// }
		double total;
		double[] probs = softmax(predict, actionsHotspot);
		double rand = random.nextDouble();
		total = 0;
		for (int i = 0; i < predict.length; i++) {
			total += probs[i];
			if (rand <= total) {
				return i;
			}
		}
		double[] probs2 = softmax(predict, actionsHotspot);

		throw new IllegalStateException("rand not found %f for total %f".formatted(rand, total));
	}

	public static double[][] softmax(double[][] logits) {
		double[][] softmax = new double[logits.length][];
		for (int i = 0; i < softmax.length; i++) {
			softmax[i] = softmax(logits[i]);
		}
		return softmax;
	}

	public static double[] softmax(double[] logits, double[] hots) {
		double[] probs = new double[logits.length];
		double total = 0;
		for (int i = 0; i < logits.length; i++) {
			if (hots[i] == 1) {
				double exp = Math.exp(logits[i]);
				probs[i] = exp;
				total += exp;
			}
		}
		for (int i = 0; i < logits.length; i++) {
			if (hots[i] == 1) {
				probs[i] = probs[i] / total;
			}
		}
		return probs;
	}

	public static double[] softmax(double[] logits) {
		double[] probs = new double[logits.length];
		double total = 0;
		for (int i = 0; i < logits.length; i++) {
			double exp = Math.exp(logits[i]);
			probs[i] = exp;
			total += exp;
		}
		for (int i = 0; i < logits.length; i++) {
			probs[i] = probs[i] / total;
		}
		return probs;
	}

	public void train(TicTacToe env, int iterations) {
		List<Integer> rewardRecords = new ArrayList<>();
		RandomAgent randomAgent = new RandomAgent(env, TicTacToe.X);

		char player = TicTacToe.X;
		for (int iter = 0; iter < iterations; iter++) {
			double[] state = env.reset();
			player = TicTacToe.X;
			boolean dead = false;
			int totalReward = 0;
			memory.reset();

			while (!dead) {
				Result currentResult;
				if (player == TicTacToe.X) {
					randomAgent.play(0d);
					currentResult = env.getResult();
					dead = !currentResult.equals(Result.PENDING);
					if (dead) {
						double reward = currentResult.equals(Result.O) ? 1
								: currentResult.equals(Result.X) ? -1 : currentResult.equals(Result.DROW) ? 0.3d : 0;
						memory.updateLastReward(reward);
					}
				} else {
					int action = pickAction(env, state);
					double[] nextState;
					double reward;

					// Simulation step
					env.play(TicTacToe.O, action / 3, action % 3);
					nextState = env.boardToState();
					currentResult = env.getResult();
					dead = !currentResult.equals(Result.PENDING);
					reward = currentResult.equals(Result.O) ? 1
							: currentResult.equals(Result.X) ? -1 : currentResult.equals(Result.DROW) ? 0.3d : 0;
					memory.add(state, action, reward, nextState, dead);
					totalReward += reward;
					state = nextState;
				}
				player = player == TicTacToe.X ? TicTacToe.O : TicTacToe.X;
			}

			optimize();

			rewardRecords.add(totalReward);
			System.out.printf("Iteration %d - Reward: %d%n", iter + 1, totalReward);

			if (rewardRecords.size() > 200 && rewardRecords.subList(rewardRecords.size() - 200, rewardRecords.size())
					.stream().mapToDouble(Integer::doubleValue).average().orElse(0) >= 495) {
				break;
			}
		}

		// Matplot.print(rewardRecords.stream().mapToDouble(Integer::doubleValue).boxed().toList());
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
				for (int j = 0; j < param[i].length; j++) {
					// Mise à jour de la moyenne des gradients (moment de premier ordre)
					double d = grad[i][j];
					m.get(key)[i][j] = beta1
							* m.get(key)[i][j]
							+ (1 - beta1) * d;
					// Mise à jour de la variance des gradients (moment de second ordre)
					v.get(key)[i][j] = beta2 * v.get(key)[i][j] + (1 - beta2) * d * d;

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
				matrixAddForBias(matrixDot(stateBatch, weights1, stateBatch.length, hiddenDim, inputDim), bias1));

		// Calcul des valeurs Q : W2 * hidden + b2
		double[][] output = matrixAddForBias(matrixDot(hidden, weights2, hidden.length, outputDim, hiddenDim), bias2);

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

	static double[][] matrixOperation(double[][] mat, double[][] bias, BiFunction<Double, Double, Double> operator) {
		double[][] result = new double[mat.length][mat[0].length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				result[i][j] = operator.apply(mat[i][j], bias[i][j]);
			}
		}
		return result;
	}

	// Ajout d'un biais à chaque ligne d'une matrice
	static double[][] matrixAddForBias(double[][] mat, double[][] bias) {
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

	public void updateLastReward(double reward) {
		buffer.getLast()[2][0] = reward;
	}

	public List<double[][]> getAll() {
		return buffer;
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

	public void reset() {
		buffer.clear();
	}
}

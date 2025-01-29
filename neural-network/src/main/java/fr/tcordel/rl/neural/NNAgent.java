package fr.tcordel.rl.neural;

import fr.tcordel.game.Move;
import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.Agent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NNAgent implements Agent {

	private final TicTacToe game;
	private final char player;
	private final NeuralNetwork neuralNetwork;
	List<Move> moves = new ArrayList<>();
	private double[] ins;

	public NNAgent(TicTacToe game, char player) {
		this.game = game;
		this.player = player;
		this.neuralNetwork = new NeuralNetwork(18, 200, 200, 9);
		ins = new double[18];
	}

	@Override
	public void play(double explorationRate) {
		String serialize = game.serialize();
		char[] charArray = serialize.toCharArray();
		Arrays.fill(ins, 0);
		for (int i = 0; i < charArray.length; i++) {
			if (charArray[i] == player) {
				ins[i] = 1;
			} else if (charArray[i] != TicTacToe.EMPTY_CELL) {
				ins[i + 9] = 1;
			}
		}
		double[] predict = neuralNetwork.predict(ins);
		int maxIndex = -1;
		double maxPredict = -1d;
		for (int i = 0; i < predict.length; i++) {
			if (predict[i] > maxPredict && charArray[i] == TicTacToe.EMPTY_CELL) {
				maxIndex = i;
				maxPredict = predict[i];
			}
		}
		
		int x = maxIndex / 3;
		int y = maxIndex % 3;
		game.play(player, x, y);
		moves.add(new Move(serialize, "%d%d".formatted(x, y)));
	}

	@Override
	public void updateStrategy(double reward) {


	}

	@Override
	public void reset() {
		moves.clear();
	}

}

package fr.tcordel.rl.qlearning;

import fr.tcordel.game.Result;
import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.Agent;
import fr.tcordel.rl.dummy.RandomAgent;

public class QLearning {

	private final TicTacToe game;

	int numEpisodes = 10000;
	private double explorationRate = 0.5; // means 50%

	private final Agent playerX;
	private final Agent playerO;

	public QLearning(TicTacToe game) {
		this.game = game;
		this.playerX = new RandomAgent(game, TicTacToe.X);

		this.playerO = new QAgent(game, TicTacToe.O);
	}

	void learn() {
		int winO = 0;
		int winX = 0;
		for (int episode = 0; episode < numEpisodes; episode++) {

			game.resetBoard();
			playerO.reset();
			playerX.reset();

			System.out.println("Learning %d".formatted(episode));
			Result result = Result.PENDING;
			Agent player = playerX;

			while (Result.PENDING.equals(result)) {
				player.play(explorationRate);
				result = game.getResult();
				player = player == playerO ? playerX : playerO;
			}

			switch (result) {
				case O:
				winO ++;
					playerO.updateStrategy(1d);
					playerX.updateStrategy(-1d);
					break;
				case X:
				winX ++;
					playerO.updateStrategy(-1d);
					playerX.updateStrategy(1d);
					break;
				default:
					playerO.updateStrategy(0.3d);
					playerX.updateStrategy(0.1d);
					break;
			}

			explorationRate *= 0.99;
		}
		System.out.println("training result : " + winO + " vs " + winX);
		game.resetBoard();
	}

	public Agent getPlayerO() {
		return playerO;
	}

}

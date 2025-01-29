package fr.tcordel.rl.dummy;

import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.Agent;

import java.util.List;
import java.util.Random;

public class RandomAgent implements Agent {
	private final TicTacToe game;
	private final char player;
	private final Random random = new Random();

	public RandomAgent(TicTacToe game, char player) {
		this.game = game;
		this.player = player;
	}

	@Override
	public void play(double explorationRate) {
		List<String> actions = game.getAllAvailableActions();
		String act = actions.get(random.nextInt(actions.size()));
		game.play(player,
				Integer.parseInt(String.valueOf(act.charAt(0))),
				Integer.parseInt(String.valueOf(act.charAt(1))));
	}

	@Override
	public void updateStrategy(double reward) {
	}

}

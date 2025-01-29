package fr.tcordel.rl.qlearning;

import fr.tcordel.game.Result;
import fr.tcordel.game.TicTacToe;

import java.util.Map;
import java.util.Scanner;

public class QlearningLauncher {

	public static void main(String[] args) {
		TicTacToe game = new TicTacToe();
		QLearning qLearning = new QLearning(game);
		qLearning.learn();

		Scanner in = new Scanner(System.in);

		QAgent playerO = (QAgent) qLearning.getPlayerO();
		while (true) {
			System.err.println("new game");
			game.resetBoard();
			playerO.reset();
			Result result = game.getResult();

			char player = TicTacToe.X;

			while (Result.PENDING.equals(result)) {
				String action;
				if (player == TicTacToe.O) {
					Map<String,Double> map = playerO.qTable.get(game.serialize());
					if (map != null) {
						map.entrySet()
							.stream()
							.forEach(e -> System.err.println("%s -> %f".formatted(e.getKey(), e.getValue())));
					}
					playerO.play(0);
				} else {
					System.out.println("Choose action :");
					System.out.println(game.toString());
					action = in.nextLine();
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
}

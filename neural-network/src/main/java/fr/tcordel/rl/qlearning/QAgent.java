package fr.tcordel.rl.qlearning;

import fr.tcordel.game.Move;
import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.Agent;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

public class QAgent implements Agent {

	private final Random random = new Random();

	private final double learningRate = 0.9;
	private final double discountFactor = 0.9;
	private final TicTacToe game;
	private final char player;

	Map<String, Map<String, Double>> qTable = new HashMap<>();
	List<Move> moves = new ArrayList<>();

	public QAgent(TicTacToe game, char player) {
		this.game = game;
		this.player = player;
	}

	@Override
	public void reset() {
		moves.clear();
	}

	@Override
	public void play(double explorationRate) {
		String act;
		String state = game.serialize();
		if (random.nextDouble() < explorationRate
				|| !qTable.containsKey(state)
				|| qTable.get(state).isEmpty()) {
			Map<String, Double> qActions = new HashMap<>();
			qTable.put(state, qActions);
			List<String> actions = game.getAllAvailableActions();
			actions.forEach(action -> qActions.put(action, 0d));
			act = actions.get(random.nextInt(actions.size()));
		} else {
			act = chooseBestAction(state);
		}
		game.play(player,
				Integer.parseInt(String.valueOf(act.charAt(0))),
				Integer.parseInt(String.valueOf(act.charAt(1))));
		moves.add(new Move(state, act));
	}

	public String chooseBestAction(String state) {
		List<Entry<String,Double>> list = qTable.get(state)
						.entrySet()
						.stream()
						.sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
					.toList();
		int maxIndex = 1;
		double highestValue = list.get(0).getValue();
		for (int i = 1; i < list.size(); i++) {
			if (highestValue == list.get(i).getValue()) {
				maxIndex ++;
			} else {
				break;
			}
		}
		return list.get(random.nextInt(0, maxIndex)).getKey();
	}

	@Override
	public void updateStrategy(double reward) {
		for (int i = moves.size() - 1; i >= 0; i--) {
			Move move = moves.get(i);
			double r = reward;
			String nextState = null;
			if (i < moves.size() - 1) {
				r = 0;
				nextState = moves.get(i + 1).state();
			}
			updateQTable(move.state(), move.action(), nextState, r);
		}
	}

	private void updateQTable(String state, String action, String newState, double reward) {
		Map<String, Double> qState = qTable.get(state);
		double maxNextQValue = 0;
		if (newState != null && qTable.containsKey(newState)) {
			Map<String, Double> qNewState = qTable.get(newState);
			maxNextQValue = qNewState.values()
					.stream()
					.mapToDouble(a -> a)
					.max()
					.orElse(0d);
		}

		double qValue = qState.computeIfAbsent(action, s -> 0d);
		double qValue2 = (1-learningRate) * qValue 
		+ learningRate * (reward + discountFactor * maxNextQValue);
		qState.put(action, qValue2);
	}
}

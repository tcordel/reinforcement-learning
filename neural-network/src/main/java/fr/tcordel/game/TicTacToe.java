package fr.tcordel.game;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TicTacToe implements Game {

	public static final char EMPTY_CELL = ' ';
	public static final char X = Result.X.toString().charAt(0);
	public static final char O = Result.O.toString().charAt(0);
	char[][] board = new char[3][3];

	public void resetBoard() {
		for (int i = 0; i < board.length; i++) {
			Arrays.fill(board[i], 0, board[i].length, EMPTY_CELL);
		}
	}

	@Override
	public String serialize() {
		return serialize(board);
	}

	String serialize(char[][] board) {

		String serialized = "";
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				serialized += board[i][j];
			}
		}
		return serialized;
	}

	@Override
	public Result getResult() {
		return getGameResult(board);
	}

	Result getGameResult(char[][] state) {
		for (int i = 0; i < state.length; i++) {
			if (state[i][0] != EMPTY_CELL && state[i][0] == state[i][1] && state[i][0] == state[i][2]) {
				return Result.valueOf(String.valueOf(state[i][0]));
			}
			if (state[0][i] != EMPTY_CELL && state[0][i] == state[1][i] && state[0][i] == state[2][i]) {
				return Result.valueOf(String.valueOf(state[0][i]));
			}
		}

		if (state[0][0] != EMPTY_CELL && state[0][0] == state[1][1] && state[0][0] == state[2][2]) {
			return Result.valueOf(String.valueOf(state[0][0]));
		}
		if (state[2][0] != EMPTY_CELL && state[2][0] == state[1][1] && state[2][0] == state[0][2]) {
			return Result.valueOf(String.valueOf(state[2][0]));
		}
		for (int x = 0; x < state.length; x++) {
			for (int y = 0; y < state.length; y++) {
				if (state[x][y] == EMPTY_CELL) {
					return Result.PENDING;
				}
			}
		}

		return Result.DROW;
	}

	public void play(char player, int x, int y) {
		board[x][y] = player;
	}

	public List<String> getAllAvailableActions() {
		List<String> actions = new ArrayList<>();

		for (int x = 0; x < board.length; x++) {
			for (int y = 0; y < board.length; y++) {
				char c = board[x][y];
				System.err.println(c);
				if (c == EMPTY_CELL) {
					actions.add("%d%d".formatted(x, y));
				}
			}
		}
		return actions;

	}

	@Override
	public String toString() {

		String display = "";
		for (int x = 0; x < board.length; x++) {
			for (int y = 0; y < board.length; y++) {
				display += board[x][y];
			}
			display += System.lineSeparator();
		}
		return display;

	}
}

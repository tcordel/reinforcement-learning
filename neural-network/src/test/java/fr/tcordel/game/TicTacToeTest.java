package fr.tcordel.game;

import static fr.tcordel.game.TicTacToe.EMPTY_CELL;
import static fr.tcordel.game.TicTacToe.O;
import static fr.tcordel.game.TicTacToe.X;
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

class TicTacToeTest {

	@ParameterizedTest
	@MethodSource("gameOver")
	void testGameOver(char[][] board, Result result) {

		TicTacToe ttt = new TicTacToe();
		assertThat(ttt.getGameResult(board))
				.isEqualTo(result);
	}

	static Stream<Arguments> gameOver() {
		return Stream.of(
				Arguments.of(new char[][] {
						{ EMPTY_CELL, EMPTY_CELL, EMPTY_CELL },
						{ EMPTY_CELL, EMPTY_CELL, EMPTY_CELL },
						{ EMPTY_CELL, EMPTY_CELL, EMPTY_CELL }
				}, Result.PENDING),

				Arguments.of(new char[][] {
						{ X, X, O },
						{ O, O, X },
						{ X, X, O },
				}, Result.DROW),

				Arguments.of(new char[][] {
						{ X, X, X },
						{ O, O, X },
						{ X, X, O },
				}, Result.X),
				Arguments.of(new char[][] {
						{ O, X, O },
						{ O, O, X },
						{ X, X, O },
				}, Result.O)
		);
	}



}

package fr.tcordel.rl.neural;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import fr.tcordel.game.Result;
import fr.tcordel.game.TicTacToe;
import fr.tcordel.rl.dummy.RandomAgent;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

class ACAgentTest {

    private TicTacToe game;
    private ACAgent agent;

    @BeforeEach
    void setUp() {
        game = new TicTacToe();
        agent = new ACAgent(game);
    }

    @Test
    void testAgentInitialization() {
        assertNotNull(agent);
        assertEquals(TicTacToe.X, agent.player);
    }

    @Test
    void testPlay() {
        agent.setPlayer(TicTacToe.O);
        agent.play(0.1);
        // Check if a move was made
        assertNotEquals(Result.PENDING, game.getResult());
    }

    @Test
    void testTrain() {
        RandomAgent randomAgent = new RandomAgent(game, TicTacToe.X);
        assertDoesNotThrow(() -> agent.train(randomAgent, 10));
    }

    @Test
    void testComputeCriticGradients() {
        // Create a dummy batch
        List<double[][]> batch = new ArrayList<>();
        batch.add(new double[][]{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0}, {0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0}});
        Map<String, double[][]> gradients = agent.computeCriticGradients(agent.critic, batch);
        assertNotNull(gradients);
        assertEquals(4, gradients.size());
    }
}

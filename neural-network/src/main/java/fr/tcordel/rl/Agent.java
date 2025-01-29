package fr.tcordel.rl;

public interface Agent {

	void play(double explorationRate);

	void updateStrategy(double reward);

	default void reset() {
	}
}

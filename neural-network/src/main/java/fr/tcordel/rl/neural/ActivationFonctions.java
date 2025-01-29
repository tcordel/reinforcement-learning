package fr.tcordel.rl.neural;

import java.util.function.DoubleUnaryOperator;

public interface ActivationFonctions {
	DoubleUnaryOperator NONE = value -> value;
	DoubleUnaryOperator SIGMOID = value -> 1d / (1 + Math.exp(-value));
	DoubleUnaryOperator TANH = value -> (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
	DoubleUnaryOperator RELU = value -> value > 0 ? value : 0;
}

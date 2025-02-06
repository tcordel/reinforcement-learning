package fr.tcordel.rl.neural;

import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.IntFunction;

public interface WeightInitializor {

	DoubleSupplier ZERO = () -> 0d;
	DoubleSupplier RANDOM_POSITIVE = new DoubleSupplier() {
		Random random = new Random();

		@Override
		public double getAsDouble() {
			return random.nextDouble();
		}
	};
	DoubleSupplier RANDOM = new DoubleSupplier() {
		Random random = new Random();

		@Override
		public double getAsDouble() {
			return random.nextDouble() * 2 - 1;
		}
	};

	// for sigmoid and tanh
	IntFunction<DoubleSupplier> XAVIER = inputNodeLength -> new DoubleSupplier() {

		Random random = new Random();
		double threshold = 1 / (Math.sqrt(inputNodeLength));

		public double getAsDouble() {

			return random.nextDouble(-threshold, threshold);
		}
	};

	// for ReLU
	IntFunction<DoubleSupplier> GAUSSIAN_HE = inputNodeLength -> new DoubleSupplier() {
		Random random = new Random();
		double threshold = Math.sqrt(2d / inputNodeLength);

		public double getAsDouble() {
			return random.nextGaussian(0, threshold);
		}
	};
}

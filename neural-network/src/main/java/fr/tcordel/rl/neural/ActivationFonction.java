package fr.tcordel.rl.neural;

public enum ActivationFonction {
	NONE {
		public double forward(double value) {
			return value;
		}

		public double backward(double value) {
			return 1d;
		}
	},
	SIGMOID {
		public double forward(double value) {
			return 1d / (1 + Math.exp(-value));
		}

		public double backward(double value) {
			return value * (1d - value);
		}
	},
	TANH {
		public double forward(double value) {
			return (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
		}

		public double backward(double value) {
			double tanh = forward(value);
			return 1d - tanh * tanh;
		}
	},
	RELU {
		public double forward(double value) {
			return value > 0 ? value : 0;
		}

		public double backward(double value) {
			return value > 0 ? 1d : 0d;
		}
	};

	public abstract double forward(double value);

	public abstract double backward(double value);
}

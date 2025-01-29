package fr.tcordel.cartpole;

import java.util.Random;

public class CartPole {
	public static final double CART_MASS = 0.31; // (kg)
	public static final double POLE_MASS = 0.055; // (kg)
	public static final double POLE_LENGTH = 0.4; // (m)
	public static final double X_THRESHOLD = 1.0;
	public static final double THETA_THRESHOLD = 12 * 2 * Math.PI / 360;

	private double cartPosition;
	private double cartVelocity;
	private double poleAngle;
	private double poleAngularVelocity;

	private boolean done;
	private double[] state;
	private int step;

	private final Random random = new Random();

	public CartPole() {
		this.state = new double[4];
		this.done = true;
	}

	public double[] reset() {
		this.step = 0;
		this.cartPosition = Math.tanh(random.nextGaussian() * 0.01) * 4.8; // (m)
		this.cartVelocity = random.nextDouble() * 0.1 - 0.05; // (m/s)
		double initialPoleAngle = random.nextDouble() * 0.1 - 0.05;
		this.poleAngle = (initialPoleAngle + Math.PI) % (2 * Math.PI) - Math.PI; // (rad)
		this.poleAngularVelocity = random.nextDouble() * 0.1 - 0.05; // (rad/s)

		this.state = new double[] { cartPosition, cartVelocity, poleAngle, poleAngularVelocity };
		this.done = false;
		return this.state;
	}

	public StepResult step(int action) {
		return step((action - 0.5) * 2);
	}

	public StepResult step(double action) {
		if (this.done) {
			throw new IllegalStateException("Cannot run step() before reset");
		}

		this.step++;

		// Add a small random noise
		double force = 1.0 * (action + (random.nextDouble() * 0.04 - 0.02));

		double totalMass = CART_MASS + POLE_MASS;
		double poleHalfLength = POLE_LENGTH / 2;
		double poleMassLength = POLE_MASS * poleHalfLength;

		double cosTheta = Math.cos(this.poleAngle);
		double sinTheta = Math.sin(this.poleAngle);

		double temp = (force + poleMassLength * Math.pow(this.poleAngularVelocity, 2) * sinTheta) / totalMass;
		double angularAccel = (9.8 * sinTheta - cosTheta * temp) /
				(poleHalfLength * (4.0 / 3.0 - (POLE_MASS * Math.pow(cosTheta, 2)) / totalMass));
		double linearAccel = temp - (poleMassLength * angularAccel * cosTheta) / totalMass;

		this.cartPosition += 0.02 * this.cartVelocity;
		this.cartVelocity += 0.02 * linearAccel;

		this.poleAngle = (this.poleAngle + 0.02 * this.poleAngularVelocity);
		this.poleAngle = (this.poleAngle + Math.PI) % (2 * Math.PI) - Math.PI;

		this.poleAngularVelocity += 0.02 * angularAccel;

		this.state = new double[] { cartPosition, cartVelocity, poleAngle, poleAngularVelocity };

		boolean term = this.state[0] < -X_THRESHOLD ||
				this.state[0] > X_THRESHOLD ||
				this.state[2] < -THETA_THRESHOLD ||
				this.state[2] > THETA_THRESHOLD;

		boolean trunc = (this.step == 500);
		this.done = term || trunc;

		return new StepResult(this.state, 1.0, term, trunc);
	}

	public boolean isDone() {
		return this.done;
	}

	public static record StepResult(double[] state, double reward, boolean terminated, boolean truncated) {
	}
}

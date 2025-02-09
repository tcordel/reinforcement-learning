package fr.tcordel.basic;

import static org.assertj.core.api.Assertions.assertThat;

import fr.tcordel.rl.neural.NeuralNetwork;
import fr.tcordel.rl.neural.WeightInitializor;

import org.junit.jupiter.api.Test;

import java.util.List;

class SolutionTest {

	@Test
	void testFirstStep() {

		int inputs = 1;
		int outputs = 1;
		int hiddenLayers = 0;
		int testInputs = 2;
		int trainingExamples = 2;
		int trainingIterations = 7;
		List<double[]> ins = List.of(
				new double[] { 0 },
				new double[] { 1 });

		List<double[]> insTest = List.of(
				new double[] { 0 },
				new double[] { 1 });
		List<double[]> ousTest = List.of(
				new double[] { 0 },
				new double[] { 1 });
		NeuralNetwork neuralNetwork = new NeuralNetwork(WeightInitializor.RANDOM_POSITIVE, inputs, outputs);
		for (int j = 0; j < trainingIterations; j++) {
			for (int i = 0; i < trainingExamples; i++) {
				neuralNetwork.train(insTest.get(i), ousTest.get(i));
			}
		}
		System.out.println(neuralNetwork.predict(new double[] { 0 })[0]);
		System.out.println(neuralNetwork.predict(new double[] { 1 })[0]);
	}

	@Test
	void testOr() {
		int inputs = 2;
		int outputs = 1;
		int hiddenLayers = 0;
		int testInputs = 2;
		int trainingExamples = 2;
		int trainingIterations = 7;
		List<double[]> ins = List.of(
				new double[] { 0, 0 },
				new double[] { 1, 1 });

		List<double[]> insTest = List.of(
				new double[] { 0, 0 },
				new double[] { 1, 0 },
				new double[] { 0, 1 },
				new double[] { 1, 1 });
		List<double[]> ousTest = List.of(
				new double[] { 0 },
				new double[] { 1 },
				new double[] { 1 },
				new double[] { 1 });
		NeuralNetwork neuralNetwork = new NeuralNetwork(WeightInitializor.RANDOM_POSITIVE, inputs, outputs);
		for (int j = 0; j < trainingIterations; j++) {
			for (int i = 0; i < trainingExamples; i++) {
				neuralNetwork.train(insTest.get(i), ousTest.get(i));
			}
		}
		System.out.println(neuralNetwork.predict(new double[] { 0, 0 })[0]);
		System.out.println(neuralNetwork.predict(new double[] { 1, 0 })[0]);
	}

	@Test
	void testXor() {
		NeuralNetwork neuralNetwork = new NeuralNetwork(WeightInitializor.RANDOM_POSITIVE, 2, 2, 1);

		int trainingExamples = 4;
		int trainingIterations = 2400;

		List<double[]> ins = List.of(
				new double[] { 0, 0 },
				new double[] { 0, 1 },
				new double[] { 1, 0 },
				new double[] { 1, 1 });

		List<double[]> insTest = List.of(
				new double[] { 0, 0 },
				new double[] { 0, 1 },
				new double[] { 1, 0 },
				new double[] { 1, 1 });
		List<double[]> ousTest = List.of(
				new double[] { 0 },
				new double[] { 1 },
				new double[] { 1 },
				new double[] { 0 });

		for (int j = 0; j < trainingIterations; j++) {
			for (int i = 0; i < trainingExamples; i++) {
				neuralNetwork.train(insTest.get(i), ousTest.get(i));
			}
		}
		for (int i = 0; i < trainingExamples; i++) {
			assertThat(Math.round(neuralNetwork.predict(ins.get(i))[0])).isEqualTo((int) ousTest.get(i)[0]);
		}
		ins.forEach(d -> System.out.println(neuralNetwork.predict(d)[0]));
	}

	@Test
	void testHidden() {
	NeuralNetwork neuralNetwork = new NeuralNetwork(WeightInitializor.RANDOM_POSITIVE, 4, 2, 2, 1);

		int trainingExamples = 16;
		int trainingIterations = 2400;

		List<double[]> ins = List.of(
				new double[] { 0, 0, 0, 0 },
				new double[] { 0, 0, 0, 1 },
				new double[] { 0, 0, 1, 0 },
				new double[] { 0, 0, 1, 1 },
				new double[] { 0, 1, 0, 0 },
				new double[] { 0, 1, 0, 1 },
				new double[] { 0, 1, 1, 0 },
				new double[] { 0, 1, 1, 1 },
				new double[] { 1, 0, 0, 0 },
				new double[] { 1, 0, 0, 1 },
				new double[] { 1, 0, 1, 0 },
				new double[] { 1, 0, 1, 1 },
				new double[] { 1, 1, 0, 0 },
				new double[] { 1, 1, 0, 1 },
				new double[] { 1, 1, 1, 0 },
				new double[] { 1, 1, 1, 1 });
		List<double[]> insTest = ins;
		List<double[]> ousTest = List.of(
				new double[] { 1 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 1 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 1 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 0 },
				new double[] { 1 });

		for (int j = 0; j < trainingIterations; j++) {
			for (int i = 0; i < trainingExamples; i++) {
				neuralNetwork.train(insTest.get(i), ousTest.get(i));
			}
			for (int i = trainingExamples - 1; i >= 0; i--) {
				neuralNetwork.train(insTest.get(i), ousTest.get(i));
			}
		}

		for (int i = 0; i < trainingExamples; i++) {
			assertThat(Math.round(neuralNetwork.predict(ins.get(i))[0])).isEqualTo((int) ousTest.get(i)[0]);
		}
		ins.forEach(d -> System.out.println(neuralNetwork.predict(d)[0]));
	}

}

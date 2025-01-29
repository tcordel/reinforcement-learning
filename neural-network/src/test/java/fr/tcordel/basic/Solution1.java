package fr.tcordel.basic;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
class CodinGame {

	public static void main(String args[]) {
		Scanner in = new Scanner(System.in);
		int inputs = in.nextInt();
		int outputs = in.nextInt();
		int hiddenLayers = in.nextInt();
		int testInputs = in.nextInt();
		int trainingExamples = in.nextInt();
		int trainingIterations = in.nextInt();
		int[] layers = new int[2 + hiddenLayers];
		layers[0] = inputs;
		layers[layers.length - 1] = outputs;
		System.err.println(
				"inputs " + inputs +
						" outputs " + outputs +
						" hiddenLayers " + hiddenLayers +
						" testInputs " + testInputs +
						" trainingExamples " + trainingExamples +
						" trainingIterations " + trainingIterations);
		for (int i = 0; i < hiddenLayers; i++) {
			int nodes = in.nextInt();
			layers[i + 1] = nodes;
			System.err.println("Node" + nodes);
		}
		in.nextLine();
		List<double[]> testInputsData = new ArrayList<>();
		NeuralNetwork nn = new NeuralNetwork(layers);
		for (int i = 0; i < testInputs; i++) {
			testInputsData.add(getCharArray(in));
		}
		List<double[]> trainingInputs = new ArrayList<>();
		List<double[]> trainingOutputs = new ArrayList<>();
		for (int i = 0; i < trainingExamples; i++) {
			trainingInputs.add(getCharArray(in));
			trainingOutputs.add(getCharArray(in));
			System.err.println("training " + CodinGame.doublesToString(trainingInputs.get(i)) + " out "
					+ CodinGame.doublesToString(trainingOutputs.get(i)));
		}

		for (int i = 0; i < trainingIterations; i++) {

			for (int j = 0; j < trainingExamples; j++) {
				nn.train(trainingInputs.get(j), trainingOutputs.get(j));
			}
		}

		for (int i = 0; i < testInputs; i++) {

			double[] predict = nn.predict(testInputsData.get(i));
			String answer = doublesToString(predict);
			String expected = "none";
			String doublesToString = doublesToString(testInputsData.get(i));
			for (int j = 0; j < trainingInputs.size(); j++) {
				if (doublesToString.equalsIgnoreCase(doublesToString(trainingInputs.get(i)))) {
					expected = doublesToString(trainingOutputs.get(i));

				}
			}
			System.err.println("answer for " + doublesToString + " " + answer + " expected " + expected);
			System.out.println(answer);
		}
	}

	static String doublesToString(double[] predict) {
		String answer = Arrays.stream(predict)
				.map(Math::round)
				.mapToInt(d -> (int) d)
				.mapToObj(String::valueOf)
				.collect(Collectors.joining(""));
		return answer;
	}

	private static double[] getCharArray(Scanner in) {
		String next = in.next();
		System.err.println(next);
		char[] charArray = next.toCharArray();
		return IntStream.range(0, charArray.length)
				.mapToDouble(c -> Double.parseDouble(String.valueOf(charArray[c]))).toArray();
	}
}

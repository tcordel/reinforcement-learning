package fr.tcordel.utils;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public abstract class Matplot {

	private Matplot() {
	}

	public static void print(List<Double> values) throws IOException, PythonExecutionException {

		List<Double> meanValues = new ArrayList<>();

		for (int i = 0; i < values.size(); i++) {
			meanValues.add(values.subList(Math.max(0, i - 49), i + 1).stream().mapToDouble(Double::doubleValue)
					.average().orElse(0));

		}

		Plot plot = Plot.create();
		plot.plot().add(
				IntStream.range(0, values.size()).mapToObj(i -> i).toList(),
				values)
				.linestyle("-")
				.color("blue");
		plot.plot().add(
				IntStream.range(0, values.size()).mapToObj(i -> i).toList(),
				meanValues).linestyle("-")
				.color("orange");
		plot.show();
	}

}

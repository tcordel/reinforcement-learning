package fr.tcordel.cartpole;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

public class PolicyGradientTest {
    @Test
    void testCrossEntropy() {
		assertThat(PolicyGradient.crossEntropy(new double[] {1,1}, new double[]{1,1}))
			.asString()
			.startsWith("1.3863");

    }
}


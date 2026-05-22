"""Approximate is_leap_year with an ordinary one-hidden-layer neural network.

This follows the visual idea in Michael Nielsen's universal approximation
chapter: a sigmoid neuron with a large weight behaves like a soft step. Two
shifted steps can be subtracted to make a narrow tower, and many towers can
approximate a function's graph.

Here the input is the raw year. The hidden layer contains two sigmoid neurons
for each leap year in the chosen fitting interval, 1600-2399:

    left sigmoid:   sigmoid(slope * x + bias_left)
    right sigmoid:  sigmoid(slope * x + bias_right)

The final output-layer neuron takes all hidden activations as its inputs. In its
pre-activation sum, each left sigmoid gets a positive weight and each right
sigmoid gets a negative weight. Their difference creates one local tower for
each known leap year; the final sigmoid turns the tower sum into a 0-to-1
score. Outside the fitting interval there are no towers, so the network does
not extrapolate the Gregorian rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def is_leap_year(year: int) -> bool:
    """Return whether a year is a Gregorian leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class SigmoidNeuron:
    """A standard neuron: activation = sigmoid(weight * input + bias)."""

    weight: float
    bias: float

    def activate(self, x: float) -> float:
        return sigmoid(self.weight * x + self.bias)


@dataclass(frozen=True)
class OutputNeuron:
    """A sigmoid output neuron with one weight per hidden activation."""

    weights: list[float]
    bias: float

    def activate(self, hidden_outputs: list[float]) -> float:
        z = self.bias
        for weight, hidden_output in zip(self.weights, hidden_outputs):
            z += weight * hidden_output
        return sigmoid(z)


class OneHiddenLayerNetwork:
    """Input layer -> sigmoid hidden layer -> sigmoid output layer."""

    def __init__(self, hidden_layer: list[SigmoidNeuron], output_neuron: OutputNeuron) -> None:
        self.hidden_layer = hidden_layer
        self.output_neuron = output_neuron

    def forward(self, x: float) -> float:
        hidden_outputs = [neuron.activate(x) for neuron in self.hidden_layer]
        return self.output_neuron.activate(hidden_outputs)


class LeapYearApproximator:
    """Build a neural network whose hidden neurons form leap-year bumps."""

    def __init__(
        self,
        fit_start: int = 1600,
        fit_stop: int = 2400,
        half_width: float = 0.45,
        hidden_slope: float = 20.0,
        output_slope: float = 12.0,
    ) -> None:
        self.fit_start = fit_start
        self.fit_stop = fit_stop
        self.half_width = half_width
        self.hidden_slope = hidden_slope
        self.output_slope = output_slope
        self.leap_years = [year for year in range(fit_start, fit_stop) if is_leap_year(year)]
        self.network = self._build_network()

    def _build_network(self) -> OneHiddenLayerNetwork:
        hidden_layer: list[SigmoidNeuron] = []
        output_weights: list[float] = []

        for leap_year in self.leap_years:
            left_edge = leap_year - self.half_width
            right_edge = leap_year + self.half_width

            # Left sigmoid: sigmoid(k * (x - left_edge)).
            left_sigmoid = SigmoidNeuron(
                weight=self.hidden_slope,
                bias=-self.hidden_slope * left_edge,
            )
            # Right sigmoid: sigmoid(k * (x - right_edge)).
            right_sigmoid = SigmoidNeuron(
                weight=self.hidden_slope,
                bias=-self.hidden_slope * right_edge,
            )

            hidden_layer.extend([left_sigmoid, right_sigmoid])
            output_weights.extend([self.output_slope, -self.output_slope])

        # The hidden layer sums to about 1 on a leap-year tower and about 0
        # elsewhere, so this output threshold sits halfway between them.
        output_neuron = OutputNeuron(
            weights=output_weights,
            bias=-0.5 * self.output_slope,
        )
        return OneHiddenLayerNetwork(hidden_layer, output_neuron)

    def score(self, year: int) -> float:
        """Return a neural-network score near 1 for leap years and near 0 otherwise."""
        return self.network.forward(float(year))

    def predict(self, year: int) -> bool:
        return self.score(year) >= 0.5


def evaluate(model: LeapYearApproximator, start: int, stop: int) -> tuple[int, int]:
    """Return `(correct, total)` for years in `[start, stop)`."""
    total = stop - start
    correct = sum(model.predict(year) == is_leap_year(year) for year in range(start, stop))
    return correct, total


def evaluate_ranges(model: LeapYearApproximator, ranges: list[tuple[int, int]]) -> tuple[int, int]:
    """Return `(correct, total)` across multiple `[start, stop)` ranges."""
    correct = 0
    total = 0
    for start, stop in ranges:
        range_correct, range_total = evaluate(model, start, stop)
        correct += range_correct
        total += range_total
    return correct, total


def print_sample_predictions(model: LeapYearApproximator) -> None:
    years = [
        1896,
        1900,
        1996,
        2000,
        2001,
        2004,
        2100,
        2400,
    ]

    print("Sample predictions")
    print("year  exact  network_score  predicted")
    print("----  -----  -------------  ---------")
    for year in years:
        exact = is_leap_year(year)
        score = model.score(year)
        predicted = model.predict(year)
        print(f"{year}  {str(exact):5}  {score:13.8f}  {predicted}")


def print_out_of_range_predictions(model: LeapYearApproximator) -> None:
    years = [
        1200,
        1500,
        1582,
        2401,
        2800,
        3000,
        3200,
    ]

    print("Predictions outside the fitting range")
    print("year  exact  network_score  predicted")
    print("----  -----  -------------  ---------")
    for year in years:
        exact = is_leap_year(year)
        score = model.score(year)
        predicted = model.predict(year)
        print(f"{year}  {str(exact):5}  {score:13.8f}  {predicted}")


def print_network_summary(model: LeapYearApproximator) -> None:
    print("Network structure")
    print("input neurons: 1  (x = raw year)")
    print(f"hidden neurons: {len(model.network.hidden_layer)} sigmoid neurons")
    print("output neurons: 1 sigmoid neuron")
    print(f"fitting interval: {model.fit_start}-{model.fit_stop - 1}")
    print()
    print("First eight hidden neurons")
    print("index  weight       bias")
    print("-----  ---------  ---------")
    for index, neuron in enumerate(model.network.hidden_layer[:8]):
        print(f"{index:5}  {neuron.weight:9.3f}  {neuron.bias:9.3f}")


def main() -> None:
    model = LeapYearApproximator(hidden_slope=20.0, output_slope=12.0)

    in_correct, in_total = evaluate(model, 1600, 2400)
    out_correct, out_total = evaluate_ranges(model, [(1200, 1600), (2400, 2800)])
    print_sample_predictions(model)
    print()
    print_out_of_range_predictions(model)
    print()
    print_network_summary(model)
    print()
    print(f"Accuracy on fitted years 1600-2399: {in_correct}/{in_total} = {in_correct / in_total:.1%}")
    print(
        "Accuracy outside fitted years 1200-1599 and 2400-2799: "
        f"{out_correct}/{out_total} = {out_correct / out_total:.1%}"
    )
    print()
    print("Interpretation:")
    print("- Every hidden neuron has the usual weight and bias.")
    print("- Two hidden neurons create one sigmoid bump around one fitted leap year.")
    print("- The output neuron combines all bumps and applies a final sigmoid.")
    print("- Years outside the fitting interval have no bumps, so extrapolation fails.")


if __name__ == "__main__":
    main()

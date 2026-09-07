"""Tests for the scalar neural-network building blocks."""

import math
import random

import pytest

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Module, Neuron


def set_neuron_parameters(neuron, weights, bias=0.0):
    for parameter, value in zip(neuron.w, weights, strict=True):
        parameter.data = value
    neuron.b.data = bias


class TestModule:
    def test_default_parameters_are_empty(self):
        assert Module().parameters() == []

    def test_default_zero_grad_is_safe(self):
        assert Module().zero_grad() is None


class TestNeuron:
    def test_initializes_one_weight_per_input(self):
        neuron = Neuron(3)

        assert len(neuron.w) == 3
        assert all(isinstance(weight, Value) for weight in neuron.w)
        assert isinstance(neuron.b, Value)
        assert neuron.b.data == pytest.approx(0.0)
        assert all(-1.0 <= weight.data <= 1.0 for weight in neuron.w)

    def test_parameters_returns_weights_followed_by_bias(self):
        neuron = Neuron(3)

        assert neuron.parameters() == [*neuron.w, neuron.b]

    def test_seed_makes_initialization_reproducible(self):
        original_state = random.getstate()
        try:
            random.seed(42)
            first = Neuron(3)
            random.seed(42)
            second = Neuron(3)
        finally:
            random.setstate(original_state)

        assert [weight.data for weight in first.w] == [weight.data for weight in second.w]

    def test_forward_matches_manual_calculation(self):
        neuron = Neuron(2)
        set_neuron_parameters(neuron, [0.5, -0.25], bias=0.1)

        output = neuron([2.0, 4.0])

        assert output.data == pytest.approx(math.tanh(0.1))

    def test_forward_accepts_value_inputs(self):
        neuron = Neuron(2)
        set_neuron_parameters(neuron, [0.5, -0.25])

        output = neuron([Value(2.0), Value(2.0)])

        assert isinstance(output, Value)
        assert output.data == pytest.approx(math.tanh(0.5))

    def test_rejects_wrong_input_width(self):
        neuron = Neuron(2)

        with pytest.raises(ValueError):
            neuron([1.0])

    def test_backward_reaches_inputs_weights_and_bias(self):
        neuron = Neuron(2)
        set_neuron_parameters(neuron, [0.5, -0.25], bias=0.1)
        left = Value(2.0)
        right = Value(4.0)
        output = neuron([left, right])

        output.backward()

        local_gradient = 1.0 - output.data**2
        assert neuron.w[0].grad == pytest.approx(local_gradient * left.data)
        assert neuron.w[1].grad == pytest.approx(local_gradient * right.data)
        assert neuron.b.grad == pytest.approx(local_gradient)
        assert left.grad == pytest.approx(local_gradient * neuron.w[0].data)
        assert right.grad == pytest.approx(local_gradient * neuron.w[1].data)


class TestLayer:
    def test_initializes_requested_number_of_neurons(self):
        layer = Layer(3, 4)

        assert len(layer.neurons) == 4
        assert all(len(neuron.w) == 3 for neuron in layer.neurons)

    def test_parameters_flattens_all_neuron_parameters(self):
        layer = Layer(3, 4)
        expected = [parameter for neuron in layer.neurons for parameter in neuron.parameters()]

        assert layer.parameters() == expected
        assert len(layer.parameters()) == 16

    def test_multiple_neurons_return_list(self):
        layer = Layer(2, 2)
        set_neuron_parameters(layer.neurons[0], [1.0, 0.0])
        set_neuron_parameters(layer.neurons[1], [0.0, 1.0])

        outputs = layer([0.5, -0.25])

        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert outputs[0].data == pytest.approx(math.tanh(0.5))
        assert outputs[1].data == pytest.approx(math.tanh(-0.25))

    def test_single_neuron_returns_value(self):
        layer = Layer(2, 1)

        output = layer([0.5, -0.25])

        assert isinstance(output, Value)


class TestMLP:
    def test_builds_requested_architecture(self):
        model = MLP(3, [4, 4, 1])

        assert [len(layer.neurons) for layer in model.layers] == [4, 4, 1]
        assert [len(layer.neurons[0].w) for layer in model.layers] == [3, 4, 4]

    def test_architecture_has_expected_parameter_count(self):
        model = MLP(3, [4, 4, 1])
        parameters = model.parameters()

        assert len(parameters) == 41
        assert all(isinstance(parameter, Value) for parameter in parameters)

    def test_parameters_preserves_layer_order(self):
        model = MLP(2, [3, 1])
        expected = [parameter for layer in model.layers for parameter in layer.parameters()]

        assert model.parameters() == expected

    def test_zero_grad_clears_every_parameter_gradient(self):
        model = MLP(2, [3, 1])
        parameters = model.parameters()
        original_data = [parameter.data for parameter in parameters]
        for index, parameter in enumerate(parameters, start=1):
            parameter.grad = float(index)

        returned = model.zero_grad()

        assert returned is None
        assert all(parameter.grad == 0.0 for parameter in parameters)
        assert [parameter.data for parameter in parameters] == original_data

    def test_forward_passes_output_of_each_layer_to_next(self):
        model = MLP(2, [2, 1])
        set_neuron_parameters(model.layers[0].neurons[0], [1.0, 0.0])
        set_neuron_parameters(model.layers[0].neurons[1], [0.0, 1.0])
        set_neuron_parameters(model.layers[1].neurons[0], [1.0, -1.0])

        output = model([0.5, -0.25])

        expected = math.tanh(math.tanh(0.5) - math.tanh(-0.25))
        assert isinstance(output, Value)
        assert output.data == pytest.approx(expected)

    def test_multi_output_final_layer_returns_list(self):
        model = MLP(2, [3, 2])

        outputs = model([0.5, -0.25])

        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert all(isinstance(output, Value) for output in outputs)

    def test_backward_reaches_every_layer(self):
        model = MLP(2, [2, 1])
        for layer in model.layers:
            for neuron in layer.neurons:
                set_neuron_parameters(neuron, [0.2] * len(neuron.w), bias=0.1)

        output = model([Value(0.5), Value(-0.25)])
        output.backward()

        parameters = model.parameters()
        assert all(math.isfinite(parameter.grad) for parameter in parameters)
        assert all(parameter.grad != 0.0 for parameter in parameters)

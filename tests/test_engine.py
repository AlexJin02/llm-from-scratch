"""Tests for the scalar autograd engine."""

import math

import pytest
import torch

from micrograd.engine import Value


class TestValueBasics:
    def test_initial_state(self):
        value = Value(2.0)

        assert value.data == pytest.approx(2.0)
        assert value.grad == pytest.approx(0.0)
        assert value.children == set()
        assert value.op == ""

    def test_repr_contains_data_and_gradient(self):
        value = Value(2.0)
        value.grad = 3.0

        assert repr(value) == "Value(data=2.0, grad=3.0)"

    def test_operation_records_graph_metadata(self):
        left = Value(2.0)
        right = Value(3.0)
        result = left + right

        assert result.children == {left, right}
        assert result.op == "+"


class TestForwardOperations:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (Value(2.0) + Value(3.0), 5.0),
            (Value(2.0) + 3.0, 5.0),
            (3.0 + Value(2.0), 5.0),
            (Value(5.0) - Value(2.0), 3.0),
            (5.0 - Value(2.0), 3.0),
            (Value(2.0) * Value(3.0), 6.0),
            (Value(2.0) * 3.0, 6.0),
            (3.0 * Value(2.0), 6.0),
            (-Value(2.0), -2.0),
            (Value(3.0) ** 2, 9.0),
            (Value(6.0) / Value(2.0), 3.0),
            (6.0 / Value(2.0), 3.0),
        ],
    )
    def test_arithmetic(self, result, expected):
        assert result.data == pytest.approx(expected)

    def test_power_rejects_value_exponent(self):
        with pytest.raises(AssertionError):
            Value(2.0) ** Value(3.0)

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (-2.0, 0.0),
            (0.0, 0.0),
            (2.0, 2.0),
        ],
    )
    def test_relu(self, input_value, expected):
        assert Value(input_value).relu().data == pytest.approx(expected)

    @pytest.mark.parametrize("input_value", [-1.0, 0.0, 1.0])
    def test_exp(self, input_value):
        assert Value(input_value).exp().data == pytest.approx(math.exp(input_value))

    @pytest.mark.parametrize("input_value", [-1.0, 0.0, 1.0])
    def test_tanh(self, input_value):
        assert Value(input_value).tanh().data == pytest.approx(math.tanh(input_value))

    def test_composed_expression(self):
        left = Value(2.0)
        right = Value(-3.0)

        result = (left * right + 10.0) / 2.0

        assert result.data == pytest.approx(2.0)


class TestLocalBackward:
    def test_addition(self):
        left = Value(2.0)
        right = Value(3.0)
        result = left + right
        result.grad = 2.5

        result._backward()

        assert left.grad == pytest.approx(2.5)
        assert right.grad == pytest.approx(2.5)

    def test_addition_accumulates_repeated_input(self):
        value = Value(2.0)
        result = value + value
        result.grad = 1.0

        result._backward()

        assert value.grad == pytest.approx(2.0)

    def test_multiplication(self):
        left = Value(2.0)
        right = Value(3.0)
        result = left * right
        result.grad = 2.0

        result._backward()

        assert left.grad == pytest.approx(6.0)
        assert right.grad == pytest.approx(4.0)

    def test_power(self):
        value = Value(3.0)
        result = value**2
        result.grad = 2.0

        result._backward()

        assert value.grad == pytest.approx(12.0)

    @pytest.mark.parametrize(
        ("input_value", "expected_gradient"),
        [
            (-2.0, 0.0),
            (0.0, 0.0),
            (2.0, 3.0),
        ],
    )
    def test_relu(self, input_value, expected_gradient):
        value = Value(input_value)
        result = value.relu()
        result.grad = 3.0

        result._backward()

        assert value.grad == pytest.approx(expected_gradient)

    def test_exp(self):
        value = Value(1.0)
        result = value.exp()
        result.grad = 2.0

        result._backward()

        assert value.grad == pytest.approx(2.0 * math.exp(1.0))
        assert isinstance(value.grad, int | float)

    def test_tanh(self):
        value = Value(0.5)
        result = value.tanh()
        result.grad = 2.0

        result._backward()

        assert value.grad == pytest.approx(2.0 * (1.0 - result.data**2))


class TestFullBackward:
    def test_simple_expression(self):
        left = Value(2.0)
        right = Value(3.0)
        result = left * right + left

        returned = result.backward()

        assert returned is None
        assert result.grad == pytest.approx(1.0)
        assert left.grad == pytest.approx(4.0)
        assert right.grad == pytest.approx(2.0)

    def test_shared_intermediate_accumulates_both_paths(self):
        value = Value(2.0)
        shared = value * 2.0
        result = shared * 3.0 + shared**2

        result.backward()

        assert shared.grad == pytest.approx(11.0)
        assert value.grad == pytest.approx(22.0)

    def test_backward_does_not_print_debug_output(self, capsys):
        result = Value(2.0) * Value(3.0)

        result.backward()

        assert capsys.readouterr().out == ""

    def test_matches_torch_autograd_for_smooth_expression(self):
        micro_a = Value(1.5)
        micro_b = Value(-0.5)
        micro_out = ((micro_a * micro_b + micro_a**2) / (micro_b + 2.0)).tanh()

        torch_a = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        torch_b = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        torch_out = torch.tanh((torch_a * torch_b + torch_a**2) / (torch_b + 2.0))

        micro_out.backward()
        torch_out.backward()

        assert torch_a.grad is not None
        assert torch_b.grad is not None
        assert micro_out.data == pytest.approx(torch_out.item())
        assert micro_a.grad == pytest.approx(torch_a.grad.item())
        assert micro_b.grad == pytest.approx(torch_b.grad.item())

    def test_matches_torch_autograd_for_relu_and_exp(self):
        micro_a = Value(2.0)
        micro_b = Value(3.0)
        micro_out = (micro_a * micro_b).relu() + (micro_a / micro_b).exp()

        torch_a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        torch_b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        torch_out = torch.relu(torch_a * torch_b) + torch.exp(torch_a / torch_b)

        micro_out.backward()
        torch_out.backward()

        assert torch_a.grad is not None
        assert torch_b.grad is not None
        assert micro_out.data == pytest.approx(torch_out.item())
        assert micro_a.grad == pytest.approx(torch_a.grad.item())
        assert micro_b.grad == pytest.approx(torch_b.grad.item())

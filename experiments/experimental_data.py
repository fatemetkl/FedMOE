import torch


def create_linear_line(num_points: int, a: float = 2.0, b: float = 1.0) -> torch.Tensor:
    # Generate data_length evenly spaced x values between 0 and data_length
    input = torch.linspace(-1, +1, steps=num_points)
    # Compute the corresponding y values using the equation y = ax + b
    data = a * input + b
    return data


def quadratic_data(num_points: int) -> torch.Tensor:
    # Generate equally spaced x-values between 1 and num_points
    x_values = torch.linspace(1, num_points, num_points)

    # y = a * x^2 + b * x + c
    a = -4 / (num_points - 1) ** 2  # to ensure it starts from -1 and ends at 1
    b = 4 / (num_points - 1)
    c = -1

    y_values = a * x_values**2 + b * x_values + c

    # Normalize the y-values to be between -1 and 1
    y_min, y_max = torch.min(y_values), torch.max(y_values)
    y_values_normalized = 2 * (y_values - y_min) / (y_max - y_min) - 1
    return y_values_normalized


def sine_signal(num_points: int) -> torch.Tensor:
    x_values = torch.linspace(1, num_points, num_points)
    return torch.sin(x_values)

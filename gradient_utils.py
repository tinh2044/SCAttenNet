import torch
import torch.nn as nn
import numpy as np


def clip_gradients(model, max_norm=1.0):
    """
    Clip gradients to prevent gradient explosion

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        grad_norm: Gradient norm before clipping
    """
    if max_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return grad_norm
    return None


def check_model_gradients(model, step=None):
    """
    Check for NaN or inf gradients in model parameters

    Args:
        model: PyTorch model
        step: Current training step (for logging)

    Returns:
        has_nan_grad: Boolean indicating if any gradient is NaN
        has_inf_grad: Boolean indicating if any gradient is inf
    """
    has_nan_grad = False
    has_inf_grad = False

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name} at step {step}")
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                print(f"Inf gradient detected in {name} at step {step}")
                has_inf_grad = True

    return has_nan_grad, has_inf_grad


def get_gradient_norm(model):
    """
    Calculate the gradient norm of model parameters

    Args:
        model: PyTorch model

    Returns:
        grad_norm: Gradient norm
    """
    total_norm = 0
    param_count = 0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    total_norm = total_norm ** (1.0 / 2)
    return total_norm if param_count > 0 else 0


def stabilize_model_weights(model):
    """
    Stabilize model weights by clamping extreme values

    Args:
        model: PyTorch model
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Reinitializing parameter {name} due to NaN/inf values")
                if len(param.shape) == 2:  # Linear layer weights
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) == 1:  # Bias or LayerNorm
                    nn.init.constant_(param, 0)
                else:
                    nn.init.normal_(param, 0, 0.02)


def log_gradient_stats(model, step):
    """
    Log gradient statistics for debugging

    Args:
        model: PyTorch model
        step: Current training step
    """
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
            }

    # Log statistics for key layers
    key_layers = [
        "recognition_head",
        "coordinates_fusion",
        "body_encoder",
        "left_encoder",
        "right_encoder",
    ]
    for layer_name in key_layers:
        layer_grads = [v for k, v in grad_stats.items() if layer_name in k]
        if layer_grads:
            avg_norm = np.mean([g["norm"] for g in layer_grads])
            print(f"Step {step} - {layer_name} avg gradient norm: {avg_norm:.6f}")


def reset_model_on_nan(model, optimizer):
    """
    Reset model and optimizer state when NaN is detected

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
    """
    print("Resetting model due to NaN/inf detection...")

    # Reset model weights
    stabilize_model_weights(model)

    # Reset optimizer state
    optimizer.state = {}

    # Zero gradients
    optimizer.zero_grad()

    print("Model and optimizer reset completed.")


def warmup_learning_rate(optimizer, step, warmup_steps, base_lr):
    """
    Apply learning rate warmup

    Args:
        optimizer: PyTorch optimizer
        step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
    """
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    return base_lr


def check_data_validity(data_dict):
    """
    Check if input data contains NaN or inf values

    Args:
        data_dict: Dictionary containing input data

    Returns:
        is_valid: Boolean indicating if data is valid
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN detected in input data: {key}")
                return False
            if torch.isinf(value).any():
                print(f"Inf detected in input data: {key}")
                return False
    return True


def safe_backward(loss, model, optimizer, max_grad_norm=1.0):
    """
    Safe backward pass with gradient clipping and NaN checking

    Args:
        loss: Loss tensor
        model: PyTorch model
        optimizer: PyTorch optimizer
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        success: Boolean indicating if backward pass was successful
    """
    try:
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss}")
            return False

        # Backward pass
        loss.backward()

        # Check gradients
        has_nan_grad, has_inf_grad = check_model_gradients(model)

        if has_nan_grad or has_inf_grad:
            print("Invalid gradients detected, skipping update")
            optimizer.zero_grad()
            return False

        # Clip gradients
        grad_norm = clip_gradients(model, max_grad_norm)

        # Check if gradient norm is reasonable
        if grad_norm is not None and grad_norm > 100:
            print(f"Large gradient norm detected: {grad_norm}")

        return True

    except Exception as e:
        print(f"Error in backward pass: {e}")
        optimizer.zero_grad()
        return False

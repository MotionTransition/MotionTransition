import torch
def apply_guidance(out, obs_x0, y, get_weight=False):
    """
    Apply linear blending guidance to the output of the diffusion model.
    
    Args:
        out (torch.Tensor): The model's output tensor of shape (B, 263, 1, T)
        obs_x0 (torch.Tensor): The ground truth observation tensor of same shape as out
        y (dict): Dictionary containing 'lengths' key with actual frame lengths for each sample
    
    Returns:
        torch.Tensor: Guided output with blended regions
    """
    batch_size, _, _, max_length = out.shape
    device = out.device
    
    # Initialize blending weights tensor
    weights = torch.zeros((batch_size, 1, 1, max_length), device=device)
    
    for b in range(batch_size):
        L = y['lengths'][b].item()  # Actual sequence length
        if L == 0:
            continue
        
        # Calculate head region (first 25%)
        h_length = max(1, int(round(0.25 * L)))
        if h_length > 0:
            # Create linearly decreasing weights from 1.0 to 0.0
            if h_length == 1:
                head_weights = torch.tensor([1.0], device=device)
            else:
                head_weights = torch.linspace(1.0, 0.0, steps=h_length, device=device)
            # head_weights = torch.tensor([1.0], device=device)
            weights[b, :, :, :h_length] = head_weights.view(1, 1, -1)
        
        # Calculate tail region (last 25%)
        t_length = h_length  # Keep same length as head
        t_start = max(0, L - t_length)
        if t_length > 0:
            # Create linearly increasing weights from 0.0 to 1.0
            if t_length == 1:
                tail_weights = torch.tensor([1.0], device=device)
            else:
                tail_weights = torch.linspace(0.0, 1.0, steps=t_length, device=device)
            # tail_weights = torch.tensor([1.0], device=device)
            
            # Handle cases where sequence is shorter than t_length
            end = min(t_start + t_length, L)
            actual_t_length = end - t_start
            if actual_t_length < t_length:
                tail_weights = tail_weights[:actual_t_length]
            
            weights[b, :, :, t_start:end] = tail_weights.view(1, 1, -1)
    
    if get_weight:
        return weights
    # Perform linear blending
    guided_out = out * (1 - weights) + obs_x0 * weights
    return guided_out
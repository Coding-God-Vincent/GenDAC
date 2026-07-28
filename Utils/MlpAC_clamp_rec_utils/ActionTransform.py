import torch

# action : (batch_size, action_dim)
# return : (batch_size, action_dim)
def center_action(action: torch.Tensor) -> torch.Tensor:
    """
    將每一筆 action 沿 action dimension 中心化。

    例如：
        [2, 1, 0] -> [1, 0, -1]

    Args:
        action: shape (..., action_dim)

    Returns:
        centered action，且最後一維平均值為 0
    """
    return action - action.mean(dim=-1, keepdim=True)


# return (batch_size, action_dim)
def clamp_and_center_action(
    raw_action: torch.Tensor,
    bound: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    產生最終交給 Critic、Replay Buffer 與 softmax 的 latent action。

    流程：
    1. 將原始 Gaussian sample clamp 到 [-bound, bound]
    2. 中心化
    3. 若中心化後超過 bound，進行等比例縮放

    最終同時保證：
    - 每筆 action 的平均值為 0
    - 每個元素的絕對值不超過 bound

    Args:
        raw_action: shape (..., action_dim)
        bound: action bound，目前設為 3
        eps: 避免除以 0

    Returns:
        processed action
    """
    if bound <= 0:
        raise ValueError(f"bound must be positive, got {bound}")

    # Gaussian 可能抽到很大的值，先直接截斷。
    clamped_action = torch.clamp(
        raw_action,
        min= -bound,
        max= bound,
    )

    # Softmax 對整體平移不敏感，因此消除無意義的共同偏移量。
    # (batch_size, action_dim)
    centered_action = center_action(clamped_action)

    # clamp 後再中心化，有可能重新超過 ±bound。
    # 使用等比例縮放可保留中心化與各維度之間的相對關係。
    max_abs = centered_action.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(
        bound / (max_abs + eps),
        max= 1.0,
    )

    return centered_action * scale
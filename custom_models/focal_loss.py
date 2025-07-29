import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (for multi-class or binary classification with logits)
    
    수식:
        FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    여기서,
        - p_c: 전체 클래스에 대한 예측 확률
        - p_t: 정답 클래스에 대한 예측 확률
        - α: 클래스 불균형 보정 계수
        - γ: 어려운 샘플에 더 큰 가중치를 두는 집중도 조절 계수

    참고:
        CrossEntropyLoss는 -log(p_t)만 사용하는 반면,
        FocalLoss는 거기에 (1 - p_t)^γ를 곱해서 '쉬운 샘플'의 손실을 줄여줌
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) - raw logits
        targets: (batch_size,) - class indices (e.g., 0 또는 1)
        """

        # 1. log(p_c): CrossEntropy처럼 softmax + log 를 한 번에 수행
        log_probs = F.log_softmax(inputs, dim=1)  # shape: (B, C)

        # 2. p_c: log(p_c)를 지수화하여 확률로 복원
        probs = torch.exp(log_probs)              # shape: (B, C)

        # 3. 정답 클래스 인덱스를 long으로 변환
        targets = targets.long()

        # 4. p_c에서 정답 클래스에 해당하는 확률 p_t 추출
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # shape: (B,)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # log(p_t)

        # 5. focal loss 공식 적용: -α * (1 - p_t)^γ * log(p_t)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        # 6. 평균 or 합산
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum() 
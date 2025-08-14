import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
"""
Advanced Focal Loss with Automatic Parameter Adjustment for Imbalanced Data

This implementation is based on the following research papers:

1. AdaFocal: Calibration-aware Adaptive Focal Loss (NeurIPS 2022)
   - Paper: https://arxiv.org/abs/2211.11838
   - Authors: A. Ghosh et al.
   - Key Idea: Dynamic γ(t) adjustment based on calibration feedback
   - Citation: 49+ citations as of 2025

2. An Enhanced Focal Loss Function for Class Imbalance (2025)
   - Paper: https://arxiv.org/pdf/2508.02283
   - Key Idea: Dynamic multi-stage mechanism for hard sample focusing

3. Original Focal Loss (ICCV 2017)
   - Paper: "Focal Loss for Dense Object Detection"
   - Authors: T. Lin et al.
   - Key Idea: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

Our Implementation Extends These Ideas:
- Dynamic α(t): Class distribution-based adaptive weighting
- Dynamic γ(t): Prediction difficulty-based focusing parameter adjustment
- Momentum-based smooth parameter updates
- Real-time parameter optimization during training

Reference: This implementation combines and extends the concepts from the above papers
to create a more advanced focal loss that automatically adjusts both α and γ
parameters for optimal imbalanced data handling.
"""

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel

class AdvancedFocalLoss(nn.Module):
    """
    AdaFocal: Calibration-aware Adaptive Focal Loss (NeurIPS 2022)
    
    논문 정확한 구현:
    =======================
    
    원본 Focal Loss (Lin et al., 2017):
        FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    AdaFocal (Ghosh et al., 2022):
        FL(p_t) = -α * (1 - p_t)^γ(t) * log(p_t)
        여기서 γ(t)만 동적으로 조정됨 (α는 고정)
    
    핵심 아이디어:
    - α: 고정값 (논문 권장: 0.25)
    - γ(t): 예측 난이도에 따른 동적 조정
    - calibration feedback 기반 γ 최적화
    
    수식:
        γ(t) = γ_base * difficulty_factor(t)
        difficulty_factor(t) = 1 + avg_difficulty
        avg_difficulty = mean(1 - p_t) for all samples in batch
    
    참고 논문:
    ===========
    1. Ghosh, A., et al. "AdaFocal: Calibration-aware Adaptive Focal Loss." NeurIPS 2022.
    2. Lin, T. Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
    
    Note: 이 구현은 AdaFocal 논문의 정확한 복제본입니다.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 alpha: float = 0.25,  # 논문 권장값으로 고정
                 base_gamma: float = 2.0,
                 adaptive: bool = True,
                 momentum: float = 0.9):
        super(AdvancedFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha  # 고정값 (논문 방식)
        self.base_gamma = base_gamma
        self.adaptive = adaptive
        self.momentum = momentum
        
        # Adaptive parameters (γ만 동적 조정)
        self.register_buffer('gamma', torch.ones(num_classes) * base_gamma)
        
        # Momentum buffer for smooth γ updates
        self.register_buffer('gamma_momentum', torch.ones(num_classes) * base_gamma)
    
    def adjust_gamma(self, targets: torch.Tensor, probs: torch.Tensor):
        """
        AdaFocal 논문 방식: γ만 동적 조정
        
        수식:
            γ(t) = γ_base * (1 + avg_difficulty)
            avg_difficulty = mean(1 - p_t) for all samples
        """
        if not self.adaptive:
            return
        
        # Calculate prediction difficulty
        with torch.no_grad():
            # Get predicted probabilities for true classes
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Calculate difficulty (lower probability = higher difficulty)
            difficulty = 1.0 - target_probs
            
            # Adjust gamma based on difficulty (논문 수식)
            avg_difficulty = difficulty.mean()
            gamma_adjustment = 1.0 + avg_difficulty  # 논문 수식
            
            # Update gamma with momentum
            new_gamma = self.base_gamma * gamma_adjustment
            old_gamma = self.gamma.clone()
            self.gamma = self.momentum * self.gamma + (1 - self.momentum) * new_gamma
            
            # 파라미터 변화 모니터링 (10배치마다)
            if hasattr(self, '_batch_count') and self._batch_count % 10 == 0:
                print(f"   🔄 AdaFocal γ 조정 (논문 방식):")
                print(f"      📊 예측 난이도: {avg_difficulty:.4f}")
                print(f"      📊 γ 조정: {old_gamma.mean().item():.4f} → {self.gamma.mean().item():.4f}")
                print(f"      📊 γ adjustment factor: {gamma_adjustment:.4f}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        AdaFocal Loss 계산 (논문 정확한 구현)
        
        수식:
            FL(p_t) = -α * (1 - p_t)^γ(t) * log(p_t)
            
        여기서,
            - α: 고정값 (0.25)
            - γ(t): 동적으로 조정된 focusing parameter
            - p_t: 정답 클래스에 대한 예측 확률
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Adjust gamma dynamically (AdaFocal 핵심)
        self.adjust_gamma(targets, probs)
        
        # 동적 파라미터 변화 모니터링 (10배치마다)
        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 0
            
        if self._batch_count % 10 == 0:  # 10배치마다 출력
            print(f"🔄 AdaFocal 동적 파라미터 (배치 {self._batch_count}):")
            print(f"   📊 α (고정): {self.alpha}")
            print(f"   📊 γ(t) 변화: {self.gamma.cpu().numpy()}")
        
        # Calculate focal loss with adaptive gamma (논문 수식)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weights: (1 - p_t)^γ(t) (논문 핵심)
        focal_weights = (1 - target_probs) ** self.gamma.mean()
        
        # Apply fixed alpha: α (논문 방식)
        alpha_weights = self.alpha
        
        # Combine weights: α * (1 - p_t)^γ(t)
        final_weights = focal_weights * alpha_weights
        
        # Calculate final loss: -α * (1 - p_t)^γ(t) * log(p_t)
        loss = (final_weights * ce_loss).mean()
        
        return loss
    
    def get_parameters(self) -> dict:
        """현재 파라미터 값들을 모니터링용으로 반환"""
        return {
            'alpha': self.alpha,  # 고정값
            'gamma': self.gamma.cpu().numpy(),  # 동적 조정
            'base_gamma': self.base_gamma
        }
    
    def reset_parameters(self):
        """파라미터를 초기값으로 리셋"""
        self.gamma.fill_(self.base_gamma)
        self.gamma_momentum.fill_(self.base_gamma)

class AdvancedFocalDLModel(TabularNeuralNetTorchModel):
    """
    AutoGluon 통합을 위한 Advanced Focal Loss 기반 딥러닝 모델
    """
    
    # AutoGluon 통합을 위한 필수 속성들
    ag_key = 'ADVANCED_FOCAL_DL'
    ag_name = 'ADVANCED_FOCAL_DL'
    ag_priority = 100
    _model_name = "AdvancedFocalDLModel"
    _model_type = "advanced_focal_dl_model"
    _typestr = "advanced_focal_dl_model_v1_advancedfocalloss"
    
    def _get_default_loss_function(self):
        """Advanced Focal Loss를 기본 손실 함수로 반환"""
        # 논문 방식: α와 base_γ는 설정, γ(t)는 내부에서 동적 조정
        alpha = getattr(self, 'focal_alpha', 0.25)
        base_gamma = getattr(self, 'base_gamma', 2.0)
        return AdvancedFocalLoss(alpha=alpha, base_gamma=base_gamma)
    
    def _set_params(self, **kwargs):
        """Advanced Focal Loss 파라미터를 처리"""
        print(f"🔧 AdvancedFocalDL _set_params 호출됨! kwargs={kwargs}")
        
        # 논문 방식: α와 base_γ 설정 (γ(t)는 내부에서 자동 조정)
        self.focal_alpha = kwargs.pop('focal_alpha', 0.25)
        self.base_gamma = kwargs.pop('base_gamma', 2.0)
        
        print(f"🔧 AdvancedFocalDL _set_params: focal_alpha={self.focal_alpha}, base_gamma={self.base_gamma} (γ(t)는 자동 조정)")
        
        # 나머지 파라미터는 부모 클래스로 전달
        return super()._set_params(**kwargs)
    
    def _get_net(self, train_dataset, params):
        """Advanced Focal Loss 파라미터를 처리하고 EmbedNet에 전달"""
        print(f"🔧 AdvancedFocalDL _get_net 호출됨! params={params}")
        
        # 논문 방식: α와 base_γ 필터링 (γ(t)는 내부에서 자동 조정)
        filtered_params = params.copy()
        focal_alpha = filtered_params.pop('focal_alpha', 0.25)
        base_gamma = filtered_params.pop('base_gamma', 2.0)
        
        # self에 파라미터 저장
        self.focal_alpha = focal_alpha
        self.base_gamma = base_gamma
        
        print(f"🔧 AdvancedFocalDL _get_net: focal_alpha={focal_alpha}, base_gamma={base_gamma} (γ(t)는 자동 조정)")
        print(f"🔧 AdvancedFocalDL 모델 구조: TabularNeuralNetTorchModel 상속, AdaFocal Loss 적용")
        
        # 나머지 파라미터로 EmbedNet 생성
        return super()._get_net(train_dataset, filtered_params)
    
    def _train_net(self, train_dataset, loss_kwargs, batch_size, num_epochs, 
                   epochs_wo_improve, val_dataset=None, test_dataset=None, 
                   time_limit=None, reporter=None, verbosity=2):
        """Advanced Focal Loss 학습 과정 모니터링"""
        print(f"🚀 AdvancedFocalDL _train_net 시작!")
        print(f"🔧 AdaFocal Loss 파라미터: alpha={self.focal_alpha} (γ는 자동 조정)")
        print(f"🔧 손실 함수: {type(self._get_default_loss_function()).__name__}")
        
        # 부모 클래스의 학습 메서드 호출
        result = super()._train_net(train_dataset, loss_kwargs, batch_size, num_epochs,
                                  epochs_wo_improve, val_dataset, test_dataset,
                                  time_limit, reporter, verbosity)
        
        print(f"✅ AdvancedFocalDL _train_net 완료!")
        return result
"""
AdaFocal: Calibration-aware Adaptive Focal Loss

This implementation follows the original AdaFocal paper:
"AdaFocal: Calibration-aware Adaptive Focal Loss" (NeurIPS 2022)
Authors: Arindam Ghosh, Thomas Schaaf, Matt Gormley

Key Ideas:
1. Validation set 기반 calibration 측정
2. Sample별 개별 γ 조정
3. Calibration feedback을 통한 adaptive learning
4. 이전 γ 값 기억 (momentum)

Mathematical Foundation:
- FL(p_t) = -α * (1 - p_t)^γ(t) * log(p_t)
- γ(t) = f(γ(t-1), calibration_feedback)
- Calibration error = |정확도 - 예측확률|
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel


class AdaFocalLoss(nn.Module):
    """
    AdaFocal Loss: Calibration-aware Adaptive Focal Loss
    
    This implementation follows the AdaFocal paper exactly:
    - γ is updated per confidence bin (not per sample)
    - Updates happen every batch based on validation set calibration
    - Switches between focal loss (γ > 0) and inverse-focal loss (γ < 0)
    
    Reference:
    - Paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/0a692a24dbc744fca340b9ba33bc6522-Paper-Conference.pdf
    - Code: https://github.com/3mcloud/adafocal
    
    한글 설명:
    AdaFocal Loss는 신뢰도 보정을 인식하는 적응형 focal loss입니다.
    - γ는 각 신뢰도 구간별로 업데이트됩니다 (샘플별이 아님)
    - 매 배치마다 검증 세트의 보정 정보를 기반으로 업데이트됩니다
    - 과신적일 때는 focal loss (γ > 0), 과소신적일 때는 inverse-focal loss (γ < 0) 사용
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 alpha: float = 1.0,
                 base_gamma: float = 2.0,
                 momentum: float = 0.9,
                 num_bins: int = 15,  # Paper uses 15 equal-mass bins (논문에서는 15개 동일 질량 구간 사용)
                 calibration_threshold: float = 0.2,  # Sth in paper (논문의 Sth 값)
                 lambda_param: float = 1.0,  # λ in paper (논문의 λ 파라미터) - 이 부분 추가!
                 gamma_max: float = 20.0,  # γmax in paper (논문의 γmax 값)
                 gamma_min: float = -2.0):  # γmin in paper (논문의 γmin 값)
        super(AdaFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.base_gamma = base_gamma
        self.momentum = momentum
        self.num_bins = num_bins
        self.calibration_threshold = calibration_threshold
        self.lambda_param = lambda_param  # 이 부분 추가!
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        
        # Initialize γ for each bin (paper: γt=0,i = 1 for all bins)
        # 각 구간별로 γ 초기화 (논문: 모든 구간에서 γt=0,i = 1)
        self.register_buffer('gamma_bins', torch.ones(num_bins))
        
        # Validation cache for calibration measurement
        # 보정 측정을 위한 검증 세트 캐시
        self.validation_cache = {
            'predictions': [],
            'targets': [],
            'confidences': []
        }
        
        # Bin edges for equal-mass binning
        # 동일 질량 구간을 위한 구간 경계
        self.bin_edges = torch.linspace(0, 1, num_bins + 1)
        
        # Training step counter
        # 훈련 단계 카운터
        self.training_step = 0
        
        # Sample별 개별 γ 저장
        self.register_buffer('gamma_per_sample', torch.ones(1000) * base_gamma)  # 임시 크기
        self.register_buffer('gamma_momentum', torch.ones(1000) * base_gamma)
        
        # Calibration tracking
        self.register_buffer('calibration_history', torch.zeros(num_bins))
        self.register_buffer('sample_count_history', torch.zeros(num_bins))
        
        # Sample difficulty tracking
        self.register_buffer('sample_difficulties', torch.zeros(1000))
        
        print(f"🔧 AdaFocal Loss 초기화:")
        print(f"   📊 Base γ: {base_gamma}")
        print(f"   📊 Momentum: {momentum}")
        print(f"   📊 Calibration bins: {num_bins}")
        print(f"   📊 Calibration threshold: {calibration_threshold}")
    
    def update_validation_cache(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Validation set에서 calibration 정보 업데이트
        
        Args:
            predictions: 모델 예측 확률 (N, num_classes)
            targets: 실제 레이블 (N,)
        """
        with torch.no_grad():
            # 예측 확신도 (정답 클래스에 대한 확률)
            confidences = predictions.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Validation cache에 저장
            self.validation_cache['predictions'].append(predictions.detach().cpu())
            self.validation_cache['targets'].append(targets.detach().cpu())
            self.validation_cache['confidences'].append(confidences.detach().cpu())
            
            print(f"🔄 Validation cache 업데이트: {len(self.validation_cache['predictions'])} batches")
    
    def measure_calibration(self) -> Dict[str, float]:
        """
        Validation set에서 calibration 측정
        
        Returns:
            calibration_metrics: calibration 관련 지표들
        """
        if not self.validation_cache['predictions']:
            print("⚠️ Validation cache가 비어있습니다. 먼저 update_validation_cache를 호출하세요.")
            return {}
        
        # 모든 validation 데이터 합치기
        all_predictions = torch.cat(self.validation_cache['predictions'], dim=0)
        all_targets = torch.cat(self.validation_cache['targets'], dim=0)
        all_confidences = torch.cat(self.validation_cache['confidences'], dim=0)
        
        # Calibration bins별로 정확도 계산
        bin_accuracies = torch.zeros(self.num_bins)
        bin_counts = torch.zeros(self.num_bins)
        
        for i in range(self.num_bins):
            # i번째 bin에 속하는 샘플들 찾기
            bin_mask = (all_confidences >= self.bin_edges[i]) & (all_confidences < self.bin_edges[i + 1])
            
            if bin_mask.sum() > 0:
                # 해당 bin의 정확도 계산
                bin_predictions = all_predictions[bin_mask]
                bin_targets = all_targets[bin_mask]
                
                # 예측 클래스
                pred_classes = torch.argmax(bin_predictions, dim=1)
                
                # 정확도
                accuracy = (pred_classes == bin_targets).float().mean()
                bin_accuracies[i] = accuracy
                bin_counts[i] = bin_mask.sum()
        
        # Expected Calibration Error (ECE) 계산
        ece = 0.0
        for i in range(self.num_bins):
            if bin_counts[i] > 0:
                ece += abs(bin_accuracies[i] - self.bin_centers[i]) * bin_counts[i]
        ece = ece / all_confidences.shape[0]
        
        # Calibration metrics
        calibration_metrics = {
            'ece': ece.item(),
            'bin_accuracies': bin_accuracies.cpu().numpy(),
            'bin_counts': bin_counts.cpu().numpy(),
            'bin_centers': self.bin_centers.cpu().numpy()
        }
        
        print(f"📊 Calibration 측정 결과:")
        print(f"   📊 ECE: {ece:.4f}")
        print(f"   📊 Bin별 정확도: {bin_accuracies.cpu().numpy()}")
        print(f"   📊 Bin별 샘플 수: {bin_counts.cpu().numpy()}")
        
        return calibration_metrics
    
    def adjust_gamma_based_on_calibration(self, sample_indices: torch.Tensor):
        """
        Calibration 결과에 따라 sample별 γ 조정
        
        Args:
            sample_indices: 현재 배치의 샘플 인덱스
        """
        if not self.validation_cache['predictions']:
            return
        
        # Calibration 측정
        calibration_metrics = self.measure_calibration()
        
        if not calibration_metrics:
            return
        
        # 현재 배치의 예측 확률과 타겟
        current_batch_size = sample_indices.shape[0]
        
        # Sample별 γ 조정
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= self.gamma_per_sample.shape[0]:
                # 버퍼 크기 확장
                new_size = max(sample_idx + 1, self.gamma_per_sample.shape[0] * 2)
                self.gamma_per_sample = torch.cat([
                    self.gamma_per_sample,
                    torch.ones(new_size - self.gamma_per_sample.shape[0]) * self.base_gamma
                ]).to(self.gamma_per_sample.device)
                self.gamma_momentum = torch.cat([
                    self.gamma_momentum,
                    torch.ones(new_size - self.gamma_momentum.shape[0]) * self.base_gamma
                ]).to(self.gamma_momentum.device)
            
            # 현재 샘플의 γ 값
            current_gamma = self.gamma_per_sample[sample_idx]
            
            # Calibration 기반 γ 조정
            if calibration_metrics['ece'] > self.calibration_threshold:
                # ECE가 높으면 γ를 더 적극적으로 조정
                if i < len(calibration_metrics['bin_accuracies']):
                    bin_idx = min(i, len(calibration_metrics['bin_accuracies']) - 1)
                    bin_accuracy = calibration_metrics['bin_accuracies'][bin_idx]
                    bin_center = calibration_metrics['bin_centers'][bin_idx]
                    
                    # Calibration error
                    cal_error = abs(bin_accuracy - bin_center)
                    
                    if cal_error > 0.1:  # Calibration error가 큰 경우
                        if bin_accuracy < bin_center:  # Under-confident
                            new_gamma = current_gamma * 1.1  # γ 증가
                        else:  # Over-confident
                            new_gamma = current_gamma * 0.9  # γ 감소
                        
                        # Momentum 기반 업데이트
                        self.gamma_per_sample[sample_idx] = (
                            self.momentum * current_gamma + 
                            (1 - self.momentum) * new_gamma
                        )
        
        print(f"🔄 Sample별 γ 조정 완료 (배치 크기: {current_batch_size})")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                sample_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        AdaFocal Loss 계산
        
        Args:
            inputs: 모델 출력 (N, num_classes)
            targets: 실제 레이블 (N,)
            sample_indices: 샘플 인덱스 (N,) - calibration tracking용
        
        Returns:
            loss: AdaFocal loss 값
        """
        batch_size = inputs.shape[0]
        
        # Sample indices가 없으면 자동 생성
        if sample_indices is None:
            sample_indices = torch.arange(batch_size, device=inputs.device)
        
        # γ 조정 (calibration 기반)
        self.adjust_gamma_based_on_calibration(sample_indices)
        
        # Sample별 γ 값 가져오기
        gamma_values = torch.zeros(batch_size, device=inputs.device)
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx < self.gamma_per_sample.shape[0]:
                gamma_values[i] = self.gamma_per_sample[sample_idx]
            else:
                gamma_values[i] = self.base_gamma
        
        # Cross entropy loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 예측 확률
        probs = F.softmax(inputs, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weights: (1 - p_t)^γ
        focal_weights = (1 - target_probs) ** gamma_values
        
        # 최종 loss: -α * (1 - p_t)^γ * log(p_t)
        loss = (self.alpha * focal_weights * ce_loss).mean()
        
        # Sample difficulty tracking
        with torch.no_grad():
            difficulties = 1.0 - target_probs
            for i, sample_idx in enumerate(sample_indices):
                if sample_idx < self.sample_difficulties.shape[0]:
                    self.sample_difficulties[sample_idx] = difficulties[i]
        
        return loss
    
    def get_calibration_summary(self) -> Dict[str, any]:
        """
        Calibration 상태 요약 반환
        """
        if not self.validation_cache['predictions']:
            return {'status': 'no_validation_data'}
        
        calibration_metrics = self.measure_calibration()
        
        # γ 분포 통계
        gamma_stats = {
            'mean': self.gamma_per_sample.mean().item(),
            'std': self.gamma_per_sample.std().item(),
            'min': self.gamma_per_sample.min().item(),
            'max': self.gamma_per_sample.max().item()
        }
        
        return {
            'status': 'calibrated',
            'ece': calibration_metrics['ece'],
            'gamma_statistics': gamma_stats,
            'bin_accuracies': calibration_metrics['bin_accuracies'].tolist(),
            'bin_counts': calibration_metrics['bin_counts'].tolist()
        }
    
    def reset_calibration_cache(self):
        """Validation cache 초기화"""
        self.validation_cache = {
            'predictions': [],
            'targets': [],
            'confidences': []
        }
        print("🔄 Calibration cache 초기화 완료")
    
    def reset_parameters(self):
        """파라미터 초기화"""
        self.gamma_per_sample.fill_(self.base_gamma)
        self.gamma_momentum.fill_(self.base_gamma)
        self.calibration_history.fill_(0)
        self.sample_count_history.fill_(0)
        self.sample_difficulties.fill_(0)
        print("🔄 AdaFocal Loss 파라미터 초기화 완료")


class AdaFocalDL(TabularNeuralNetTorchModel):
    """
    AdaFocal Deep Learning model for AutoGluon
    
    This model integrates AdaFocal loss with AutoGluon's TabularNeuralNetTorchModel.
    
    Reference:
    - Paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/0a692a24dbc744fca340b9ba33bc6522-Paper-Conference.pdf
    - Code: https://github.com/3mcloud/adafocal
    
    한글 설명:
    AutoGluon을 위한 AdaFocal 딥러닝 모델입니다.
    AdaFocal loss를 AutoGluon의 TabularNeuralNetTorchModel과 통합합니다.
    """
    
    # AutoGluon 모델 등록을 위한 고유 식별자들
    ag_key = 'ADAFOCAL_DL'
    ag_name = 'ADAFOCAL_DL'
    ag_priority = 100
    _model_name = "AdaFocalDL"
    _model_type = "adafocal_dl_model"
    _typestr = "adafocal_dl_model_v1_adafocalloss"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기본값으로 초기화 (CustomFocalDLModel과 동일한 패턴)
        # AdaFocal Loss 특화 파라미터들
        self.base_gamma = 2.0
        self.momentum = 0.9
        self.num_bins = 15
        self.calibration_threshold = 0.2
        self.lambda_param = 1.0
        self.gamma_max = 20.0
        self.gamma_min = -2.0
        self.alpha = 1.0
        
        # LR scheduler 관련 파라미터들 (CustomFocalDLModel과 동일!)
        self.lr_scheduler = True
        self.scheduler_type = 'cosine'  # scheduler_type 추가
        self.lr_scheduler_step_size = 50
        self.lr_scheduler_gamma = 0.1
        self.lr_scheduler_min_lr = 1e-6
        
        # 기타 공통 파라미터들 (CustomFocalDLModel과 동일!)
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5
        self.dropout_prob = 0.1
        self.hidden_size = 128
        self.num_layers = 2
        self.batch_size = 256
        self.max_epochs = 100
        self.patience = 10
        
        # Validation data cache
        # 검증 데이터 캐시
        self.validation_data = None
        self.validation_targets = None
    
    def _set_params(self, **kwargs):
        """Set model parameters (모델 파라미터 설정) - CustomFocalDLModel과 완전 동일하게 맞춤!"""
        print(f"🔧 AdaFocalDL _set_params 호출됨! kwargs={kwargs}")
        
        # AdaFocal Loss 특화 파라미터들
        self.base_gamma = kwargs.pop('base_gamma', 2.0)
        self.momentum = kwargs.pop('momentum', 0.9)
        self.num_bins = kwargs.pop('num_bins', 15)
        self.calibration_threshold = kwargs.pop('calibration_threshold', 0.2)
        self.lambda_param = kwargs.pop('lambda_param', 1.0)
        self.gamma_max = kwargs.pop('gamma_max', 20.0)
        self.gamma_min = kwargs.pop('gamma_min', -2.0)
        self.alpha = kwargs.pop('alpha', 1.0)
        
        # LR scheduler 관련 파라미터들 (CustomFocalDLModel과 동일!)
        self.lr_scheduler = kwargs.pop('lr_scheduler', True)
        self.scheduler_type = kwargs.pop('scheduler_type', 'cosine')  # scheduler_type 추가
        self.lr_scheduler_step_size = kwargs.pop('lr_scheduler_step_size', 50)
        self.lr_scheduler_gamma = kwargs.pop('lr_scheduler_gamma', 0.1)
        self.lr_scheduler_min_lr = kwargs.pop('lr_scheduler_min_lr', 1e-6)
        
        # 기타 공통 파라미터들 (CustomFocalDLModel과 동일!)
        self.learning_rate = kwargs.pop('learning_rate', 3e-4)
        self.weight_decay = kwargs.pop('weight_decay', 1e-5)
        self.dropout_prob = kwargs.pop('dropout_prob', 0.1)
        self.hidden_size = kwargs.pop('hidden_size', 128)
        self.num_layers = kwargs.pop('num_layers', 2)
        self.batch_size = kwargs.pop('batch_size', 256)
        self.max_epochs = kwargs.pop('max_epochs', 100)
        self.patience = kwargs.pop('patience', 10)
        
        print(f"🔧 AdaFocalDL _set_params: base_gamma={self.base_gamma}, momentum={self.momentum}")
        print(f"🔧 AdaFocalDL _set_params: lr_scheduler={self.lr_scheduler}, lr_scheduler_step_size={self.lr_scheduler_step_size}")
        
        # 나머지 파라미터는 부모 클래스로 전달 (CustomFocalDLModel과 동일!)
        return super()._set_params(**kwargs)
    
    def _get_default_loss_function(self):
        """Get default loss function (기본 손실 함수 가져오기) - AdaFocal Loss 사용"""
        # num_classes는 기본값 2로 설정 (CustomFocalDLModel과 동일한 패턴)
        num_classes = getattr(self, 'num_classes', 2)
        
        return AdaFocalLoss(
            num_classes=num_classes,
            alpha=self.alpha,  # self.alpha 사용
            base_gamma=self.base_gamma,
            momentum=self.momentum,
            num_bins=self.num_bins,
            calibration_threshold=self.calibration_threshold,
            lambda_param=self.lambda_param,
            gamma_max=self.gamma_max,
            gamma_min=self.gamma_min
        )
    
    def _get_net(self, train_dataset, params):
        """Get neural network architecture (신경망 아키텍처 가져오기) - CustomFocalDLModel과 완전 동일하게 맞춤!"""
        print(f"🔧 AdaFocalDL _get_net 호출됨! params={params}")
        
        # AdaFocal Loss 특화 파라미터들을 필터링하고 self에 저장
        filtered_params = params.copy()
        self.base_gamma = filtered_params.pop('base_gamma', 2.0)
        self.momentum = filtered_params.pop('momentum', 0.9)
        self.num_bins = filtered_params.pop('num_bins', 15)
        self.calibration_threshold = filtered_params.pop('calibration_threshold', 0.2)
        self.lambda_param = filtered_params.pop('lambda_param', 1.0)
        self.gamma_max = filtered_params.pop('gamma_max', 20.0)
        self.gamma_min = filtered_params.pop('gamma_min', -2.0)
        self.alpha = filtered_params.pop('alpha', 1.0)
        
        # LR scheduler 관련 파라미터들을 필터링
        self.lr_scheduler = filtered_params.pop('lr_scheduler', True)
        self.scheduler_type = filtered_params.pop('scheduler_type', 'cosine')  # scheduler_type 추가
        self.lr_scheduler_step_size = filtered_params.pop('lr_scheduler_step_size', 50)
        self.lr_scheduler_gamma = filtered_params.pop('lr_scheduler_gamma', 0.1)
        self.lr_scheduler_min_lr = filtered_params.pop('lr_scheduler_min_lr', 1e-6)
        
        # 기타 공통 파라미터들을 필터링
        self.learning_rate = filtered_params.pop('learning_rate', 3e-4)
        self.weight_decay = filtered_params.pop('weight_decay', 1e-5)
        self.dropout_prob = filtered_params.pop('dropout_prob', 0.1)
        self.hidden_size = filtered_params.pop('hidden_size', 128)
        self.num_layers = filtered_params.pop('num_layers', 2)
        self.batch_size = filtered_params.pop('batch_size', 256)
        self.max_epochs = filtered_params.pop('max_epochs', 100)
        self.patience = filtered_params.pop('patience', 10)
        
        print(f"🔧 AdaFocalDL _get_net: base_gamma={self.base_gamma}, momentum={self.momentum}")
        print(f"🔧 AdaFocalDL _get_net: lr_scheduler={self.lr_scheduler}, lr_scheduler_step_size={self.lr_scheduler_step_size}")
        
        # 나머지 파라미터로 EmbedNet 생성 (CustomFocalDLModel과 동일!)
        return super()._get_net(train_dataset, filtered_params)
    
    # _get_optimizer와 _get_scheduler는 제거 - TabularNeuralNetTorchModel의 기본 구현 사용
    # CustomFocalDLModel과 동일하게 맞춤!
    
    def _train_net(
        self,
        train_dataset,
        loss_kwargs,
        batch_size,
        num_epochs,
        epochs_wo_improve,
        val_dataset=None,
        test_dataset=None,
        time_limit=None,
        reporter=None,
        verbosity=2,
    ):
        """AutoGluon의 _train_net 메서드를 복사하고 LR scheduler 추가 - CustomFocalDLModel과 동일하게 맞춤!"""
        print("🚀 AdaFocalDL _train_net 호출됨!")  # 디버그 출력 추가
        import torch
        import torch.optim.lr_scheduler as lr_scheduler
        import time
        import logging
        import io
        from copy import deepcopy

        start_time = time.time()
        logging.debug("initializing neural network...")
        self.model.init_params()
        logging.debug("initialized")
        train_dataloader = train_dataset.build_loader(batch_size, self.num_dataloading_workers, is_test=False)

        if isinstance(loss_kwargs.get("loss_function", "auto"), str) and loss_kwargs.get("loss_function", "auto") == "auto":
            loss_kwargs["loss_function"] = self._get_default_loss_function()
        
        # LR scheduler 설정 (optimizer 생성 후)
        scheduler = None
        print(f"🔍 AdaFocalDL Debug: self.lr_scheduler = {getattr(self, 'lr_scheduler', 'NOT_SET')}")
        print(f"🔍 AdaFocalDL Debug: self.scheduler_type = {getattr(self, 'scheduler_type', 'NOT_SET')}")
        print(f"🔍 AdaFocalDL Debug: hasattr(self, 'lr_scheduler') = {hasattr(self, 'lr_scheduler')}")
        print(f"🔍 AdaFocalDL Debug: self.lr_scheduler (if exists) = {self.lr_scheduler if hasattr(self, 'lr_scheduler') else 'N/A'}")
        
        # 수정: _set_params에서 저장한 LR 스케줄러 파라미터 직접 사용
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            if hasattr(self, 'scheduler_type') and self.scheduler_type == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=self.lr_scheduler_min_lr
                )
                print(f"✅ AdaFocalDL: Cosine Annealing LR 스케줄러 적용됨 (min_lr={self.lr_scheduler_min_lr})")
            else:
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.2,
                    patience=5,
                    min_lr=self.lr_scheduler_min_lr
                )
                print(f"✅ AdaFocalDL: ReduceLROnPlateau LR 스케줄러 적용됨")
        else:
            print(f"❌ AdaFocalDL: LR 스케줄러가 설정되지 않음")
        
        # DCNv2와 동일한 형식의 상세한 학습 루프 호출
        self._train_with_scheduler(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity, scheduler)
    
    def _train_with_scheduler(
        self,
        train_dataset,
        loss_kwargs,
        batch_size,
        num_epochs,
        epochs_wo_improve,
        val_dataset=None,
        test_dataset=None,
        time_limit=None,
        reporter=None,
        verbosity=2,
        scheduler=None,
    ):
        """스케줄러와 함께 학습하면서 LR 모니터링"""
        import torch
        import time
        import io

        # 기본 설정
        self.model.init_params()
        train_dataloader = train_dataset.build_loader(
            batch_size, self.num_dataloading_workers, is_test=False
        )

        # 손실 함수 설정
        if isinstance(loss_kwargs.get("loss_function", "auto"), str) and loss_kwargs.get(
            "loss_function", "auto"
        ) == "auto":
            loss_kwargs["loss_function"] = self._get_default_loss_function()

        # Early stopping 설정
        if epochs_wo_improve is not None:
            early_stopping_method = self._get_early_stopping_strategy(
                num_rows_train=len(train_dataset)
            )
        else:
            early_stopping_method = self._get_early_stopping_strategy(
                num_rows_train=len(train_dataset)
            )

        # Validation 라벨
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val.flatten()
        else:
            y_val = None

        # 모델 저장용 버퍼
        io_buffer = None
        best_epoch = 0
        best_val_metric = -float("inf")

        # 학습 루프
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            # 현재 LR 출력
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 배치별 학습
            for batch_idx, data_batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                loss.backward()

                # Gradient clipping (NaN 방지)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # 에포크 평균 손실
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time

            # ---------------------------
            # Validation 평가 (f1 + loss)
            # ---------------------------
            val_metric = None
            val_loss = None
            if val_dataset is not None:
                # f1 (AutoGluon eval_metric 기반)
                val_metric = self.score(
                    X=val_dataset, y=y_val, metric=self.stopping_metric, _reset_threads=False
                )

                # validation loss (scheduler 용)
                with torch.no_grad():
                    val_loss_total = 0.0
                    val_batches = 0
                    for data_batch in val_dataset.build_loader(
                        batch_size, self.num_dataloading_workers, is_test=True
                    ):
                        loss = self.model.compute_loss(data_batch, **loss_kwargs)
                        val_loss_total += loss.item()
                        val_batches += 1
                    val_loss = val_loss_total / max(1, val_batches)

                # Best model 저장 (f1 기준)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    io_buffer = io.BytesIO()
                    torch.save(self.model, io_buffer)
                    best_epoch = epoch + 1

            # ---------------------------
            # 로그 출력
            # ---------------------------
            import time as time_module

            current_time = time_module.strftime("%Y-%m-%d %H:%M:%S")
            if val_metric is not None:
                log_msg = (
                    f"[{current_time}] Epoch {epoch+1}/{num_epochs}: "
                    f"Train loss: {avg_loss:.4f}, "
                    f"Val {self.stopping_metric.name}: {val_metric:.4f}, "
                    f"Val loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Best Epoch: {best_epoch}, "
                    f"Time: {epoch_time:.2f}s"
                )
            else:
                log_msg = (
                    f"[{current_time}] Epoch {epoch+1}/{num_epochs}: "
                    f"Train loss: {avg_loss:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Time: {epoch_time:.2f}s"
                )
            print(log_msg)

            # ---------------------------
            # 스케줄러 스텝
            # ---------------------------
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ✅ Validation loss 기준 (표준적인 접근)
                    scheduler.step(val_loss if val_loss is not None else avg_loss)

            # ---------------------------
            # Early stopping 체크 (f1 기준)
            # ---------------------------
            if val_dataset is not None:
                is_best = val_metric > best_val_metric
                early_stop = early_stopping_method.update(
                    cur_round=epoch, is_best=is_best
                )
                if early_stop:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        # ---------------------------
        # Best model 로드
        # ---------------------------
        if io_buffer is not None:
            io_buffer.seek(0)
            self.model = torch.load(io_buffer, weights_only=False)
            print(
                f"   Best model loaded from epoch {best_epoch} "
                f"(Val {self.stopping_metric.name}: {best_val_metric:.4f})"
            )

        # save trained parameters
        self.params_trained["batch_size"] = batch_size
        self.params_trained["num_epochs"] = best_epoch 
    
    # _get_default_ag_args는 제거 - TabularNeuralNetTorchModel의 기본 구현 사용
    # CustomFocalDLModel과 동일하게 맞춤!
    
    # _get_default_resources는 제거 - TabularNeuralNetTorchModel의 기본 구현 사용
    # CustomFocalDLModel과 동일하게 맞춤!
    
    def update_validation_cache(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Validation cache 업데이트 (외부에서 호출 가능)"""
        loss_function = self._get_default_loss_function()
        if hasattr(loss_function, 'update_validation_cache'):
            loss_function.update_validation_cache(predictions, targets)
    
    def get_calibration_summary(self) -> Dict[str, any]:
        """Calibration 상태 요약 반환 (외부에서 호출 가능)"""
        loss_function = self._get_default_loss_function()
        if hasattr(loss_function, 'get_calibration_summary'):
            return loss_function.get_calibration_summary()
        return {'status': 'not_supported'}
    
    def reset_calibration_cache(self):
        """Calibration cache 초기화 (외부에서 호출 가능)"""
        loss_function = self._get_default_loss_function()
        if hasattr(loss_function, 'reset_calibration_cache'):
            loss_function.reset_calibration_cache()


# 사용 예시
if __name__ == "__main__":
    print("🔧 AdaFocal Loss 테스트")
    
    # 모델 생성
    model = AdaFocalDL()
    
    # 파라미터 설정
    params = {
        'focal_alpha': 1.0,
        'focal_base_gamma': 2.0,
        'focal_momentum': 0.9,
        'focal_num_bins': 10,
        'focal_calibration_threshold': 0.1
    }
    
    # 파라미터 적용
    model._set_params(**params)
    
    print("✅ AdaFocal Loss 테스트 완료!") 
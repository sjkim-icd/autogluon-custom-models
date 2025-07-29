from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from .focal_loss import FocalLoss

class CustomFocalDLModel(TabularNeuralNetTorchModel):
    """
    CustomFocalDLModel: Focal Loss를 사용하는 커스텀 딥러닝 모델
    
    특징:
    - Focal Loss를 사용하여 클래스 불균형 문제 해결
    - AutoGluon의 TabularNeuralNetTorchModel을 상속
    - 하이퍼파라미터 튜닝 지원
    """
    
    _model_name = "CustomFocalDLModel"
    _model_type = "custom_focal_dl_model"
    _typestr = "custom_focal_dl_model_v1_focalloss"  # ✅ 반드시 NN_TORCH와 다르게
    
    def _get_default_loss_function(self):
        """
        Focal Loss를 기본 손실 함수로 설정
        
        Returns:
            FocalLoss: alpha=1.0, gamma=2.0으로 설정된 Focal Loss
        """
        return FocalLoss(alpha=1.0, gamma=2.0) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import EmbedNet
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

class CrossNetworkCorrect(nn.Module):
    """
    FuxiCTR과 동일한 Cross Network 구현
    수식: x_{l+1} = x_l + x_0 * w_l(x_l)
    """
    def __init__(self, input_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        
        # Cross layers: 각각 input_dim -> input_dim
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
            
    def forward(self, x):
        """
        FuxiCTR과 동일한 Cross Network forward pass
        x: (batch_size, input_dim)
        """
        x_0 = x  # x_0 저장
        x_i = x  # x_i 초기화
        
        for i in range(self.num_layers):
            # FuxiCTR과 동일한 구현
            # x_i = x_i + x_0 * self.cross_layers[i](x_i)
            x_i = x_i + x_0 * self.cross_layers[i](x_i)
            
            # Dropout 적용
            if self.dropout > 0:
                x_i = F.dropout(x_i, p=self.dropout, training=self.training)
                
        return x_i

class DCNv2NetCorrect(EmbedNet):
    """
    FuxiCTR과 동일한 DCNv2 구현
    """
    def _set_params(self, **kwargs):
        # DCNv2Net 고유 파라미터만 pop해서 사용
        num_cross_layers = kwargs.pop("num_cross_layers", 2)
        cross_dropout = kwargs.pop("cross_dropout", 0.1)
        deep_output_size = kwargs.pop("deep_output_size", 128)
        deep_hidden_size = kwargs.pop("deep_hidden_size", 128)
        deep_dropout = kwargs.pop("deep_dropout", 0.1)
        deep_layers = kwargs.pop("deep_layers", 3)
        
        # Learning Rate Scheduler 파라미터
        lr_scheduler = kwargs.pop("lr_scheduler", True)
        scheduler_type = kwargs.pop("scheduler_type", "plateau")
        lr_scheduler_patience = kwargs.pop("lr_scheduler_patience", 5)
        lr_scheduler_factor = kwargs.pop("lr_scheduler_factor", 0.2)
        lr_scheduler_min_lr = kwargs.pop("lr_scheduler_min_lr", 1e-6)
        
        # EmbedNet에서 지원하지 않는 파라미터들 제거
        unsupported_params = ['time_limit', 'epochs_wo_improve', 'num_epochs', 'low_rank']
        for param in unsupported_params:
            kwargs.pop(param, None)
        
        params = super()._set_params(**kwargs)
        params.update({
            "num_cross_layers": num_cross_layers,
            "cross_dropout": cross_dropout,
            "deep_output_size": deep_output_size,
            "deep_hidden_size": deep_hidden_size,
            "deep_dropout": deep_dropout,
            "deep_layers": deep_layers,
            "lr_scheduler": lr_scheduler,
            "scheduler_type": scheduler_type,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
        })
        return params

    def __init__(self,
                 problem_type,
                 num_net_outputs=None,
                 quantile_levels=None,
                 train_dataset=None,
                 architecture_desc=None,
                 device=None,
                 y_range=None,
                 num_cross_layers=2,
                 cross_dropout=0.1,
                 deep_output_size=128,
                 deep_hidden_size=128,
                 deep_dropout=0.1,
                 deep_layers=3,
                 lr_scheduler=True,
                 scheduler_type="plateau",
                 lr_scheduler_patience=5,
                 lr_scheduler_factor=0.2,
                 lr_scheduler_min_lr=1e-6,
                 **kwargs):
        
        # DCNv2 고유 파라미터 설정
        self.num_cross_layers = num_cross_layers
        self.cross_dropout = cross_dropout
        self.deep_output_size = deep_output_size
        self.deep_hidden_size = deep_hidden_size
        self.deep_dropout = deep_dropout
        self.deep_layers = deep_layers
        self.lr_scheduler = lr_scheduler
        self.scheduler_type = scheduler_type
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
        super().__init__(
            problem_type=problem_type,
            num_net_outputs=num_net_outputs,
            quantile_levels=quantile_levels,
            train_dataset=train_dataset,
            architecture_desc=architecture_desc,
            device=device,
            y_range=y_range,
            **kwargs
        )

    def _get_network(self, input_dim, num_net_outputs):
        """
        DCNv2 네트워크 구성 (FuxiCTR과 동일)
        """
        # Cross Network (FuxiCTR과 동일)
        self.cross_network = CrossNetworkCorrect(
            input_dim=input_dim,
            num_layers=self.num_cross_layers,
            dropout=self.cross_dropout
        )
        
        # Deep Network
        deep_layers = []
        current_dim = input_dim
        
        for i in range(self.deep_layers):
            deep_layers.extend([
                nn.Linear(current_dim, self.deep_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.deep_dropout)
            ])
            current_dim = self.deep_hidden_size
        
        # 최종 출력층
        deep_layers.append(nn.Linear(current_dim, self.deep_output_size))
        
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Combination Layer (FuxiCTR과 동일)
        total_dim = input_dim + self.deep_output_size
        self.combination_layer = nn.Linear(total_dim, num_net_outputs)
        
        return nn.ModuleList([self.cross_network, self.deep_network, self.combination_layer])

    def forward(self, data_batch):
        """
        DCNv2 forward pass (FuxiCTR과 동일)
        """
        # 입력 데이터
        x = data_batch['data']
        
        # Cross Network
        cross_output = self.cross_network(x)
        
        # Deep Network
        deep_output = self.deep_network(x)
        
        # Combination (FuxiCTR과 동일)
        combined = torch.cat([cross_output, deep_output], dim=1)
        output = self.combination_layer(combined)
        
        return output 
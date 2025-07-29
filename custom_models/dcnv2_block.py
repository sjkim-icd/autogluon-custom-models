import torch
import torch.nn as nn
import torch.nn.functional as F
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import EmbedNet
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, dropout=0.1, low_rank=32):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.low_rank = low_rank
        
        # Low-rank 구조: W = U * V^T (DCNv2 논문의 핵심)
        self.U_layers = nn.ModuleList()
        self.V_layers = nn.ModuleList()
        for i in range(num_layers):
            self.U_layers.append(nn.Linear(input_dim, low_rank, bias=False))
            self.V_layers.append(nn.Linear(input_dim, low_rank, bias=False))
            
    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            # Low-rank 구조: W = U * V^T
            U_out = self.U_layers[i](x)  # (batch_size, low_rank)
            V_out = self.V_layers[i](x)  # (batch_size, low_rank)
            xw = torch.sum(U_out * V_out, dim=1, keepdim=True)  # (batch_size, 1)
            x = x0 * xw + x  # (batch_size, input_dim)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DCNv2Net(EmbedNet):
    def _set_params(self, **kwargs):
        # DCNv2Net 고유 파라미터만 pop해서 사용
        num_cross_layers = kwargs.pop("num_cross_layers", 2)
        cross_dropout = kwargs.pop("cross_dropout", 0.1)
        low_rank = kwargs.pop("low_rank", 32)  # DCNv2 논문의 핵심 하이퍼파라미터
        deep_output_size = kwargs.pop("deep_output_size", 128)  # 하이퍼파라미터로 추가
        deep_hidden_size = kwargs.pop("deep_hidden_size", 128)  # 하이퍼파라미터로 추가
        deep_dropout = kwargs.pop("deep_dropout", 0.1)  # Deep Network 드롭아웃
        deep_layers = kwargs.pop("deep_layers", 3)  # Deep Network 레이어 수
        
        # Learning Rate Scheduler 파라미터
        lr_scheduler = kwargs.pop("lr_scheduler", True)
        scheduler_type = kwargs.pop("scheduler_type", "plateau")
        lr_scheduler_patience = kwargs.pop("lr_scheduler_patience", 5)
        lr_scheduler_factor = kwargs.pop("lr_scheduler_factor", 0.2)
        lr_scheduler_min_lr = kwargs.pop("lr_scheduler_min_lr", 1e-6)
        
        params = super()._set_params(**kwargs)
        params.update({
            "num_cross_layers": num_cross_layers,
            "cross_dropout": cross_dropout,
            "low_rank": low_rank,
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
                 low_rank=32,  # DCNv2 논문의 핵심 하이퍼파라미터
                 deep_output_size=128,  # 하이퍼파라미터로 추가
                 deep_hidden_size=128,  # 하이퍼파라미터로 추가
                 deep_dropout=0.1,  # Deep Network 드롭아웃
                 deep_layers=3,  # Deep Network 레이어 수
                 lr_scheduler=True,  # Learning Rate Scheduler 사용 여부
                 scheduler_type="plateau",  # 스케줄러 타입
                 lr_scheduler_patience=5,  # ReduceLROnPlateau patience
                 lr_scheduler_factor=0.2,  # ReduceLROnPlateau factor
                 lr_scheduler_min_lr=1e-6,  # 최소 learning rate
                 **kwargs):
        super().__init__(problem_type=problem_type,
                         num_net_outputs=num_net_outputs,
                         quantile_levels=quantile_levels,
                         train_dataset=train_dataset,
                         architecture_desc=architecture_desc,
                         device=device,
                         y_range=y_range,
                         **kwargs)
        
        self.num_cross_layers = num_cross_layers
        self.cross_dropout = cross_dropout
        self.low_rank = low_rank  # 저장
        self.deep_output_size = deep_output_size  # 저장
        self.deep_hidden_size = deep_hidden_size  # 저장
        self.deep_dropout = deep_dropout  # 저장
        self.deep_layers = deep_layers  # 저장
        
        # Learning Rate Scheduler 파라미터 저장
        self.lr_scheduler = lr_scheduler
        self.scheduler_type = scheduler_type
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
        # EmbedNet 초기화 완료 후 input_size 계산
        input_size = 0
        
        # Embedding features 처리
        if self.has_embed_features:
            for embed_block in self.embed_blocks:
                input_size += embed_block.embedding_dim
        
        # Vector features 처리
        if self.has_vector_features:
            input_size += train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        
        # Cross Network 추가 (low_rank 포함)
        self.cross_network = CrossNetwork(input_size, num_cross_layers, cross_dropout, self.low_rank)
        
        # Deep Network - 동적으로 레이어 생성
        deep_layers_list = []
        # 첫 번째 레이어
        deep_layers_list.append(nn.Linear(input_size, self.deep_hidden_size))
        deep_layers_list.append(nn.ReLU())
        deep_layers_list.append(nn.Dropout(self.deep_dropout))
        
        # 중간 레이어들
        for _ in range(self.deep_layers - 1):
            deep_layers_list.append(nn.Linear(self.deep_hidden_size, self.deep_hidden_size))
            deep_layers_list.append(nn.ReLU())
            deep_layers_list.append(nn.Dropout(self.deep_dropout))
        
        # 마지막 출력 레이어
        deep_layers_list.append(nn.Linear(self.deep_hidden_size, self.deep_output_size))
        
        self.deep_network = nn.Sequential(*deep_layers_list)
        
        # Final combination layer
        # Cross(input_size) + Deep(deep_output_size)
        combination_input_size = input_size + self.deep_output_size
        self.combination_layer = nn.Linear(combination_input_size, num_net_outputs)

    def forward(self, data_batch):
        # EmbedNet의 forward 로직을 그대로 사용
        input_data = []
        input_offset = 0
        if self.has_vector_features:
            input_data.append(data_batch[0].to(self.device))
            input_offset += 1
        if self.has_embed_features:
            embed_data = data_batch[input_offset]
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i].to(self.device)))

        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]

        # DCNv2 로직 적용
        cross_output = self.cross_network(input_data)  # 원본 크기 유지 (30차원)
        deep_output = self.deep_network(input_data)    # 128차원
        
        # Combine cross and deep outputs
        combined = torch.cat([cross_output, deep_output], dim=1)  # [30 + 128 = 158차원]
        output_data = self.combination_layer(combined)
        
        # EmbedNet의 출력 처리 로직
        if self.problem_type in [REGRESSION, QUANTILE]:
            if self.y_constraint is None:
                return output_data
            else:
                if self.y_constraint == "nonnegative":
                    return self.y_lower + torch.abs(output_data)
                elif self.y_constraint == "nonpositive":
                    return self.y_upper - torch.abs(output_data)
                else:
                    return torch.sigmoid(output_data) * self.y_span + self.y_lower
        elif self.problem_type == SOFTCLASS:
            return self.log_softmax(output_data)
        elif self.problem_type in [BINARY, MULTICLASS]:
            return self.softmax(output_data)  # Binary/Multiclass 분류를 위한 Softmax
        else:
            return output_data 
import torch
import torch.nn as nn
import torch.nn.functional as F
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import EmbedNet
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

class CrossNetV2FuxiCTR(nn.Module):
    """FuxiCTR 원본과 동일한 CrossNetV2 구현 (Low-rank 구조)"""
    def __init__(self, input_dim, num_layers, dropout=0.1, low_rank=32):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.low_rank = low_rank
        
        # FuxiCTR 원본과 동일: Low-rank 구조 사용
        self.U_layers = nn.ModuleList()
        self.V_layers = nn.ModuleList()
        for i in range(num_layers):
            self.U_layers.append(nn.Linear(input_dim, low_rank, bias=False))
            self.V_layers.append(nn.Linear(input_dim, low_rank, bias=False))
            
    def forward(self, X_0):
        """
        FuxiCTR 원본과 동일한 forward pass (Low-rank 구조)
        X_0: (batch_size, input_dim)
        """
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            # Low-rank 구조: W = U * V^T
            U_out = self.U_layers[i](X_i)  # (batch_size, low_rank)
            V_out = self.V_layers[i](X_i)  # (batch_size, low_rank)
            xw = torch.sum(U_out * V_out, dim=1, keepdim=True)  # (batch_size, 1)
            X_i = X_i + X_0 * xw  # (batch_size, input_dim)
            if self.dropout > 0:
                X_i = F.dropout(X_i, p=self.dropout, training=self.training)
        return X_i

class CrossNetMixFuxiCTR(nn.Module):
    """FuxiCTR 원본과 동일한 CrossNetMix 구현"""
    def __init__(self, input_dim, num_layers, dropout=0.1, low_rank=32, num_experts=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.low_rank = low_rank
        self.num_experts = num_experts
        
        # FuxiCTR 원본과 동일: Mixture of Experts 구조
        self.U_layers = nn.ModuleList()
        self.V_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 각 expert에 대한 U, V 행렬
            U_experts = nn.ModuleList()
            V_experts = nn.ModuleList()
            for j in range(num_experts):
                U_experts.append(nn.Linear(input_dim, low_rank, bias=False))
                V_experts.append(nn.Linear(input_dim, low_rank, bias=False))
            self.U_layers.append(U_experts)
            self.V_layers.append(V_experts)
            
            # Gate network
            self.gate_layers.append(nn.Linear(input_dim, num_experts))
            
    def forward(self, X_0):
        """
        FuxiCTR 원본과 동일한 forward pass
        X_0: (batch_size, input_dim)
        """
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            # Gate 계산
            gate_weights = F.softmax(self.gate_layers[i](X_i), dim=1)  # (batch_size, num_experts)
            
            # 각 expert의 출력 계산
            expert_outputs = []
            for j in range(self.num_experts):
                U_out = self.U_layers[i][j](X_i)  # (batch_size, low_rank)
                V_out = self.V_layers[i][j](X_i)  # (batch_size, low_rank)
                expert_output = torch.sum(U_out * V_out, dim=1, keepdim=True)  # (batch_size, 1)
                expert_outputs.append(expert_output)
            
            # Expert 출력들을 결합
            expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch_size, num_experts)
            xw = torch.sum(expert_outputs * gate_weights, dim=1, keepdim=True)  # (batch_size, 1)
            
            X_i = X_i + X_0 * xw  # FuxiCTR 원본과 동일한 구조
            if self.dropout > 0:
                X_i = F.dropout(X_i, p=self.dropout, training=self.training)
        return X_i

class DCNv2NetFuxiCTR(EmbedNet):
    # AutoGluon 모델 등록을 위한 속성들
    ag_key = "DCNV2_FUXICTR"
    ag_name = "DCNV2_FUXICTR"
    ag_priority = 100
    
    def _set_params(self, **kwargs):
        # DCNv2Net 고유 파라미터만 pop해서 사용
        num_cross_layers = kwargs.pop("num_cross_layers", 2)
        cross_dropout = kwargs.pop("cross_dropout", 0.1)
        low_rank = kwargs.pop("low_rank", 32)
        use_low_rank_mixture = kwargs.pop("use_low_rank_mixture", False)
        num_experts = kwargs.pop("num_experts", 4)
        model_structure = kwargs.pop("model_structure", "parallel")
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
        unsupported_params = ['time_limit', 'epochs_wo_improve', 'num_epochs']
        for param in unsupported_params:
            kwargs.pop(param, None)
        
        params = super()._set_params(**kwargs)
        params.update({
            "num_cross_layers": num_cross_layers,
            "cross_dropout": cross_dropout,
            "low_rank": low_rank,
            "use_low_rank_mixture": use_low_rank_mixture,
            "num_experts": num_experts,
            "model_structure": model_structure,
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
                 low_rank=32,
                 use_low_rank_mixture=False,
                 num_experts=4,
                 model_structure="parallel",
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
        self.low_rank = low_rank
        self.use_low_rank_mixture = use_low_rank_mixture
        self.num_experts = num_experts
        self.model_structure = model_structure
        self.deep_output_size = deep_output_size
        self.deep_hidden_size = deep_hidden_size
        self.deep_dropout = deep_dropout
        self.deep_layers = deep_layers
        
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
        
        # Cross Network 추가 (FuxiCTR 스타일)
        if self.use_low_rank_mixture:
            self.cross_network = CrossNetMixFuxiCTR(
                input_size, num_cross_layers, cross_dropout, self.low_rank, self.num_experts
            )
        else:
            self.cross_network = CrossNetV2FuxiCTR(
                input_size, num_cross_layers, cross_dropout, self.low_rank
            )
        
        # Deep Network - 동적으로 레이어 생성
        def create_deep_network(input_dim):
            deep_layers_list = []
            # 첫 번째 레이어
            deep_layers_list.append(nn.Linear(input_dim, self.deep_hidden_size))
            deep_layers_list.append(nn.ReLU())
            deep_layers_list.append(nn.Dropout(self.deep_dropout))
            
            # 중간 레이어들
            for _ in range(self.deep_layers - 1):
                deep_layers_list.append(nn.Linear(self.deep_hidden_size, self.deep_hidden_size))
                deep_layers_list.append(nn.ReLU())
                deep_layers_list.append(nn.Dropout(self.deep_dropout))
            
            # 마지막 출력 레이어
            deep_layers_list.append(nn.Linear(self.deep_hidden_size, self.deep_output_size))
            
            return nn.Sequential(*deep_layers_list)
        
        # Parallel Deep Network (원본 입력용) - parallel 구조용
        self.parallel_deep_network = create_deep_network(input_size)
        
        # Stacked Deep Network (CrossNet 출력용) - stacked, stacked_parallel 구조용
        self.stacked_deep_network = create_deep_network(input_size)
        
        # FuxiCTR 스타일의 유연한 구조에 따른 최종 출력층
        if self.model_structure == "crossnet_only":
            combination_input_size = input_size
        elif self.model_structure == "stacked":
            combination_input_size = self.deep_output_size
        elif self.model_structure == "parallel":
            combination_input_size = input_size + self.deep_output_size
        elif self.model_structure == "stacked_parallel":
            combination_input_size = self.deep_output_size + self.deep_output_size  # stacked_dnn + parallel_dnn
        else:
            combination_input_size = input_size + self.deep_output_size  # 기본값
        
        self.combination_layer = nn.Linear(combination_input_size, num_net_outputs)

    def forward(self, data_batch):
        """
        FuxiCTR 스타일의 DCNv2 forward pass - AutoGluon 데이터 형식에 맞게 수정
        """
        # EmbedNet의 forward 로직을 그대로 사용 (원래 잘 작동하는 방식)
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
        
        # Cross Network
        cross_output = self.cross_network(input_data)
        
        # FuxiCTR 스타일의 유연한 구조
        if self.model_structure == "crossnet_only":
            final_output = cross_output
        elif self.model_structure == "stacked":
            # FuxiCTR 원본: stacked_dnn(cross_output)
            final_output = self.stacked_deep_network(cross_output)
        elif self.model_structure == "parallel":
            # FuxiCTR 원본: cross_out + parallel_dnn(x)
            parallel_deep_output = self.parallel_deep_network(input_data)
            final_output = torch.cat([cross_output, parallel_deep_output], dim=1)
        elif self.model_structure == "stacked_parallel":
            # FuxiCTR 원본: stacked_dnn(cross_out) + parallel_dnn(x)
            stacked_deep_output = self.stacked_deep_network(cross_output)  # stacked_dnn(cross_out)
            parallel_deep_output = self.parallel_deep_network(input_data)   # parallel_dnn(x)
            final_output = torch.cat([stacked_deep_output, parallel_deep_output], dim=1)
        else:
            # 기본값: parallel 구조
            parallel_deep_output = self.parallel_deep_network(input_data)
            final_output = torch.cat([cross_output, parallel_deep_output], dim=1)
        
        output_data = self.combination_layer(final_output)
        
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
            return output_data  # Softmax는 AutoGluon의 predict에서 처리
        else:
            return output_data 
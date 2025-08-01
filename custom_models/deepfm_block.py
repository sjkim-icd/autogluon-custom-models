# [my_models] DeepFM 구조 정의
import torch
import torch.nn as nn
import torch.nn.functional as F

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, SOFTCLASS
from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import EmbedNet


class FMFirstOrderLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class FMSecondOrderLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, field_embeddings):
        # field_embeddings: (batch_size, num_fields, embed_dim)
        sum_square = torch.sum(field_embeddings, dim=1) ** 2  # (batch_size, embed_dim)
        square_sum = torch.sum(field_embeddings ** 2, dim=1)  # (batch_size, embed_dim)
        second_order = 0.5 * (sum_square - square_sum)  # (batch_size, embed_dim)
        return torch.sum(second_order, dim=1, keepdim=True)  # (batch_size, 1)


class DeepFMNet(EmbedNet):
    def _set_params(self, **kwargs):
        # DeepFMNet 고유 파라미터만 pop해서 사용
        fm_dropout = kwargs.pop("fm_dropout", 0.2)
        fm_embedding_dim = kwargs.pop("fm_embedding_dim", 10)
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
            "fm_dropout": fm_dropout,
            "fm_embedding_dim": fm_embedding_dim,
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
                 fm_dropout=0.2,
                 fm_embedding_dim=10,
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
        
        self.fm_dropout = fm_dropout
        self.fm_embedding_dim = fm_embedding_dim
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
        
        # 논문에 맞는 구현: 범주형만 임베딩, 수치형은 원본 사용
        self.field_embeddings = nn.ModuleList()
        self.field_sizes = []
        
        # Embedding features 처리 - 범주형만 임베딩 테이블 생성
        if self.has_embed_features:
            for embed_block in self.embed_blocks:
                field_size = embed_block.num_embeddings
                self.field_sizes.append(field_size)
                self.field_embeddings.append(
                    nn.Embedding(field_size, self.fm_embedding_dim)
                )
        
        # FM Components 계산
        total_input_size = 0
        if self.has_embed_features:
            total_input_size += sum(self.field_sizes)
        if self.has_vector_features:
            total_input_size += train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        
        self.fm_first = FMFirstOrderLayer(total_input_size)
        
        # FM Second Order는 범주형 임베딩만 사용
        if self.has_embed_features:
            self.fm_second = FMSecondOrderLayer(fm_embedding_dim)
        else:
            self.fm_second = None
        
        # Deep Network - 수치형 원본 + 범주형 임베딩
        deep_input_size = 0
        if self.has_embed_features:
            deep_input_size += len(self.field_embeddings) * self.fm_embedding_dim
        if self.has_vector_features:
            deep_input_size += train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        
        deep_layers_list = []
        # 첫 번째 레이어
        deep_layers_list.append(nn.Linear(deep_input_size, self.deep_hidden_size))
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
        # FM first(1) + second(1 or 0) + deep(deep_output_size)
        combination_input_size = 1 + (1 if self.fm_second else 0) + self.deep_output_size
        self.combination_layer = nn.Linear(combination_input_size, num_net_outputs)

    def forward(self, data_batch):
        # 논문에 맞는 구현
        field_embeddings_list = []
        original_inputs = []
        input_offset = 0
        
        # Embedding features 처리 (범주형)
        if self.has_embed_features:
            embed_data = data_batch[input_offset]
            for i, embed_block in enumerate(self.embed_blocks):
                # 원본 임베딩 출력을 1차 항에 사용
                embed_output = embed_block(embed_data[i].to(self.device))
                original_inputs.append(embed_output)
                field_embeddings_list.append(embed_output)
            input_offset += 1
        
        # Vector features 처리 (수치형 - 원본 그대로 사용)
        if self.has_vector_features:
            vector_data = data_batch[0].to(self.device)
            original_inputs.append(vector_data)
            # 수치형은 원본 그대로 Deep Network에 사용
        
        # FM 1차 항 계산을 위한 원본 입력
        if len(original_inputs) > 1:
            original_input = torch.cat(original_inputs, dim=1)
        else:
            original_input = original_inputs[0]
        
        fm_first = self.fm_first(original_input)  # (batch, 1)
        
        # FM 2차 항 계산 (범주형 임베딩만 사용)
        fm_second = None
        if self.fm_second and len(field_embeddings_list) > 0:
            # 필드별 임베딩을 (batch, num_fields, embed_dim) 형태로 재구성
            num_fields = len(field_embeddings_list)
            batch_size = field_embeddings_list[0].size(0)
            field_embeddings = torch.stack(field_embeddings_list, dim=1)  # (batch, num_fields, embed_dim)
            fm_second = self.fm_second(field_embeddings)  # (batch, 1)
        
        # Deep Network - 수치형 원본 + 범주형 임베딩
        deep_inputs = []
        if self.has_embed_features:
            for embed_output in field_embeddings_list:
                deep_inputs.append(embed_output)
        if self.has_vector_features:
            deep_inputs.append(vector_data)
        
        if len(deep_inputs) > 1:
            deep_input = torch.cat(deep_inputs, dim=1)
        else:
            deep_input = deep_inputs[0]
        
        deep_output = self.deep_network(deep_input)  # deep_output_size차원
        
        # Combine FM and Deep outputs
        if fm_second is not None:
            combined = torch.cat([fm_first, fm_second, deep_output], dim=1)
        else:
            combined = torch.cat([fm_first, deep_output], dim=1)
        
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

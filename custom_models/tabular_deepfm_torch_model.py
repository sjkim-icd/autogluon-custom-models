# [my_models] DeepFM 진입점

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.common import space
from autogluon.tabular.models.tabular_nn.hyperparameters.parameters import get_default_param
from autogluon.tabular.models.tabular_nn.hyperparameters.searchspaces import get_default_searchspace
from custom_models.deepfm_block import DeepFMNet
import os

class TabularDeepFMTorchModel(TabularNeuralNetTorchModel):
    ag_key = "DEEPFM"         # ← 반드시 문자열, hyperparameters에서 사용할 키
    ag_name = "DEEPFM"        # ← 사람이 읽기 좋은 이름
    ag_priority = 100         # ← 우선순위(정수, 높을수록 먼저 학습)
    _model_name = "TabularDeepFMTorchModel"
    _model_type = "tabular_deepfm_torch_model"
    _typestr = "tabular_deepfm_torch_model_v1_deepfm"  # ✅ 반드시 NN_TORCH와 다르게

    def _set_default_params(self):
        """기본 하이퍼파라미터 설정 - AutoGluon 기본 + DeepFM 커스텀"""
        # AutoGluon 기본 파라미터 가져오기
        default_params = get_default_param(problem_type=self.problem_type, framework="pytorch")
        
        # DeepFM 커스텀 파라미터 추가
        deepfm_params = {
            'fm_dropout': 0.1,
            'fm_embedding_dim': 10,
            'deep_output_size': 128,
            'deep_hidden_size': 128,
            'deep_dropout': 0.1,
            'deep_layers': 3,
            # Learning Rate Scheduler 설정
            'lr_scheduler': True,
            'scheduler_type': 'plateau',
            'lr_scheduler_patience': 5,
            'lr_scheduler_factor': 0.2,
            'lr_scheduler_min_lr': 1e-6,
        }
        default_params.update(deepfm_params)
        
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        """DeepFM 모델의 기본 검색 공간 정의 - AutoGluon 기본 + DeepFM 커스텀"""
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=self.problem_type, framework="pytorch")
        
        # DeepFM 커스텀 Search Space 추가
        if self.problem_type == 'binary':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
                # Learning Rate Scheduler Search Space
                'lr_scheduler': space.Categorical(True, False),
                'scheduler_type': space.Categorical('plateau', 'cosine', 'onecycle'),
                'lr_scheduler_patience': space.Categorical(3, 5, 7),
                'lr_scheduler_factor': space.Categorical(0.1, 0.2, 0.3),
                'lr_scheduler_min_lr': space.Real(1e-7, 1e-5, default=1e-6, log=True),
            }
        elif self.problem_type == 'multiclass':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
                # Learning Rate Scheduler Search Space
                'lr_scheduler': space.Categorical(True, False),
                'scheduler_type': space.Categorical('plateau', 'cosine', 'onecycle'),
                'lr_scheduler_patience': space.Categorical(3, 5, 7),
                'lr_scheduler_factor': space.Categorical(0.1, 0.2, 0.3),
                'lr_scheduler_min_lr': space.Real(1e-7, 1e-5, default=1e-6, log=True),
            }
        elif self.problem_type == 'regression':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
                # Learning Rate Scheduler Search Space
                'lr_scheduler': space.Categorical(True, False),
                'scheduler_type': space.Categorical('plateau', 'cosine', 'onecycle'),
                'lr_scheduler_patience': space.Categorical(3, 5, 7),
                'lr_scheduler_factor': space.Categorical(0.1, 0.2, 0.3),
                'lr_scheduler_min_lr': space.Real(1e-7, 1e-5, default=1e-6, log=True),
            }
        else:
            deepfm_searchspace = {}
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(deepfm_searchspace)
        return base_searchspace

    @classmethod
    def get_default_searchspace(cls, problem_type, num_classes=None, **kwargs):
        """DeepFM 모델의 기본 검색 공간 정의 (클래스 메서드 - 외부 호출용)"""
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=problem_type, framework="pytorch")
        
        # DeepFM 커스텀 Search Space 추가
        if problem_type == 'binary':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
            }
        elif problem_type == 'multiclass':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
            }
        elif problem_type == 'regression':
            deepfm_searchspace = {
                'fm_dropout': space.Categorical(0.1, 0.2, 0.3),
                'fm_embedding_dim': space.Categorical(8, 10, 12),
                'deep_output_size': space.Categorical(64, 128, 256),
                'deep_hidden_size': space.Categorical(64, 128, 256),
                'deep_dropout': space.Categorical(0.1, 0.2, 0.3),
                'deep_layers': space.Categorical(2, 3, 4),
            }
        else:
            deepfm_searchspace = {}
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(deepfm_searchspace)
        return base_searchspace

    def _get_net(self, train_dataset, params):
        # EmbedNet 대신 DeepFMNet 생성
        params = self._set_net_defaults(train_dataset, params)
        
        # DeepFMNet 생성
        model = DeepFMNet(
            problem_type=self.problem_type,
            num_net_outputs=self._get_num_net_outputs(),
            quantile_levels=self.quantile_levels,
            train_dataset=train_dataset,
            device=self.device,
            **params,
        )
        model = model.to(self.device)
        
        # self.model 설정
        self.model = model
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        return model
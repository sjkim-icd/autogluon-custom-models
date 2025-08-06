# [my_models] FuxiCTR 스타일 DCNv2 진입점

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.common import space
from autogluon.tabular.models.tabular_nn.hyperparameters.parameters import get_default_param
from autogluon.tabular.models.tabular_nn.hyperparameters.searchspaces import get_default_searchspace
from custom_models.dcnv2_block_fuxictr import DCNv2NetFuxiCTR
import os

class TabularDCNv2FuxiCTRTorchModel(TabularNeuralNetTorchModel):
    ag_key = "DCNV2_FUXICTR"         # ← 새로운 키
    ag_name = "DCNV2_FUXICTR"        # ← 새로운 이름
    ag_priority = 100
    _model_name = "TabularDCNv2FuxiCTRTorchModel"
    _model_type = "tabular_dcnv2_fuxictr_torch_model"
    _typestr = "tabular_dcnv2_fuxictr_torch_model_v1_fuxictr"  # ✅ 새로운 타입

    @classmethod
    def register(cls):
        """모델을 AutoGluon에 등록"""
        from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import ag_model_registry
        ag_model_registry.add(cls)
        print(f"✅ {cls.ag_name} 모델이 AutoGluon에 등록되었습니다!")

    def __init__(self, **kwargs):
        print("🔧 DCNv2 FuxiCTR __init__() 호출됨!")
        print(f"📋 받은 kwargs: {list(kwargs.keys())}")
        super().__init__(**kwargs)
        print("✅ DCNv2 FuxiCTR 초기화 완료!")

    def _set_default_params(self):
        """FuxiCTR 스타일 기본 하이퍼파라미터 설정"""
        print("🔧 DCNv2 FuxiCTR _set_default_params() 호출됨!")
        
        # AutoGluon 기본 파라미터 가져오기
        default_params = get_default_param(problem_type=self.problem_type, framework="pytorch")
        
        # FuxiCTR 스타일 DCNv2 파라미터
        dcnv2_fuxictr_params = {
            'num_cross_layers': 2,
            'cross_dropout': 0.1,
            'low_rank': 32,
            'use_low_rank_mixture': False,
            'num_experts': 4,
            'model_structure': 'parallel',
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
        default_params.update(dcnv2_fuxictr_params)
        
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        
        print(f"✅ DCNv2 FuxiCTR 기본 파라미터 설정 완료: {list(default_params.keys())}")

    def _get_default_searchspace(self):
        """FuxiCTR 스타일 검색 공간 정의"""
        print("🔍 DCNv2 FuxiCTR _get_default_searchspace() 호출됨!")
        
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=self.problem_type, framework="pytorch")
        
        # FuxiCTR 스타일 DCNv2 Search Space (문자열로 변경)
        dcnv2_fuxictr_searchspace = {
            'num_cross_layers': space.Categorical(1, 2, 3, 4),
            'cross_dropout': space.Real(0.0, 0.3),
            'low_rank': space.Categorical(16, 32, 64),
            'use_low_rank_mixture': space.Categorical(False, True),
            'num_experts': space.Categorical(2, 4, 6),
            'model_structure': space.Categorical('parallel', 'stacked', 'crossnet_only'),
            'deep_layers': space.Categorical(2, 3, 4),
            'deep_hidden_size': space.Categorical(64, 128, 256),
            'deep_dropout': space.Real(0.0, 0.3),
            'learning_rate': space.Real(0.0001, 0.01),
            'num_epochs': space.Categorical(10, 20, 30),
            'epochs_wo_improve': space.Categorical(3, 5, 8),
        }
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(dcnv2_fuxictr_searchspace)
        
        print(f"✅ DCNv2 FuxiCTR 검색 공간 생성됨: {list(base_searchspace.keys())}")
        return base_searchspace

    @classmethod
    def get_default_searchspace(cls, problem_type, num_classes=None, **kwargs):
        """FuxiCTR 스타일 검색 공간 정의 (클래스 메서드 - 외부 호출용)"""
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=problem_type, framework="pytorch")
        
        # FuxiCTR 스타일 DCNv2 Search Space (문자열로 변경)
        dcnv2_fuxictr_searchspace = {
            'num_cross_layers': space.Categorical(1, 2, 3, 4),
            'cross_dropout': space.Real(0.0, 0.3),
            'low_rank': space.Categorical(16, 32, 64),
            'use_low_rank_mixture': space.Categorical(False, True),
            'num_experts': space.Categorical(2, 4, 6),
            'model_structure': space.Categorical('parallel', 'stacked', 'crossnet_only'),
            'deep_layers': space.Categorical(2, 3, 4),
            'deep_hidden_size': space.Categorical(64, 128, 256),
            'deep_dropout': space.Real(0.0, 0.3),
            'learning_rate': space.Real(0.0001, 0.01),
            'num_epochs': space.Categorical(10, 20, 30),
            'epochs_wo_improve': space.Categorical(3, 5, 8),
        }
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(dcnv2_fuxictr_searchspace)
        
        return base_searchspace

    def _get_net(self, train_dataset, params):
        """FuxiCTR 스타일 DCNv2 네트워크 구성"""
        print("🔧 DCNv2 FuxiCTR _get_net() 호출됨!")
        print(f"📋 받은 파라미터: {list(params.keys())}")
        print(f"📊 파라미터 타입 확인:")
        for key, value in params.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # FuxiCTR 스타일 DCNv2Net 생성
        params = self._set_net_defaults(train_dataset, params)
        
        # DCNv2NetFuxiCTR 생성
        model = DCNv2NetFuxiCTR(
            problem_type=self.problem_type,
            num_net_outputs=self._get_num_net_outputs(),
            quantile_levels=self.quantile_levels,
            train_dataset=train_dataset,
            device=self.device,
            **params,  # 모든 파라미터를 그대로 전달
        )
        model = model.to(self.device)
        
        # self.model 설정
        self.model = model
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        print("✅ DCNv2NetFuxiCTR 모델 생성 완료!")
        return model

    def _train_net(self, train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset=None, test_dataset=None, time_limit=None, reporter=None, verbosity=2):
        """FuxiCTR 스타일 DCNv2 학습"""
        print("🔧 DCNv2 FuxiCTR _train_net() 호출됨!")
        
        # 기본 학습 로직
        result = super()._train_net(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity)
        
        print("✅ DCNv2 FuxiCTR 학습 완료!")
        return result 
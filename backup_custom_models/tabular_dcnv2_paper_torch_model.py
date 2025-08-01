# [my_models] DCNv2 Paper 진입점 - 논문과 동일한 구현

from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.common import space
from autogluon.tabular.models.tabular_nn.hyperparameters.parameters import get_default_param
from autogluon.tabular.models.tabular_nn.hyperparameters.searchspaces import get_default_searchspace
from custom_models.dcnv2_block_paper import DCNv2NetPaper
import os

class TabularDCNv2PaperTorchModel(TabularNeuralNetTorchModel):
    ag_key = "DCNV2_PAPER"         # ← 새로운 키
    ag_name = "DCNV2_PAPER"        # ← 새로운 이름
    ag_priority = 100
    _model_name = "TabularDCNv2PaperTorchModel"
    _model_type = "tabular_dcnv2_paper_torch_model"
    _typestr = "tabular_dcnv2_paper_torch_model_v1_paper"  # ✅ 새로운 타입

    @classmethod
    def register(cls):
        """모델을 AutoGluon에 등록"""
        from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import ag_model_registry
        ag_model_registry.add(cls)
        print(f"✅ {cls.ag_name} 모델이 AutoGluon에 등록되었습니다!")

    def __init__(self, **kwargs):
        print("🔧 DCNv2 Paper __init__() 호출됨!")
        print(f"📋 받은 kwargs: {list(kwargs.keys())}")
        super().__init__(**kwargs)
        print("✅ DCNv2 Paper 초기화 완료!")

    def _set_default_params(self):
        """논문과 동일한 기본 하이퍼파라미터 설정"""
        print("🔧 DCNv2 Paper _set_default_params() 호출됨!")
        
        # AutoGluon 기본 파라미터 가져오기
        default_params = get_default_param(problem_type=self.problem_type, framework="pytorch")
        
        # 논문과 동일한 DCNv2 파라미터 (단순화)
        dcnv2_paper_params = {
            'num_cross_layers': 2,  # 논문 기본값
            'cross_dropout': 0.1,
            'low_rank': 32,  # 논문 기본값
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
        default_params.update(dcnv2_paper_params)
        
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        
        print(f"✅ DCNv2 Paper 기본 파라미터 설정 완료: {list(default_params.keys())}")

    def _get_default_searchspace(self):
        """논문과 동일한 검색 공간 정의 (단순화)"""
        print("🔍 DCNv2 Paper _get_default_searchspace() 호출됨!")
        
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=self.problem_type, framework="pytorch")
        
        # 논문과 동일한 DCNv2 Search Space (단순화)
        dcnv2_paper_searchspace = {
            'num_cross_layers': space.Categorical(2, 3),  # 논문 범위
            'cross_dropout': space.Categorical(0.0, 0.1),
            'low_rank': space.Categorical(16, 32, 64),  # 논문 범위
            'deep_output_size': space.Categorical(64, 128),
            'deep_hidden_size': space.Categorical(64, 128),
            'deep_dropout': space.Categorical(0.1, 0.2),
            'deep_layers': space.Categorical(2, 3),
            # Learning Rate Scheduler Search Space
            'lr_scheduler': space.Categorical(True, False),
            'scheduler_type': space.Categorical('plateau', 'cosine'),
            'lr_scheduler_patience': space.Categorical(3, 5),
            'lr_scheduler_factor': space.Categorical(0.1, 0.2),
            'lr_scheduler_min_lr': space.Real(1e-7, 1e-5, default=1e-6, log=True),
            # fit에서 넘기는 고정 파라미터도 포함
            'epochs_wo_improve': space.Categorical(5),
            'num_epochs': space.Categorical(20),
        }
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(dcnv2_paper_searchspace)
        
        print(f"✅ DCNv2 Paper 검색 공간 생성됨: {list(base_searchspace.keys())}")
        return base_searchspace

    @classmethod
    def get_default_searchspace(cls, problem_type, num_classes=None, **kwargs):
        """논문과 동일한 검색 공간 정의 (클래스 메서드 - 외부 호출용)"""
        # AutoGluon 기본 Search Space 가져오기
        base_searchspace = get_default_searchspace(problem_type=problem_type, framework="pytorch")
        
        # 논문과 동일한 DCNv2 Search Space (단순화)
        dcnv2_paper_searchspace = {
            'num_cross_layers': space.Categorical(2, 3),
            'learning_rate': space.Categorical(0.0001, 0.001),
            'epochs_wo_improve': space.Categorical(5),
            'num_epochs': space.Categorical(20),
        }
        
        # 기본 Search Space와 커스텀 Search Space 합치기
        base_searchspace.update(dcnv2_paper_searchspace)
        return base_searchspace

    def _get_net(self, train_dataset, params):
        print("🔧 DCNv2 Paper _get_net() 호출됨!")
        print(f"📋 받은 파라미터: {list(params.keys())}")
        print(f"📊 파라미터 타입 확인:")
        for key, value in params.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # 논문과 동일한 DCNv2Net 생성
        params = self._set_net_defaults(train_dataset, params)
        
        # DCNv2NetPaper 생성 - 논문과 동일한 구현
        model = DCNv2NetPaper(
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
        
        print("✅ DCNv2NetPaper 모델 생성 완료!")
        return model
    
    def _train_net(self, train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset=None, test_dataset=None, time_limit=None, reporter=None, verbosity=2):
        """논문과 동일한 학습 메서드"""
        print("🚀 DCNv2 Paper _train_net 호출됨!")
        import torch
        import torch.optim.lr_scheduler as lr_scheduler
        
        # LR 스케줄러 설정
        scheduler = None
        if hasattr(self.model, 'lr_scheduler') and self.model.lr_scheduler:
            if self.model.scheduler_type == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=num_epochs,
                    eta_min=self.model.lr_scheduler_min_lr
                )
                print(f"✅ DCNv2 Paper: Cosine Annealing LR 스케줄러 적용됨 (min_lr={self.model.lr_scheduler_min_lr})")
            elif self.model.scheduler_type == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=self.model.lr_scheduler_factor,
                    patience=self.model.lr_scheduler_patience,
                    min_lr=self.model.lr_scheduler_min_lr
                )
                print(f"✅ DCNv2 Paper: ReduceLROnPlateau LR 스케줄러 적용됨 (factor={self.model.lr_scheduler_factor}, patience={self.model.lr_scheduler_patience})")
            else:
                print(f"⚠️ DCNv2 Paper: 알 수 없는 스케줄러 타입 '{self.model.scheduler_type}'")
        
        # 부모 클래스의 기본 학습 로직 실행 (스케줄러와 함께)
        if scheduler:
            # 커스텀 학습 루프로 LR 모니터링
            self._train_with_scheduler(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity, scheduler)
        else:
            # 기본 학습
            super()._train_net(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity)
    
    def _train_with_scheduler(self, train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset=None, test_dataset=None, time_limit=None, reporter=None, verbosity=2, scheduler=None):
        """논문과 동일한 스케줄러와 함께 학습"""
        import torch
        import time
        import io
        
        # 기본 설정
        self.model.init_params()
        train_dataloader = train_dataset.build_loader(batch_size, self.num_dataloading_workers, is_test=False)
        
        # 손실 함수 설정
        if isinstance(loss_kwargs.get("loss_function", "auto"), str) and loss_kwargs.get("loss_function", "auto") == "auto":
            loss_kwargs["loss_function"] = self._get_default_loss_function()
        
        # Early stopping 설정
        if epochs_wo_improve is not None:
            early_stopping_method = self._get_early_stopping_strategy(num_rows_train=len(train_dataset))
        else:
            early_stopping_method = self._get_early_stopping_strategy(num_rows_train=len(train_dataset))
        
        # Validation 데이터 준비
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val.flatten()
        else:
            y_val = None
        
        # 모델 저장용 버퍼
        io_buffer = None
        best_epoch = 0
        best_val_metric = -float('inf')
        
        # 학습 루프
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0
            
            # 현재 LR 출력
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 배치별 학습
            for batch_idx, data_batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                loss.backward()
                
                # Gradient clipping 추가 (nan 방지)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # 에포크 평균 손실
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation 평가
            val_metric = None
            if val_dataset is not None:
                val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric, _reset_threads=False)
                
                # Best model 저장
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    io_buffer = io.BytesIO()
                    torch.save(self.model, io_buffer)
                    best_epoch = epoch + 1
            
            # 간단한 epoch 로그 출력
            import time as time_module
            current_time = time_module.strftime("%Y-%m-%d %H:%M:%S")
            
            if val_metric is not None:
                log_msg = f"[{current_time}] Epoch {epoch+1}/{num_epochs}: " \
                          f"Train loss: {avg_loss:.4f}, " \
                          f"Val {self.stopping_metric.name}: {val_metric:.4f}, " \
                          f"LR: {current_lr:.2e}, " \
                          f"Best Epoch: {best_epoch}, " \
                          f"Time: {epoch_time:.2f}s"
            else:
                log_msg = f"[{current_time}] Epoch {epoch+1}/{num_epochs}: " \
                          f"Train loss: {avg_loss:.4f}, " \
                          f"LR: {current_lr:.2e}, " \
                          f"Time: {epoch_time:.2f}s"
            
            print(log_msg)
            
            # 스케줄러 스텝
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metric if val_metric is not None else avg_loss)
            
            # Early stopping 체크
            if val_dataset is not None:
                is_best = (val_metric > best_val_metric) if epoch > 0 else True
                early_stop = early_stopping_method.update(cur_round=epoch, is_best=is_best)
                if early_stop:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
        
        # Best model 로드
        if io_buffer is not None:
            io_buffer.seek(0)
            self.model = torch.load(io_buffer, weights_only=False)
            print(f"   Best model loaded from epoch {best_epoch} (Val f1: {best_val_metric:.4f})") 
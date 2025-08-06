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
    
    def _train_net(self, train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset=None, test_dataset=None, time_limit=None, reporter=None, verbosity=2):
        """LR 스케줄러가 적용된 커스텀 학습 메서드"""
        print("🚀 DeepFM _train_net 호출됨!")  # 디버그 출력 추가
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
                print(f"✅ DeepFM: Cosine Annealing LR 스케줄러 적용됨 (min_lr={self.model.lr_scheduler_min_lr})")
            elif self.model.scheduler_type == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=self.model.lr_scheduler_factor,
                    patience=self.model.lr_scheduler_patience,
                    min_lr=self.model.lr_scheduler_min_lr
                )
                print(f"✅ DeepFM: ReduceLROnPlateau LR 스케줄러 적용됨 (factor={self.model.lr_scheduler_factor}, patience={self.model.lr_scheduler_patience})")
            else:
                print(f"⚠️ DeepFM: 알 수 없는 스케줄러 타입 '{self.model.scheduler_type}'")
        
        # 부모 클래스의 기본 학습 로직 실행 (스케줄러와 함께)
        if scheduler:
            # 커스텀 학습 루프로 LR 모니터링
            self._train_with_scheduler(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity, scheduler)
        else:
            # 기본 학습
            super()._train_net(train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset, test_dataset, time_limit, reporter, verbosity)
    
    def _train_with_scheduler(self, train_dataset, loss_kwargs, batch_size, num_epochs, epochs_wo_improve, val_dataset=None, test_dataset=None, time_limit=None, reporter=None, verbosity=2, scheduler=None):
        """스케줄러와 함께 학습하면서 LR 모니터링"""
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
            
            # 출력 (AutoGluon 형식 + LR)
            if val_metric is not None:
                print(f"Epoch {epoch+1} (Update {epoch+1}).\t"
                      f"Train loss: {avg_loss:.4f}, "
                      f"Val {self.stopping_metric.name}: {val_metric:.4f}, "
                      f"LR: {current_lr:.2e}, "
                      f"Best Epoch: {best_epoch}")
            else:
                print(f"Epoch {epoch+1} (Update {epoch+1}).\t"
                      f"Train loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.2e}")
            
            print(f"   Time: {epoch_time:.2f}s")
            
            # 스케줄러 스텝
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # validation metric 사용
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
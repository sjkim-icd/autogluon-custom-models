from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (논문 원본 수식 구현)
    
    논문 수식:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    여기서,
        - p_t: 정답 클래스에 대한 예측 확률
        - α_t: 클래스별 가중치
          * α_t = α if y = 1 (positive class)
          * α_t = 1 - α if y = 0 (negative class)
        - γ: focusing parameter (논문에서는 γ = 2)
        - (1 - p_t)^γ: 쉬운 샘플의 손실을 줄이는 modulating factor

    논문 권장값:
        - α = 0.25 (positive class 가중치)
        - γ = 2.0 (focusing parameter)

    참고:
        CrossEntropyLoss는 -log(p_t)만 사용하는 반면,
        FocalLoss는 거기에 (1 - p_t)^γ를 곱해서 '쉬운 샘플'의 손실을 줄여줌
        α_t를 통해 클래스 불균형도 함께 처리
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) - raw logits
        targets: (batch_size,) - class indices (e.g., 0 또는 1)
        """

        # 1. log(p_c): CrossEntropy처럼 softmax + log 를 한 번에 수행
        log_probs = F.log_softmax(inputs, dim=1)  # shape: (B, C)

        # 2. p_c: log(p_c)를 지수화하여 확률로 복원
        probs = torch.exp(log_probs)              # shape: (B, C)

        # 3. 정답 클래스 인덱스를 long으로 변환
        targets = targets.long()

        # 4. p_c에서 정답 클래스에 해당하는 확률 p_t 추출
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # shape: (B,)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # log(p_t)

        # 5. 논문 수식: α_t 계산
        # α_t = α if y = 1 (positive class), α_t = 1 - α if y = 0 (negative class)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # 6. focal loss 공식 적용: -α_t * (1 - p_t)^γ * log(p_t)
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        # 7. 평균 or 합산
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class CustomFocalDLModel(TabularNeuralNetTorchModel):
    ag_key = "CUSTOM_FOCAL_DL"
    ag_name = "CUSTOM_FOCAL_DL"
    ag_priority = 100
    _model_name = "CustomFocalDLModel"
    _model_type = "custom_focal_dl_model"
    _typestr = "custom_focal_dl_model_v1_focalloss"
    
    def _get_default_loss_function(self):
        # 논문 권장값 사용: α=0.25, γ=2.0
        alpha = getattr(self, 'focal_alpha', 0.25)  # 기본값을 논문 권장값으로 변경
        gamma = getattr(self, 'focal_gamma', 2.0)   # 기본값을 논문 권장값으로 변경
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    def _set_params(self, **kwargs):
        """Focal Loss 파라미터와 LR scheduler 관련 파라미터를 처리"""
        print(f"🔧 CustomFocalDL _set_params 호출됨! kwargs={kwargs}")
        
        # Focal Loss 특화 파라미터 처리 (논문 권장값으로 기본값 변경)
        self.focal_alpha = kwargs.pop('focal_alpha', 0.25)  # 논문 권장값
        self.focal_gamma = kwargs.pop('focal_gamma', 2.0)   # 논문 권장값
        
        # LR scheduler 관련 파라미터들을 pop해서 저장
        self.lr_scheduler = kwargs.pop('lr_scheduler', True)
        self.scheduler_type = kwargs.pop('scheduler_type', 'cosine')
        self.lr_scheduler_min_lr = kwargs.pop('lr_scheduler_min_lr', 1e-6)
        
        print(f"🔧 CustomFocalDL _set_params: focal_alpha={self.focal_alpha}, focal_gamma={self.focal_gamma}")
        print(f"🔧 CustomFocalDL _set_params: lr_scheduler={self.lr_scheduler}, scheduler_type={self.scheduler_type}, min_lr={self.lr_scheduler_min_lr}")
        
        # 나머지 파라미터는 부모 클래스로 전달
        return super()._set_params(**kwargs)
    
    def _get_net(self, train_dataset, params):
        """Focal Loss 파라미터를 처리하고 EmbedNet에 전달"""
        print(f"🔧 CustomFocalDL _get_net 호출됨! params={params}")
        
        # Focal Loss 파라미터들을 필터링
        filtered_params = params.copy()
        focal_alpha = filtered_params.pop('focal_alpha', 0.25)  # 논문 권장값
        focal_gamma = filtered_params.pop('focal_gamma', 2.0)   # 논문 권장값
        
        # LR scheduler 관련 파라미터들을 필터링
        lr_scheduler = filtered_params.pop('lr_scheduler', True)
        scheduler_type = filtered_params.pop('scheduler_type', 'cosine')
        lr_scheduler_min_lr = filtered_params.pop('lr_scheduler_min_lr', 1e-6)
        
        # self에 파라미터 저장
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lr_scheduler = lr_scheduler
        self.scheduler_type = scheduler_type
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
        print(f"🔧 CustomFocalDL _get_net: focal_alpha={focal_alpha}, focal_gamma={focal_gamma}")
        print(f"🔧 CustomFocalDL _get_net: lr_scheduler={lr_scheduler}, scheduler_type={scheduler_type}, min_lr={lr_scheduler_min_lr}")
        
        # 람다 함수로 메서드를 덮어쓰지 않고, 인스턴스 변수로 저장
        # self._get_default_loss_function = lambda: FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # 나머지 파라미터로 EmbedNet 생성
        return super()._get_net(train_dataset, filtered_params)
    
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
        """AutoGluon의 _train_net 메서드를 복사하고 LR scheduler 추가"""
        print("🚀 CustomFocalDL _train_net 호출됨!")  # 디버그 출력 추가
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
        print(f"🔍 CustomFocalDL Debug: self.lr_scheduler = {getattr(self, 'lr_scheduler', 'NOT_SET')}")
        print(f"🔍 CustomFocalDL Debug: self.scheduler_type = {getattr(self, 'scheduler_type', 'NOT_SET')}")
        print(f"🔍 CustomFocalDL Debug: hasattr(self, 'lr_scheduler') = {hasattr(self, 'lr_scheduler')}")
        print(f"🔍 CustomFocalDL Debug: self.lr_scheduler (if exists) = {self.lr_scheduler if hasattr(self, 'lr_scheduler') else 'N/A'}")
        
        # 수정: _set_params에서 저장한 LR 스케줄러 파라미터 직접 사용
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            if hasattr(self, 'scheduler_type') and self.scheduler_type == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=self.lr_scheduler_min_lr
                )
                print(f"✅ CustomFocalDL: Cosine Annealing LR 스케줄러 적용됨 (min_lr={self.lr_scheduler_min_lr})")
            else:
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.2,
                    patience=5,
                    min_lr=self.lr_scheduler_min_lr
                )
                print(f"✅ CustomFocalDL: ReduceLROnPlateau LR 스케줄러 적용됨")
        else:
            print(f"❌ CustomFocalDL: LR 스케줄러가 설정되지 않음")
        
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
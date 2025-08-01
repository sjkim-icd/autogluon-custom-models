from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel

class CustomNNTorchModel(TabularNeuralNetTorchModel):
    ag_key = "CUSTOM_NN_TORCH"
    ag_name = "CUSTOM_NN_TORCH"
    ag_priority = 100
    _model_name = "CustomNNTorchModel"
    _model_type = "custom_nn_torch_model"
    _typestr = "custom_nn_torch_model_v1_crossentropy"
    
    def _set_params(self, **kwargs):
        """LR scheduler 관련 파라미터를 필터링"""
        print(f"🔧 CustomNNTorch _set_params 호출됨! kwargs={kwargs}")
        
        # LR scheduler 관련 파라미터들을 pop해서 저장
        self.lr_scheduler = kwargs.pop('lr_scheduler', True)
        self.scheduler_type = kwargs.pop('scheduler_type', 'cosine')
        self.lr_scheduler_min_lr = kwargs.pop('lr_scheduler_min_lr', 1e-6)
        
        print(f"🔧 CustomNNTorch _set_params: lr_scheduler={self.lr_scheduler}, scheduler_type={self.scheduler_type}, min_lr={self.lr_scheduler_min_lr}")
        
        # 나머지 파라미터는 부모 클래스로 전달
        return super()._set_params(**kwargs)
    
    def _get_net(self, train_dataset, params):
        """LR scheduler 파라미터를 필터링해서 EmbedNet에 전달"""
        print(f"🔧 CustomNNTorch _get_net 호출됨! params={params}")
        
        # LR scheduler 관련 파라미터들을 필터링
        filtered_params = params.copy()
        lr_scheduler = filtered_params.pop('lr_scheduler', True)
        scheduler_type = filtered_params.pop('scheduler_type', 'cosine')
        lr_scheduler_min_lr = filtered_params.pop('lr_scheduler_min_lr', 1e-6)
        
        # self에 LR 스케줄러 파라미터 저장 (fallback용)
        self.lr_scheduler = lr_scheduler
        self.scheduler_type = scheduler_type
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
        print(f"🔧 CustomNNTorch _get_net: self에 LR 스케줄러 파라미터 저장됨")
        print(f"🔧 CustomNNTorch _get_net: lr_scheduler={lr_scheduler}, scheduler_type={scheduler_type}, min_lr={lr_scheduler_min_lr}")
        
        # 부모 클래스의 기본 _get_net 호출 (필터링된 파라미터 사용)
        model = super()._get_net(train_dataset, filtered_params)
        
        # 모델이 None이 아닌 경우에만 LR scheduler 파라미터 설정
        if model is not None:
            model.lr_scheduler = lr_scheduler
            model.scheduler_type = scheduler_type
            model.lr_scheduler_min_lr = lr_scheduler_min_lr
            
            print(f"🔧 CustomNNTorch _get_net: model에도 LR 스케줄러 파라미터 설정됨")
        else:
            print(f"⚠️ CustomNNTorch _get_net: model이 None입니다. self에만 LR scheduler 설정됨")
        
        return model
    
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
        """DCNv2와 동일한 형식의 상세한 epoch별 출력"""
        print("🚀 CustomNNTorch _train_net 호출됨!")
        import torch
        import torch.optim.lr_scheduler as lr_scheduler
        import time
        import logging
        import io

        start_time = time.time()
        logging.debug("initializing neural network...")
        self.model.init_params()
        logging.debug("initialized")
        train_dataloader = train_dataset.build_loader(batch_size, self.num_dataloading_workers, is_test=False)

        if isinstance(loss_kwargs.get("loss_function", "auto"), str) and loss_kwargs.get("loss_function", "auto") == "auto":
            loss_kwargs["loss_function"] = self._get_default_loss_function()
        
        # LR scheduler 설정 (optimizer 생성 후)
        scheduler = None
        print(f"🔍 CustomNNTorch Debug: self.lr_scheduler = {getattr(self, 'lr_scheduler', 'NOT_SET')}")
        print(f"🔍 CustomNNTorch Debug: self.scheduler_type = {getattr(self, 'scheduler_type', 'NOT_SET')}")
        print(f"🔍 CustomNNTorch Debug: hasattr(self, 'lr_scheduler') = {hasattr(self, 'lr_scheduler')}")
        print(f"🔍 CustomNNTorch Debug: self.lr_scheduler (if exists) = {self.lr_scheduler if hasattr(self, 'lr_scheduler') else 'N/A'}")
        
        # 수정: _set_params에서 저장한 LR 스케줄러 파라미터 직접 사용
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            if hasattr(self, 'scheduler_type') and self.scheduler_type == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=self.lr_scheduler_min_lr
                )
                print(f"✅ CustomNNTorch: Cosine Annealing LR 스케줄러 적용됨 (min_lr={self.lr_scheduler_min_lr})")
            else:
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.2,
                    patience=5,
                    min_lr=self.lr_scheduler_min_lr
                )
                print(f"✅ CustomNNTorch: ReduceLROnPlateau LR 스케줄러 적용됨")
        else:
            print(f"❌ CustomNNTorch: LR 스케줄러가 설정되지 않음")
        
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
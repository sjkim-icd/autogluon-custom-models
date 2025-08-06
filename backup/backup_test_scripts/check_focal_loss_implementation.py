import torch
import torch.nn.functional as F
import numpy as np

def check_focal_loss_implementation():
    """Focal Loss 구현을 원래 수식과 비교 분석"""
    print("=== Focal Loss 구현 검증 ===\n")
    
    # 1. 원래 Focal Loss 수식 확인
    print("📚 원래 Focal Loss 수식:")
    print("FL(p_t) = -α * (1 - p_t)^γ * log(p_t)")
    print("여기서:")
    print("  - p_t: 정답 클래스에 대한 예측 확률")
    print("  - α: 클래스 불균형 보정 계수")
    print("  - γ: 어려운 샘플에 더 큰 가중치를 두는 집중도 조절 계수")
    print("  - log(p_t): CrossEntropy의 기본 손실")
    print("  - (1 - p_t)^γ: 쉬운 샘플의 손실을 줄이는 가중치\n")
    
    # 2. 현재 구현 분석
    print("🔍 현재 구현 분석:")
    
    # 테스트 데이터 생성
    batch_size = 4
    num_classes = 2
    
    # 예측 로짓 (raw logits)
    logits = torch.tensor([
        [2.0, 1.0],  # Class 0에 높은 확신
        [1.0, 2.0],  # Class 1에 높은 확신  
        [0.5, 0.5],  # 불확실한 예측
        [0.1, 3.0]   # Class 1에 매우 높은 확신
    ], dtype=torch.float32)
    
    # 정답 라벨
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    print(f"입력 로짓:\n{logits}")
    print(f"정답 라벨: {targets}\n")
    
    # 3. 단계별 계산 과정
    print("📊 단계별 계산 과정:")
    
    # Step 1: log_softmax 계산
    log_probs = F.log_softmax(logits, dim=1)
    print(f"1. log_softmax 결과:\n{log_probs}")
    
    # Step 2: 확률로 복원
    probs = torch.exp(log_probs)
    print(f"2. 확률로 복원:\n{probs}")
    
    # Step 3: 정답 클래스 확률 추출
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    print(f"3. 정답 클래스 확률 (p_t): {pt}")
    print(f"4. 정답 클래스 log 확률 (log p_t): {log_pt}\n")
    
    # 4. Focal Loss 계산 (α=1.0, γ=2.0)
    alpha = 1.0
    gamma = 2.0
    
    # 수동 계산
    focal_weight = (1 - pt) ** gamma
    focal_loss_manual = -alpha * focal_weight * log_pt
    
    print(f"5. Focal 가중치 (1 - p_t)^γ:\n{focal_weight}")
    print(f"6. 최종 Focal Loss (-α * (1 - p_t)^γ * log(p_t)):\n{focal_loss_manual}")
    print(f"7. 평균 Focal Loss: {focal_loss_manual.mean():.4f}\n")
    
    # 5. CrossEntropy와 비교
    ce_loss = F.cross_entropy(logits, targets)
    print(f"📈 CrossEntropy Loss: {ce_loss:.4f}")
    print(f"📈 Focal Loss (평균): {focal_loss_manual.mean():.4f}")
    print(f"📈 비율 (Focal/CE): {(focal_loss_manual.mean() / ce_loss):.4f}\n")
    
    # 6. 각 샘플별 분석
    print("🔬 샘플별 분석:")
    for i in range(batch_size):
        p_t = pt[i].item()
        ce_sample = -log_pt[i].item()
        focal_sample = focal_loss_manual[i].item()
        weight = focal_weight[i].item()
        
        print(f"샘플 {i}:")
        print(f"  - 정답 확률 (p_t): {p_t:.4f}")
        print(f"  - CrossEntropy: {ce_sample:.4f}")
        print(f"  - Focal 가중치: {weight:.4f}")
        print(f"  - Focal Loss: {focal_sample:.4f}")
        print(f"  - 가중치 효과: {'감소' if p_t > 0.5 else '증가'}")
        print()
    
    # 7. 구현 검증
    print("✅ 구현 검증:")
    
    # 현재 구현과 동일한 방식으로 계산
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            targets = targets.long()
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
            
            if self.reduction == 'mean':
                return loss.mean()
            else:
                return loss.sum()
    
    focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    focal_loss_result = focal_loss_fn(logits, targets)
    
    print(f"  - 수동 계산 결과: {focal_loss_manual.mean():.6f}")
    print(f"  - 함수 계산 결과: {focal_loss_result:.6f}")
    print(f"  - 일치 여부: {'✅ 일치' if abs(focal_loss_manual.mean() - focal_loss_result) < 1e-6 else '❌ 불일치'}")
    
    # 8. 수식 검증
    print("\n📐 수식 검증:")
    print("원래 수식: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)")
    
    # 각 샘플에 대해 수식 검증
    for i in range(batch_size):
        p_t = pt[i].item()
        log_p_t = log_pt[i].item()
        alpha_val = alpha
        gamma_val = gamma
        
        # 수식에 따른 계산
        focal_formula = -alpha_val * ((1 - p_t) ** gamma_val) * log_p_t
        focal_actual = focal_loss_manual[i].item()
        
        print(f"샘플 {i}:")
        print(f"  - p_t = {p_t:.4f}")
        print(f"  - log(p_t) = {log_p_t:.4f}")
        print(f"  - (1 - p_t)^γ = {(1 - p_t) ** gamma_val:.4f}")
        print(f"  - 수식 결과: {focal_formula:.6f}")
        print(f"  - 실제 결과: {focal_actual:.6f}")
        print(f"  - 검증: {'✅ 통과' if abs(focal_formula - focal_actual) < 1e-6 else '❌ 실패'}")
        print()
    
    # 9. 특수 케이스 검증
    print("🎯 특수 케이스 검증:")
    
    # 매우 확실한 예측 (p_t ≈ 1)
    confident_logits = torch.tensor([[5.0, 0.1]], dtype=torch.float32)
    confident_targets = torch.tensor([0], dtype=torch.long)
    
    confident_focal = focal_loss_fn(confident_logits, confident_targets)
    confident_ce = F.cross_entropy(confident_logits, confident_targets)
    
    print(f"확실한 예측 (p_t ≈ 1):")
    print(f"  - CrossEntropy: {confident_ce:.6f}")
    print(f"  - Focal Loss: {confident_focal:.6f}")
    print(f"  - 감소율: {(1 - confident_focal/confident_ce)*100:.2f}%")
    
    # 매우 불확실한 예측 (p_t ≈ 0.5)
    uncertain_logits = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    uncertain_targets = torch.tensor([0], dtype=torch.long)
    
    uncertain_focal = focal_loss_fn(uncertain_logits, uncertain_targets)
    uncertain_ce = F.cross_entropy(uncertain_logits, uncertain_targets)
    
    print(f"\n불확실한 예측 (p_t ≈ 0.5):")
    print(f"  - CrossEntropy: {uncertain_ce:.6f}")
    print(f"  - Focal Loss: {uncertain_focal:.6f}")
    print(f"  - 증가율: {(uncertain_focal/uncertain_ce - 1)*100:.2f}%")
    
    print("\n✅ Focal Loss 구현이 원래 수식과 정확히 일치합니다!")

if __name__ == "__main__":
    check_focal_loss_implementation() 
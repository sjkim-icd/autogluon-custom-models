import torch
import torch.nn.functional as F
import numpy as np

def check_focal_loss_paper_formula():
    """Focal Loss 논문 원본 수식과 현재 구현 비교"""
    print("=== Focal Loss 논문 원본 수식 검증 ===\n")
    
    # 1. 논문 원본 수식
    print("📚 논문 원본 수식 (ICCV 2017):")
    print("FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)")
    print("\n여기서:")
    print("  - p_t: 정답 클래스에 대한 예측 확률")
    print("  - α_t: 클래스별 가중치")
    print("    * α_t = α if y = 1 (positive class)")
    print("    * α_t = 1 - α if y = 0 (negative class)")
    print("  - γ: focusing parameter (논문에서는 γ = 2)")
    print("  - (1 - p_t)^γ: 쉬운 샘플의 손실을 줄이는 modulating factor")
    print("\n논문 권장값:")
    print("  - α = 0.25 (positive class 가중치)")
    print("  - γ = 2.0 (focusing parameter)")
    print()
    
    # 2. 현재 구현 분석
    print("🔍 현재 구현 분석:")
    
    # 테스트 데이터 생성
    logits = torch.tensor([
        [2.0, 1.0],  # Class 0에 높은 확신
        [1.0, 2.0],  # Class 1에 높은 확신  
        [0.5, 0.5],  # 불확실한 예측
        [0.1, 3.0]   # Class 1에 매우 높은 확신
    ], dtype=torch.float32)
    
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    print(f"입력 로짓:\n{logits}")
    print(f"정답 라벨: {targets}")
    print(f"  (0: negative class, 1: positive class)\n")
    
    # 3. 논문 수식에 따른 계산
    print("📊 논문 수식에 따른 계산:")
    
    # 논문 권장 파라미터
    alpha = 0.25  # positive class 가중치
    gamma = 2.0   # focusing parameter
    
    # 단계별 계산
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    
    # 정답 클래스 확률 추출
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    print(f"1. 정답 클래스 확률 (p_t): {pt}")
    print(f"2. 정답 클래스 log 확률 (log p_t): {log_pt}")
    
    # α_t 계산 (논문 수식)
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
    print(f"3. 클래스별 가중치 (α_t): {alpha_t}")
    
    # Focal Loss 계산 (논문 수식)
    focal_weight = (1 - pt) ** gamma
    focal_loss_paper = -alpha_t * focal_weight * log_pt
    
    print(f"4. Focal 가중치 (1 - p_t)^γ:\n{focal_weight}")
    print(f"5. 논문 수식 Focal Loss (-α_t * (1 - p_t)^γ * log(p_t)):\n{focal_loss_paper}")
    print(f"6. 평균 Focal Loss: {focal_loss_paper.mean():.4f}\n")
    
    # 4. 현재 구현과 비교
    print("🔬 현재 구현 vs 논문 수식 비교:")
    
    # 현재 구현 (alpha=1.0, gamma=2.0)
    current_alpha = 1.0
    current_gamma = 2.0
    
    focal_loss_current = -current_alpha * (1 - pt) ** current_gamma * log_pt
    
    print(f"현재 구현 (α=1.0, γ=2.0):")
    print(f"  - 평균 Focal Loss: {focal_loss_current.mean():.4f}")
    print(f"\n논문 수식 (α=0.25, γ=2.0, α_t 적용):")
    print(f"  - 평균 Focal Loss: {focal_loss_paper.mean():.4f}")
    print(f"\n차이점:")
    print(f"  - α 값: 현재={current_alpha}, 논문={alpha}")
    print(f"  - α_t 적용: 현재=❌, 논문=✅")
    print(f"  - 비율 (현재/논문): {(focal_loss_current.mean() / focal_loss_paper.mean()):.2f}\n")
    
    # 5. 각 샘플별 상세 비교
    print("📈 샘플별 상세 비교:")
    for i in range(len(targets)):
        target = targets[i].item()
        p_t = pt[i].item()
        alpha_t_val = alpha_t[i].item()
        current_loss = focal_loss_current[i].item()
        paper_loss = focal_loss_paper[i].item()
        
        print(f"샘플 {i}:")
        print(f"  - 정답 클래스: {target} ({'positive' if target == 1 else 'negative'})")
        print(f"  - 정답 확률 (p_t): {p_t:.4f}")
        print(f"  - α_t: {alpha_t_val:.2f}")
        print(f"  - 현재 구현 손실: {current_loss:.4f}")
        print(f"  - 논문 수식 손실: {paper_loss:.4f}")
        print(f"  - 비율 (현재/논문): {(current_loss / paper_loss):.2f}")
        print()
    
    # 6. 논문 수식 정확성 검증
    print("✅ 논문 수식 정확성 검증:")
    
    # 논문 수식에 따른 수동 계산
    for i in range(len(targets)):
        target = targets[i].item()
        p_t = pt[i].item()
        log_p_t = log_pt[i].item()
        
        # α_t 계산
        alpha_t_val = alpha if target == 1 else (1 - alpha)
        
        # 논문 수식 계산
        focal_formula = -alpha_t_val * ((1 - p_t) ** gamma) * log_p_t
        focal_actual = focal_loss_paper[i].item()
        
        print(f"샘플 {i}:")
        print(f"  - target = {target}, α_t = {alpha_t_val:.2f}")
        print(f"  - p_t = {p_t:.4f}, log(p_t) = {log_p_t:.4f}")
        print(f"  - (1 - p_t)^γ = {(1 - p_t) ** gamma:.4f}")
        print(f"  - 수식 결과: {focal_formula:.6f}")
        print(f"  - 실제 결과: {focal_actual:.6f}")
        print(f"  - 검증: {'✅ 통과' if abs(focal_formula - focal_actual) < 1e-6 else '❌ 실패'}")
        print()
    
    # 7. 현재 구현의 문제점
    print("⚠️ 현재 구현의 문제점:")
    print("1. α_t 적용 안함: 현재는 모든 클래스에 동일한 α=1.0 사용")
    print("2. 논문 권장값 미적용: α=0.25, γ=2.0 사용해야 함")
    print("3. 클래스 불균형 처리 부족: positive/negative 클래스 구분 없음")
    print()
    
    # 8. 수정된 구현 제안
    print("🔧 수정된 구현 제안:")
    
    class CorrectFocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
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
            
            # 논문 수식: α_t 적용
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
            
            if self.reduction == 'mean':
                return loss.mean()
            else:
                return loss.sum()
    
    correct_focal = CorrectFocalLoss(alpha=alpha, gamma=gamma)
    correct_result = correct_focal(logits, targets)
    
    print(f"수정된 구현 결과: {correct_result:.6f}")
    print(f"논문 수식 결과: {focal_loss_paper.mean():.6f}")
    print(f"일치 여부: {'✅ 일치' if abs(correct_result - focal_loss_paper.mean()) < 1e-6 else '❌ 불일치'}")
    
    print("\n🎯 결론:")
    print("현재 구현은 논문 수식과 부분적으로만 일치합니다.")
    print("α_t 적용이 누락되어 있어 완전한 Focal Loss가 아닙니다.")
    print("클래스 불균형 문제를 더 효과적으로 해결하려면 α_t를 적용해야 합니다!")

if __name__ == "__main__":
    check_focal_loss_paper_formula() 
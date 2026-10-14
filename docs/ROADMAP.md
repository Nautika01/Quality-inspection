# 로드맵

v0.2에서 Phase 0~5의 **기능**은 모두 구현됐습니다. 아래는 각 단계에서 무엇이 실제 동작이고 무엇이 시뮬레이션인지, 그리고 실제 현장 적용을 위해 남은 일입니다.

| Phase | 상태 | 실제 동작 | 시뮬레이션/대체 |
|---|---|---|---|
| 0 합성 데이터 MVP | ✅ | 비전·판정·SPC·리포트 | 이미지가 합성 |
| 1 검사 신뢰성 | ✅ | 가드밴드, 등급, ANOVA Gage R&R 계산 | 재고정 조건을 렌더링으로 모사 |
| 2 실제 이미지 | ✅ 도구 | import / calibrate / MVTec 로더·평가 | 실제 사진으로 검증은 사용자가 수행 |
| 3 딥러닝 | ✅ (CNN 없이) | PatchCore 방식 메모리뱅크, 코어셋, kNN, 하이브리드 규칙 | 임베딩이 수작업 특징 (CNN 교체 지점 명시) |
| 4 스마트 팩토리 | ✅ | SQLite, REST, Streamlit, MQTT 발행, 알람 | 카메라·PLC 대신 시뮬레이터 |
| 5 방산 특화 | ✅ | 해시 체인 감사, ISO 2859-1, c=0, 검사 기준서, 오프라인 번들 | MIL-STD-1916 고유 표 미구현 |

## 현장 적용을 위한 다음 작업

### 1. 실제 부품으로 검증 (가장 먼저)
- 저가 세팅: 웹캠/산업용 카메라 + 링 조명 + 와셔·너트 등 원형 금속 부품
- `qinspect import` → `calibrate`(기준 게이지) → `inspect` → 유출/과검 확인
- 공개 데이터셋: MVTec AD `metal_nut`/`screw` (비상업 라이선스), Kaggle casting, NEU-DET → `anomaly-eval`
- 예상 조정: `detect_defects` 의 `k_sigma`/`min_abs`, 이상탐지 `threshold_sigma`, 등급 길이 기준

### 2. CNN 임베딩
- `anomaly.patch_features` 를 ResNet/WideResNet 중간층(anomalib PatchCore)으로 교체
- 비교표: 수작업 특징 vs CNN — AUROC, 양품 오탐, 추론 시간, 모델 크기(폐쇄망 반입 용이성)

### 3. 설비 연동 실물
- MQTT → OPC UA 서버(`asyncua`)로 PLC 태그 쓰기, 검사 트리거 수신
- MES 연동: Lot 생성/종료 API, 작업지시 번호를 `lots` 테이블에 연결
- 다중 스테이션: PostgreSQL 백엔드, 스테이션별 관리도

### 4. 품질 시스템 문서화
- MIL-STD-1916 검증수준(VL)별 표 입력 및 테스트
- AS9102 초도품검사(FAI) 양식 자동 채움 (`specdoc` 확장)
- 변경 관리 절차: 규격/모델 해시 변경 시 승인 기록 테이블

### 5. 보안·운영
- 체인 머리 해시의 외부 타임스탬프/WORM 보관
- API 인증(mTLS 또는 토큰), 역할별 권한(검사원/품질관리자/감사)
- 로그·DB 보존 주기, 백업 검증 자동화

## 주의
실제 방산업체 도면·영상·규격치는 보안 대상일 수 있습니다. 저장소에는 공개 데이터와 가상 규격만 두세요.

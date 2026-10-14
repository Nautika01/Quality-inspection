# 폐쇄망(온프레미스) 배포

방산 현장은 외부 인터넷이 차단된 경우가 많습니다. 두 가지 경로를 지원합니다.

## A. Docker 이미지 반입

```bash
# 인터넷 가능한 빌드 PC
docker build -t qinspect:0.2 .
docker save qinspect:0.2 | gzip > qinspect-0.2.tar.gz
sha256sum qinspect-0.2.tar.gz > qinspect-0.2.tar.gz.sha256

# 폐쇄망 서버 (매체 반입 후)
sha256sum -c qinspect-0.2.tar.gz.sha256
docker load < qinspect-0.2.tar.gz
docker compose up -d        # api :8000, dashboard :8501, mqtt :1883
```

- 컨테이너는 비루트 사용자로 실행되며 `/data` 볼륨에 DB가 남습니다.
- `HEALTHCHECK`가 `/health`를 확인합니다. `/health` 응답의 `spec_hash`, `model_hash`로 배포된 규격·모델 버전을 확인하세요.

## B. 파이썬 휠하우스 반입 (Docker 불가 시)

```bash
# 빌드 PC (현장과 같은 OS/파이썬 버전으로)
./scripts/build_offline_bundle.sh      # dist/offline/*.whl + SHA256SUMS

# 현장
sha256sum -c dist/offline/SHA256SUMS
pip install --no-index --find-links dist/offline "qinspect[server]"
qinspect serve --db /data/qinspect.db
```

## 운영 체크리스트

| 항목 | 방법 |
|---|---|
| 검사 프로그램 버전 고정 | `qinspect --version`, DB `program_version` 컬럼 |
| 규격 변경 통제 | `spec.json` 개정 → `spec_hash` 변경 → 검사 기준서 재생성(`qinspect specdoc`) |
| 모델 변경 통제 | `anomaly_model.npz` 해시가 DB와 `/health`에 기록 |
| 기록 무결성 | 교대/일 단위 `qinspect audit --db ...`; 체인 머리 해시를 외부에 보관 |
| 교정 | 기준 게이지로 `qinspect calibrate` → px/mm 갱신, Gage R&R 재실시 |
| 백업 | SQLite 파일 복사(`sqlite3 db ".backup ..."`) + 체인 검증 |

## 환경 변수

| 변수 | 의미 | 기본 |
|---|---|---|
| `QINSPECT_DB` | SQLite 경로 | `qinspect.db` |
| `QINSPECT_SPEC` | 규격 JSON | 내장 예시 규격 |
| `QINSPECT_MODEL` | 이상탐지 모델 `.npz` | 미사용 |

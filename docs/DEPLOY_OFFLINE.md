# 폐쇄망(온프레미스) 배포

방산 현장은 외부 인터넷이 차단된 경우가 많습니다. 두 가지 경로를 지원합니다.

## A. Docker 이미지 반입

```bash
# 인터넷 가능한 빌드 PC
docker build -t qinspect:0.3 .
docker save qinspect:0.3 | gzip > qinspect-0.3.tar.gz
sha256sum qinspect-0.3.tar.gz > qinspect-0.3.tar.gz.sha256

# 폐쇄망 서버 (매체 반입 후)
sha256sum -c qinspect-0.3.tar.gz.sha256
docker load < qinspect-0.3.tar.gz
docker compose up -d        # api :8000, dashboard :8501, mqtt :1883  (compose는 image: qinspect:0.3 을 그대로 씀)
```

- 컨테이너는 비루트(uid 1000) 사용자로 실행되며 `/data` 볼륨에 DB·증거 이미지가 남습니다. 호스트 디렉터리를 바인드 마운트하면 `chown -R 1000 <dir>` 이 필요합니다.
- `HEALTHCHECK`가 API `/health`, 대시보드 `/_stcore/health`를 확인합니다. `/health` 응답의 `spec_hash`, `model_hash`, `auth`로 배포된 규격·모델·인증 상태를 확인하세요.
- 쓰기 API를 보호하려면 compose의 `QINSPECT_API_KEY`를 설정하고 스테이션/MES가 `X-API-Key` 헤더를 보내게 합니다.

## B. 파이썬 휠하우스 반입 (Docker 불가 시)

```bash
# 빌드 PC (현장과 같은 OS/파이썬 버전으로)
./scripts/build_offline_bundle.sh      # requirements-lock.txt 기준 dist/offline/*.whl + SHA256SUMS

# 현장
sha256sum -c dist/offline/SHA256SUMS
pip install --no-index --find-links dist/offline -r requirements-lock.txt qinspect
qinspect serve --db /data/qinspect.db --api-key "$(cat /etc/qinspect/api.key)"
```

`requirements-lock.txt`는 검증된 의존성 버전을 고정합니다(Python 3.13 기준 생성). 다른 파이썬 버전이면 빌드 PC에서 `pip freeze`로 다시 만들고 테스트를 돌린 뒤 반입하세요.

## 운영 체크리스트

| 항목 | 방법 |
|---|---|
| 검사 프로그램 버전 고정 | `qinspect --version`, DB `program_version` 컬럼 |
| 규격 변경 통제 | `spec.json` 개정 → `spec_hash` 변경 → **새 Lot** 으로만 검사 가능(기존 Lot에 다른 규격은 거부) → `qinspect specdoc` 재생성 |
| 모델 변경 통제 | `anomaly_model.npz` 해시가 DB·`/health`·검사 기준서에 기록 |
| 기록 무결성 | 교대/일 단위 `qinspect audit --db ... --anchor` (깨지면 exit 2); 앵커 머리 해시를 외부(MES, 출력물)에 보관 |
| 관리한계 | 안정 구간에서 `POST /spc/limits/freeze` 또는 `run-line --freeze-by`; 재시작 시 자동 로드 |
| 교정 | 교대 시작 시 기준 디스크로 `qinspect calibrate --db ...` → `calibrations` 기록, px/mm 갱신 |
| NG 종결 | `qinspect disposition --list` → 처분 기록; 재검사는 `reinspection_of`로 원 기록에 연결 |
| Lot 출하 | `qinspect lot-close` (샘플링 + 치명결함 규칙 + 앵커) |
| 교대 리포트 | `qinspect export --since --until --tz Asia/Seoul` 또는 `GET /reports/shift` |
| 백업 | SQLite 파일 복사(`sqlite3 db ".backup ..."`) + 체인 검증 |

## 환경 변수

| 변수 | 의미 | 기본 |
|---|---|---|
| `QINSPECT_DB` | SQLite 경로 | `qinspect.db` |
| `QINSPECT_SPEC` | 규격 JSON | 내장 예시 규격 |
| `QINSPECT_MODEL` | 이상탐지 모델 `.npz` | 미사용 |
| `QINSPECT_API_KEY` | 쓰기 API 보호 키 (`X-API-Key`) | 없음(모두 허용) |
| `QINSPECT_IMAGE_DIR` | NG/ERROR 프레임 증거 저장 폴더 | 저장 안 함 |
| `QINSPECT_MAX_UPLOAD_MB` | `/inspect` 업로드 상한 | 20 |
| `QINSPECT_TZ` | 대시보드·export 표시 시간대 | `Asia/Seoul` |

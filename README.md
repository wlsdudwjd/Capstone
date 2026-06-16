# 보안스티커 위규 탐지 시스템

웹캠 영상에서 휴대폰, 렌즈, 스티커를 검출해 정상/위규 여부를 실시간으로 확인하는 로컬 대시보드 프로젝트입니다. 저장소에는 실행 코드, 웹 UI, 학습된 모델 파일, 라벨 이미지가 함께 포함되어 있습니다.

자세한 프로젝트 설명은 [INSTRUCTION.pdf](INSTRUCTION.pdf)를 참고하세요.

## 포함된 주요 파일

```text
.
├── webcam.py                 # 웹캠 추론, 로컬 웹 서버, 위규 로그 저장
├── web_ui/                   # 브라우저 대시보드 화면
├── detection.pt              # 휴대폰/렌즈/스티커 객체 검출 모델
├── original_best.pt          # 전체 이미지 정상/위규 분류 모델
├── important_best.pt         # 크롭 이미지 정상/위규 분류 모델
├── labels/                   # 학습/검증용 이미지 데이터
└── INSTRUCTION.pdf          # 상세 설명서 PDF
```

## 빠른 실행

아래 명령은 실행 예시입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy pillow
python webcam.py
```

실행 후 브라우저에서 아래 주소를 열면 됩니다.

```text
http://127.0.0.1:8000
```

로그 화면은 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/logs
```

## 기본 실행 옵션

```powershell
python webcam.py                         # 기본값: 카메라 0, 1 두 대 사용
python webcam.py --camera-index 0        # 카메라 한 대만 사용
python webcam.py --camera-indexes 0 1    # 전면/후면 카메라 직접 지정
python webcam.py --device 0              # GPU 사용
python webcam.py --port 8080             # 서버 포트 변경
```

기본 로그 저장 위치는 `violation_logs/YYYY-MM-DD/`입니다.

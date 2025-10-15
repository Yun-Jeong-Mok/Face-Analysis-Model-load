# 클라우드 저장소(R2)에서 모델 불러오기
---

### 환경 설정
- Visual studio 2022 IDE ( 프로젝트 구성-Release 플랫폼-x64 C++ 언어 표준-ISO C++17 표준 )
- https://github.com/microsoft/vcpkg 에서 vcpkg 윈도우 환경에 설치 ( 라이브러리 설치용 )

| git clone https://github.com/microsoft/vcpkg -> cd vcpkg -> .\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install curl:x64-windows opencv4:x64-windows

- onnxruntime 라이브러리는 별도 설치 필요

| https://github.com/microsoft/onnxruntime/releases 이동 후 
onnxruntime-win-x64-1.23.1.zip 수동으로 다운, 압축 해제 후 lib 폴더와 include 폴더 안 파일들을 vcpkg 폴더 안으로 이동

- 프로젝트 -> CPP속성 -> VC++ 디렉터리 -> 일반 -> 포함 디렉터리에 vcpkg 폴더 추가 ( 라이브러리 연동 )

```
### 프로젝트 폴더 내 x64/Debug/ 안에 위치할 파일들 ( cpp.exe 있는 위치 )
libcurl.dll, onnxruntime.dll, opencv_calib3d4.dll, opencv_core4.dll, opencv_dnn4.dll
opencv_highgui4.dll, opencv_imagcodecs4.dll, openvideo4.dll, opencv_videoio4.dll
```

참고 - 전체 프로젝트 폴더 https://drive.google.com/file/d/119Kr_EhTwOkuqjo_PAql4oxZJhKCw2Cm/view?usp=drive_link 

## 코드 설명

- Cloudfare 객체 저장소인 R2에 FaceAnaysis 모델 보관
- 파이썬과 C++을 통해 메모리로 모델을 직접 저장하지 않고 동작 시에만 메모리에 로드, 종료 시 초기화
- 등록 시 JPG 형식으로 현재 시간값을 부여해 이미지 폴더에 저장 ( DB 연동 전 임시 테스트 용도 )
- 조회 시 이미지 폴더에 있는 이미지들에서 임베딩 추출해 현재 카메라에 잡힌 얼굴 임베딩과 비교 검증

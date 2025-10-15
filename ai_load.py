import requests
import cv2
import numpy as np
import io
import time
import os

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 사용자 정보 입력 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼

# 1. 버킷의 기본 공개 URL
BASE_URL = "https://pub-c87e8b5a5b7a486bb9587d5fc2a7b71f.r2.dev" 

# 2. R2 버킷 내부의 폴더 경로
R2_FOLDER_PREFIX = "insightface_models/buffalo_l"

# 3. 불러올 모델 파일 이름
# 3-1. 얼굴 인식(값 검증)용 모델
REC_MODEL_FILENAME = "w600k_r50.onnx"
# 3-2. 얼굴 탐지(얼굴 찾기)용 모델
DET_MODEL_FILENAME = "det_10g.onnx"

# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 사용자 정보 입력 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def load_model_from_url(url):
    """URL에서 모델을 다운로드하여 메모리에 바이트 형태로 로드합니다."""
    print(f"⏬ URL에서 모델 다운로드 시작: {url.split('/')[-1]}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        model_bytes = response.content
        print(f"✅ 모델 데이터 다운로드 완료 (크기: {len(model_bytes) / 1024 / 1024:.2f} MB)")
        return model_bytes
    except requests.exceptions.RequestException as e:
        print(f"❌ 다운로드 오류: {e}")
        return None

def process_and_get_best_face(frame, detections, confidence_threshold=0.5):
    """탐지 결과를 처리하여 가장 신뢰도 높은 얼굴의 좌표와 신뢰도를 함께 반환합니다."""
    frame_height, frame_width = frame.shape[:2]
    best_detection = None
    max_confidence = 0

    if detections is not None and detections.shape[0] > 0:
        best_idx = np.argmax(detections[:, 4])
        max_confidence = detections[best_idx, 4]
        
        if max_confidence > confidence_threshold:
            box = detections[best_idx, 0:4]
            x1 = int(box[0] * frame_width)
            y1 = int(box[1] * frame_height)
            x2 = int(box[2] * frame_width)
            y2 = int(box[3] * frame_height)
            best_detection = (x1, y1, x2, y2)
    
    return best_detection, max_confidence

def capture_best_face(cap, det_net, rec_net, duration=1.0):
    """주어진 시간 동안 최적의 얼굴 프레임과 임베딩을 캡처합니다."""
    print(f"\n{duration}초간 최적의 프레임을 캡처합니다...")
    capture_buffer = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret_cap, frame_cap = cap.read()
        if not ret_cap: continue

        cv2.putText(frame_cap, "Capturing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Real-time Face Embedding', frame_cap)
        cv2.waitKey(1)

        det_blob_cap = cv2.dnn.blobFromImage(frame_cap, 1.0, (320, 320), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        det_net.setInput(det_blob_cap)
        detections_cap = det_net.forward()
        
        if len(detections_cap.shape) > 2: detections_cap = detections_cap.squeeze()

        face_box_cap, max_conf_cap = process_and_get_best_face(frame_cap, detections_cap, confidence_threshold=0.4)

        if face_box_cap:
            x1, y1, x2, y2 = face_box_cap
            face_cap = frame_cap[y1:y2, x1:x2]
            if face_cap.size > 0:
                rec_blob_cap = cv2.dnn.blobFromImage(face_cap, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                rec_net.setInput(rec_blob_cap)
                embedding_cap = rec_net.forward()
                embedding_cap = embedding_cap / np.linalg.norm(embedding_cap)
                
                capture_buffer.append({"frame": frame_cap.copy(), "embedding": embedding_cap, "confidence": max_conf_cap})

    if capture_buffer:
        best_capture = max(capture_buffer, key=lambda x: x['confidence'])
        return best_capture
    return None

def cosine_similarity(vec1, vec2):
    """두 임베딩 벡터 간의 코사인 유사도를 계산합니다."""
    return np.dot(vec1.flatten(), vec2.flatten())

def is_duplicate_face(new_embedding, det_net, rec_net, image_folder="./images", similarity_threshold=0.65):
    """새로운 임베딩이 지정된 폴더의 기존 이미지들과 중복되는지 확인합니다."""
    if not os.path.exists(image_folder):
        return False

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            existing_image_path = os.path.join(image_folder, filename)
            existing_image = cv2.imread(existing_image_path)
            
            if existing_image is None: continue

            # 기존 이미지에서 얼굴 탐지 및 임베딩 추출
            det_blob = cv2.dnn.blobFromImage(existing_image, 1.0, (320, 320), (127.5, 127.5, 127.5), swapRB=True, crop=False)
            det_net.setInput(det_blob)
            detections = det_net.forward()
            if len(detections.shape) > 2: detections = detections.squeeze()

            face_box, _ = process_and_get_best_face(existing_image, detections, confidence_threshold=0.4)
            
            if face_box:
                x1, y1, x2, y2 = face_box
                face = existing_image[y1:y2, x1:x2]
                if face.size > 0:
                    rec_blob = cv2.dnn.blobFromImage(face, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                    rec_net.setInput(rec_blob)
                    existing_embedding = rec_net.forward()
                    existing_embedding = existing_embedding / np.linalg.norm(existing_embedding)

                    similarity = cosine_similarity(new_embedding, existing_embedding)
                    
                    if similarity > similarity_threshold:
                        print(f"\n[중복 감지] 유사도 {similarity:.2f} ({filename})")
                        return True
    return False


# --- 메인 코드 실행 부분 ---
if __name__ == "__main__":
    
    rec_model_url = f"{BASE_URL}/{R2_FOLDER_PREFIX}/{REC_MODEL_FILENAME}"
    det_model_url = f"{BASE_URL}/{R2_FOLDER_PREFIX}/{DET_MODEL_FILENAME}"
    
    rec_model_data = load_model_from_url(rec_model_url)
    det_model_data = load_model_from_url(det_model_url)

    if rec_model_data and det_model_data:
        try:
            print("\n--- OpenCV dnn으로 모델 로드 시도 ---")
            rec_net = cv2.dnn.readNetFromONNX(np.frombuffer(rec_model_data, np.uint8))
            det_net = cv2.dnn.readNetFromONNX(np.frombuffer(det_model_data, np.uint8))
            print("🎉 인식 및 탐지 모델을 메모리에서 성공적으로 불러왔습니다!")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 오류: 웹캠을 열 수 없습니다.")
                exit()
            
            print("\n📹 웹캠 시작: [r] 등록 | [c] 비교 | [q] 종료")

            registered_embedding = None
            
            while True:
                ret, frame = cap.read()
                if not ret: break

                det_blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (127.5, 127.5, 127.5), swapRB=True, crop=False)
                det_net.setInput(det_blob)
                detections = det_net.forward()

                if len(detections.shape) > 2: detections = detections.squeeze()
                
                face_box, max_conf = process_and_get_best_face(frame, detections, confidence_threshold=0.4)
                
                if face_box:
                    x1, y1, x2, y2 = face_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.imshow('Real-time Face Embedding', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord('r'): # 'r' 키를 눌러 등록
                    best_capture = capture_best_face(cap, det_net, rec_net)
                    if best_capture:
                        new_embedding = best_capture['embedding']
                        
                        # 중복 얼굴 확인
                        if is_duplicate_face(new_embedding, det_net, rec_net):
                            print("\n❌ 등록 실패: 이미 등록된 인물과 매우 유사합니다.")
                        else:
                            # 중복이 아니면 등록 진행
                            registered_embedding = new_embedding
                            
                            # 이미지 저장
                            capture_folder = "./images"
                            if not os.path.exists(capture_folder):
                                os.makedirs(capture_folder)
                                print(f"\n'{capture_folder}' 폴더를 생성했습니다.")
                            
                            timestamp = int(time.time())
                            capture_path = os.path.join(capture_folder, f"registered_{timestamp}.jpg")
                            cv2.imwrite(capture_path, best_capture['frame'])
                            
                            print(f"\n✅ 얼굴 등록 완료! 이미지를 저장했습니다: {capture_path}")
                            print("이제 'c' 키를 눌러 비교할 수 있습니다.")
                    else:
                        print("\n❌ 등록 실패: 1초 동안 유효한 얼굴을 찾지 못했습니다.")

                elif key == ord('c'): # 'c' 키를 눌러 비교
                    if registered_embedding is None:
                        print("\n⚠️ 먼저 'r' 키를 눌러 얼굴을 등록해야 합니다.")
                        continue
                    
                    current_capture = capture_best_face(cap, det_net, rec_net)
                    if current_capture:
                        current_embedding = current_capture['embedding']
                        similarity = cosine_similarity(registered_embedding, current_embedding)
                        
                        print(f"\n--- 비교 결과 ---")
                        print(f"유사도 점수: {similarity:.4f}")
                        
                        # 보통 0.5 또는 0.6 이상이면 동일인으로 판단합니다.
                        if similarity > 0.5:
                            print("결과: ✅ 동일 인물일 확률이 높습니다.")
                        else:
                            print("결과: ❌ 다른 인물일 확률이 높습니다.")
                    else:
                        print("\n❌ 비교 실패: 1초 동안 유효한 얼굴을 찾지 못했습니다.")

            print("\n프로그램을 종료합니다.")
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"\n❌ 처리 중 심각한 오류 발생: {e}")


// R2에서 ONNX 모델 다운로드 -> OpenCV DNN 로드 -> 웹캠 탐지/등록/비교/중복검사
// 터미널에서 빌드 시 "Developer Command Prompt for VS 2022"로 진입
// (일반 프롬프트 사용 시 안 됨)
// cd C:\<프로젝트 폴더 경로>
// msbuild cpp.sln /p:Configuration=Release 실행

// 라이브러리 설치
// vcpkg에 opencv4, curl 설치 ( x64_windows )
// opencv 관련 .dll 파일 프로젝트 debug 폴더 안으로 복사 ( cpp.exe 있는 곳 )
// 구성 release, ISO C++17 표준 사용

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <curl/curl.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cctype>

namespace fs = std::filesystem;

// 캡쳐 결과 형식
struct CaptureResult {
    cv::Mat frame;
    cv::Mat embedding; 
    float confidence;
    bool valid;
};

// curl 콜백 (모델 불러올 때 사용)
static size_t WriteToVector(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::vector<unsigned char>* buffer = reinterpret_cast<std::vector<unsigned char>*>(userp);
    const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(contents);
    buffer->insert(buffer->end(), dataPtr, dataPtr + total);
    return total;
}

bool download_to_vector(const std::string& url, std::vector<unsigned char>& out) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToVector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "❌ curl error: " << curl_easy_strerror(res) << "\n";
        curl_easy_cleanup(curl);
        return false;
    }
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);
    if (response_code >= 400) {
        std::cerr << "❌ HTTP response: " << response_code << "\n";
        return false;
    }
    return true;
}

// 임시 파일로 저장
bool save_vector_to_file(const std::vector<unsigned char>& data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    ofs.close();
    return true;
}

// 여러 ONNX 출력 포맷 처리
std::vector<cv::Mat> forward_and_squeeze(cv::dnn::Net& net, const cv::Mat& inputBlob) {
    net.setInput(inputBlob);
    cv::Mat out = net.forward();

    std::vector<cv::Mat> mats;
    if (out.empty()) return mats;

    if (out.dims == 4) {

        int d0 = out.size[0], d1 = out.size[1], d2 = out.size[2], d3 = out.size[3];
        if (d2 > 0 && d3 > 0) {
            cv::Mat reshaped(d2, d3, CV_32F, out.ptr<float>());
            mats.push_back(reshaped.clone());
            return mats;
        }
    }
    else if (out.dims == 3) {
        int r = out.size[1], c = out.size[2];
        cv::Mat reshaped(r, c, CV_32F, out.ptr<float>());
        mats.push_back(reshaped.clone());
        return mats;
    }
    else if (out.dims == 2) {
        mats.push_back(out.clone());
        return mats;
    }

    mats.push_back(out.reshape(1, static_cast<int>(out.total())));
    return mats;
}

bool process_and_get_best_face(const cv::Mat& frame, const cv::Mat& detections, cv::Rect& bestRect, float& bestScore, float confThreshold = 0.4f) {
    bestScore = 0.0f;
    bool found = false;
    int frame_h = frame.rows, frame_w = frame.cols;

    if (detections.empty()) return false;

    cv::Mat det = detections;

    if (det.rows == 1 && det.cols % 5 == 0) {
        det = det.reshape(1, det.cols / 5);
    }

    for (int r = 0; r < det.rows; ++r) {
        if (det.cols < 5) continue;
        float x1 = det.at<float>(r, 0);
        float y1 = det.at<float>(r, 1);
        float x2 = det.at<float>(r, 2);
        float y2 = det.at<float>(r, 3);
        float score = det.at<float>(r, 4);

        if (score > bestScore && score > confThreshold) {
            int ix1 = std::clamp<int>(std::lround(x1 * frame_w), 0, frame_w - 1);
            int iy1 = std::clamp<int>(std::lround(y1 * frame_h), 0, frame_h - 1);
            int ix2 = std::clamp<int>(std::lround(x2 * frame_w), 0, frame_w - 1);
            int iy2 = std::clamp<int>(std::lround(y2 * frame_h), 0, frame_h - 1);
            if (ix2 <= ix1 || iy2 <= iy1) continue;
            bestRect = cv::Rect(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
            bestScore = score;
            found = true;
        }
    }
    return found;
}

// 코사인 유사도 계산 0 ~ 1 값
double cosine_similarity(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat af = a.reshape(1, 1);
    cv::Mat bf = b.reshape(1, 1);
    double na = cv::norm(af);
    double nb = cv::norm(bf);
    if (na == 0.0 || nb == 0.0) return 0.0;
    double dot = af.dot(bf);
    return dot / (na * nb);
}

// 0.8초 동안 여러 프레임 캡처 후 최선의 값 선택
CaptureResult captureBestFace(cv::VideoCapture& cap, cv::dnn::Net& det_net, cv::dnn::Net& rec_net, double duration = 1.0) {
    std::cout << "\n⏳ 얼굴 캡처 중 (" << duration << "초)..." << std::endl;
    std::vector<CaptureResult> captures;
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < duration) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;
        if (frame.empty()) continue;

        // 데이터 전처리
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        std::vector<cv::Mat> outs = forward_and_squeeze(det_net, blob);
        if (outs.empty()) continue;
        cv::Mat dets = outs[0];

        cv::Rect faceRect;
        float conf = 0.0f;
        if (!process_and_get_best_face(frame, dets, faceRect, conf, 0.35f)) {
            cv::putText(frame, "No face", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Real-time Face Embedding", frame);
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        // crop 확장 ( 얼궅 탐지 개선 )
        cv::Rect box = faceRect;
        int dx = static_cast<int>(box.width * 0.10);
        int dy = static_cast<int>(box.height * 0.10);
        box.x = std::max<int>(0, box.x - dx);
        box.y = std::max<int>(0, box.y - dy);
        box.width = std::min<int>(frame.cols - box.x, box.width + 2 * dx);
        box.height = std::min<int>(frame.rows - box.y, box.height + 2 * dy);
        if (box.width <= 0 || box.height <= 0) continue;

        cv::Mat face = frame(box).clone();
        if (face.empty()) continue;

        // AI 전처리 
        cv::Mat blobRec = cv::dnn::blobFromImage(face, 1.0 / 127.5, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);
        std::vector<cv::Mat> rec_outs = forward_and_squeeze(rec_net, blobRec);
        cv::Mat emb;
        if (!rec_outs.empty()) {
            emb = rec_outs[0].clone();
        }
        else {
            rec_net.setInput(blobRec);
            emb = rec_net.forward().clone();
        }
        if (emb.empty()) continue;

        emb = emb.reshape(1, 1);
        double normv = cv::norm(emb);
        if (normv > 0) emb /= normv;

        captures.push_back({ frame.clone(), emb.clone(), conf, true });

        // 디버그: 현재 프레임에 박스 보여주기
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Face: " + std::to_string(conf), { 10,30 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Real-time Face Embedding", frame);
        if (cv::waitKey(1) == 27) break;
    }

    if (!captures.empty()) {
        auto it = std::max_element(captures.begin(), captures.end(), [](const CaptureResult& a, const CaptureResult& b) {
            return a.confidence < b.confidence;
            });
        std::cout << "✅ 얼굴 탐지 성공 (신뢰도 " << it->confidence << ")\n";
        return *it;
    }

    std::cout << "❌ 얼굴을 찾지 못했습니다.\n";
    return { cv::Mat(), cv::Mat(), 0.0f, false };
}

// ./images 폴더의 이미지들과 비교해서 중복 감지 ( 임시 )
bool isDuplicateFace(const cv::Mat& newEmbedding, cv::dnn::Net& det_net, cv::dnn::Net& rec_net, const std::string& folder = "./images", double threshold = 0.7) {
    if (!fs::exists(folder)) return false;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        std::vector<cv::Mat> outs = forward_and_squeeze(det_net, blob);
        if (outs.empty()) continue;
        cv::Mat dets = outs[0];

        cv::Rect faceRect;
        float conf = 0.0f;
        if (!process_and_get_best_face(img, dets, faceRect, conf, 0.35f)) continue;
        if (faceRect.width <= 0 || faceRect.height <= 0) continue;

        // crop 확장 ( 얼굴 탐지율 개선 )
        cv::Rect box = faceRect;
        int dx = static_cast<int>(box.width * 0.10);
        int dy = static_cast<int>(box.height * 0.10);
        box.x = std::max<int>(0, box.x - dx);
        box.y = std::max<int>(0, box.y - dy);
        box.width = std::min<int>(img.cols - box.x, box.width + 2 * dx);
        box.height = std::min<int>(img.rows - box.y, box.height + 2 * dy);
        if (box.width <= 0 || box.height <= 0) continue;

        cv::Mat face = img(box).clone();
        if (face.empty()) continue;

        cv::Mat blobRec = cv::dnn::blobFromImage(face, 1.0 / 127.5, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);
        std::vector<cv::Mat> rec_outs = forward_and_squeeze(rec_net, blobRec);
        cv::Mat emb;
        if (!rec_outs.empty()) emb = rec_outs[0].clone();
        else {
            rec_net.setInput(blobRec);
            emb = rec_net.forward().clone();
        }
        if (emb.empty()) continue;
        emb = emb.reshape(1, 1);
        double normv = cv::norm(emb);
        if (normv > 0) emb /= normv;

        double sim = cosine_similarity(newEmbedding, emb);
        if (sim > threshold) {
            std::cout << "⚠️ 중복 감지됨 (" << entry.path().filename().string() << ", 유사도 " << sim << ")\n";
            return true;
        }
    }
    return false;
}

int main() {
    // AI 모델 오브젝트 저장소 주소 ( Cloudfare R2 사용 )
    const std::string BASE_URL = "https://pub-c87e8b5a5b7a486bb9587d5fc2a7b71f.r2.dev";
    const std::string R2_FOLDER_PREFIX = "insightface_models/buffalo_l";
    const std::string REC_MODEL_FILENAME = "w600k_r50.onnx";
    const std::string DET_MODEL_FILENAME = "det_10g.onnx";

    std::string rec_model_url = BASE_URL + "/" + R2_FOLDER_PREFIX + "/" + REC_MODEL_FILENAME;
    std::string det_model_url = BASE_URL + "/" + R2_FOLDER_PREFIX + "/" + DET_MODEL_FILENAME;

    std::cout << "⏬ 모델 다운로드 시작...\n";
    std::vector<unsigned char> rec_data, det_data;
    bool ok1 = download_to_vector(rec_model_url, rec_data);
    bool ok2 = download_to_vector(det_model_url, det_data);

    if (!ok1 || !ok2) {
        std::cerr << "❌ 모델 다운로드 실패. URL 또는 네트워크를 확인하세요.\n";
        return -1;
    }
    std::cout << "✅ 다운로드 완료 (rec: " << (rec_data.size() / 1024.0 / 1024.0) << " MB, det: " << (det_data.size() / 1024.0 / 1024.0) << " MB)\n";

    // 임시 파일로 저장 ( 프로그램 종료 시 사라짐 )
    std::string tmp_rec = "tmp_rec.onnx";
    std::string tmp_det = "tmp_det.onnx";
    if (!save_vector_to_file(rec_data, tmp_rec) || !save_vector_to_file(det_data, tmp_det)) {
        std::cerr << "❌ 임시 파일 저장 실패\n";
        return -1;
    }

    try {
        std::cout << "\n--- OpenCV dnn으로 모델 로드 시도 ---\n";
        cv::dnn::Net rec_net = cv::dnn::readNetFromONNX(tmp_rec);
        cv::dnn::Net det_net = cv::dnn::readNetFromONNX(tmp_det);
        std::cout << "🎉 인식 및 탐지 모델을 성공적으로 불러왔습니다!\n";

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "❌ 오류: 웹캠을 열 수 없습니다.\n";
            return -1;
        }

        std::cout << "\n📹 웹캠 시작: [r] 등록 | [c] 비교 | [q] 종료\n";
        cv::Mat registered_embedding;
        bool has_registered = false;

        // 메인 루프
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            if (frame.empty()) continue;

            // 실시간 디스플레이 탐지
            cv::Mat dispBlob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(320, 320), cv::Scalar(), true, false);
            std::vector<cv::Mat> dispOuts = forward_and_squeeze(det_net, dispBlob);
            cv::Rect faceRect;
            float max_conf = 0.0f;
            if (!dispOuts.empty()) {
                process_and_get_best_face(frame, dispOuts[0], faceRect, max_conf, 0.35f);
            }
            if (faceRect.area() > 0) cv::rectangle(frame, faceRect, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Real-time Face Embedding", frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;

            if (key == 'r') {
                // 등록: 0.8초 동안 프레임 캡처하고 최적 벡터값 선택
                auto best = captureBestFace(cap, det_net, rec_net, 0.8);
                if (!best.valid) {
                    std::cout << "❌ 등록 실패: 0.8초 동안 유효한 얼굴을 찾지 못했습니다.\n";
                    continue;
                }
                if (isDuplicateFace(best.embedding, det_net, rec_net, "./images", 0.7)) {
                    std::cout << "❌ 이미 등록된 사용자입니다.\n";
                }
                else {
                    if (!fs::exists("./images")) fs::create_directories("./images");
                    std::string filename = "./images/registered_" + std::to_string(std::time(nullptr)) + ".jpg";
                    cv::imwrite(filename, best.frame);
                    registered_embedding = best.embedding.clone();
                    has_registered = true;
                    std::cout << "✅ 등록 완료: " << filename << "\n";
                }
            }
            else if (key == 'c') {
                // 조회: 0.8초 동안 캡처하고 비교
                if (!fs::exists("./images")) {
                    std::cout << "⚠️ 등록된 사용자 없음 (./images 폴더가 비어있음)\n";
                    continue;
                }
                auto cur = captureBestFace(cap, det_net, rec_net, 0.8);
                if (!cur.valid) {
                    std::cout << "❌ 비교 실패: 0.8초 동안 유효한 얼굴을 찾지 못했습니다.\n";
                    continue;
                }
                bool matched = false;
                // 폴더의 모든 이미지와 비교 ( 코사인 유사도 0.7 기준 )
                if (isDuplicateFace(cur.embedding, det_net, rec_net, "./images", 0.7)) {
                    std::cout << "✅ 인증 성공: 등록된 사용자입니다.\n";
                }
                else {
                    std::cout << "❌ 인증 실패: 등록된 사용자가 아닙니다.\n";
                }
            }
        }

        std::cout << "\n프로그램을 종료합니다.\n";
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        std::cerr << "\n❌ OpenCV 예외 발생: " << e.what() << "\n";
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ 예외 발생: " << e.what() << "\n";
        return -1;
    }

    return 0;
}

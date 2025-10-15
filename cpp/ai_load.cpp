// 관련 라이브러리 curl, opencv4, onnx runtime 설치
// windows 64비트 운영체제 vcpkg 사용하여 라이브러리 설치
// 터미널 빌드 시 Developer Command Prompt for VS 2022에서 ( 기본 cmd, 파워쉘 불가 )
// cd <프로젝트 폴더명> 후 msbuild YourProject.sln /p:Configuration=Release /p:Platform=x64 실행
// R2 오브젝트 저장소에서 모델 메모리 로드 -> 이미지 조회, 등록 수행

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <curl/curl.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

namespace fs = std::filesystem;

// curl 다운로드 콜백 
static size_t WriteToVector(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::vector<unsigned char>* buffer = reinterpret_cast<std::vector<unsigned char>*>(userp);
    const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(contents);
    buffer->insert(buffer->end(), dataPtr, dataPtr + total);
    return total;
}

static int ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    if (dltotal > 0) {
        int percent = static_cast<int>(dlnow * 100 / dltotal);
        std::cout << "\r[DOWNLOAD] 진행률: " << percent << "% " << std::flush;
    }
    return 0;
}

// url에서 벡터로 로드
bool download_to_vector(const std::string& url, std::vector<unsigned char>& out, const std::string& name) {
    std::cout << "[INFO] 다운로드 시작: " << name << "\n";
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ERROR] CURL 초기화 실패\n";
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToVector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);

    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    CURLcode res = curl_easy_perform(curl);
    std::cout << "\n";

    if (res != CURLE_OK) {
        std::cerr << "[ERROR] curl 수행 실패: " << curl_easy_strerror(res) << "\n";
        curl_easy_cleanup(curl);
        return false;
    }

    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);
    if (response_code >= 400) {
        std::cerr << "[ERROR] HTTP 응답 실패: " << response_code << "\n";
        return false;
    }

    std::cout << "[INFO] 다운로드 완료: " << name << " (크기: " << out.size() / 1024 << " KB)\n";
    return true;
}
// 이미지 -> 텐서 변환
Ort::Value mat_to_tensor(const cv::Mat& img, Ort::MemoryInfo& mem_info, bool is_rec_model = true) {
    cv::Mat img_f;
    if (is_rec_model) {
        cv::resize(img, img_f, cv::Size(112, 112));
        img_f.convertTo(img_f, CV_32F, 1.0 / 127.5, -1.0);
    }
    else {
        cv::resize(img, img_f, cv::Size(640, 640));
        img_f.convertTo(img_f, CV_32F, 1.0 / 255.0);
    }

    std::vector<float> input_tensor_values(img_f.total() * img_f.channels());
    std::vector<cv::Mat> channels(img_f.channels());
    cv::split(img_f, channels);

    size_t channel_size = img_f.rows * img_f.cols;
    for (int i = 0; i < img_f.channels(); ++i) {
        memcpy(input_tensor_values.data() + i * channel_size, channels[i].data, channel_size * sizeof(float));
    }

    std::vector<int64_t> input_shape = { 1, img_f.channels(), img_f.rows, img_f.cols };
    return Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
}

// 코사인 유사도 계산
double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ONNX(AI 모델) 세션 생성 
Ort::Session create_session_from_memory(Ort::Env& env, const std::vector<unsigned char>& model_data, const std::string& model_name) {
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        Ort::Session sess(env, model_data.data(), model_data.size(), session_options);
        std::cout << "[INFO] 모델 로드 완료: " << model_name << "\n";
        return sess;
    }

    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] 모델 로드 실패: " << model_name << " (" << e.what() << ")\n";
        throw;
    }
}

// NaN 예외 검사
bool has_nan(const std::vector<float>& v) {
    return std::any_of(v.begin(), v.end(), [](float x) { 
        return std::isnan(x); 
        });
}

int main() {

    const std::string BASE_URL = "https://pub-c87e8b5a5b7a486bb9587d5fc2a7b71f.r2.dev";
    const std::string R2_FOLDER_PREFIX = "insightface_models/buffalo_l";
    const std::string REC_MODEL_FILENAME = "w600k_r50.onnx";
    const std::string DET_MODEL_FILENAME = "det_10g.onnx";

    std::vector<unsigned char> rec_data, det_data;

    try {
        if (!download_to_vector(BASE_URL + "/" + R2_FOLDER_PREFIX + "/" + REC_MODEL_FILENAME, rec_data, "등록 모델"))
            return -1;
        if (!download_to_vector(BASE_URL + "/" + R2_FOLDER_PREFIX + "/" + DET_MODEL_FILENAME, det_data, "탐지 모델"))
            return -1;
    }

    catch (const std::exception& e) {
        std::cerr << "[ERROR] 모델 다운로드 중 예외 발생: " << e.what() << "\n";
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Face");
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // 세션 생성
    std::unique_ptr<Ort::Session> rec_sess, det_sess;

    try {
        std::cout << "[INFO] 모델 메모리에 로드 중...\n";
        rec_sess = std::make_unique<Ort::Session>(create_session_from_memory(env, rec_data, "등록 모델"));
        det_sess = std::make_unique<Ort::Session>(create_session_from_memory(env, det_data, "탐지 모델"));
        std::cout << "[INFO] 모든 모델 로드 완료!\n";
    }
    catch (...) {
        std::cerr << "[ERROR] 모델 세션 생성 실패\n";
        return -1;
    }

    // 웹캠
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { 
        std::cerr << "[ERROR] 웹캠 열기 실패\n"; return -1; 
    }

    std::vector<float> registered_embedding;
    bool registered = false;

    auto print_prompt = [&](bool is_registration = false) {
        if (is_registration) {
            std::cout << "[INFO] 상태: 사용자 등록 완료\n";
        }
        else {
            std::cout << "[INFO] 현재 상태: "
                << (registered ? "등록 사용자 있음, 인증 가능" : "등록 필요") << "\n";
        }
        std::cout << "[INFO] 웹캠 시작: [r] 등록 | [c] 비교 | [q] 종료\n";
        };

    print_prompt();

    while (true) {

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            continue;
        }

        cv::imshow("Face ORT", frame);

        int key = cv::waitKey(1) & 0xFF;

        if (key == 'q') {
            break;
        }

        if (key == 'r') {

            if (registered) {
                std::cout << "[WARN] 이미 등록된 사용자 있음. 새로운 등록 불가.\n";
                continue;
            }

            Ort::Value input_tensor = mat_to_tensor(frame, mem_info, true);

            try {

                Ort::AllocatorWithDefaultOptions allocator;

                Ort::AllocatedStringPtr input_name_ptr = rec_sess->GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr output_name_ptr = rec_sess->GetOutputNameAllocated(0, allocator);

                const char* input_names[] = { input_name_ptr.get() };
                const char* output_names[] = { output_name_ptr.get() };

                try {
                    auto output_tensors = rec_sess->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
                    float* emb_ptr = output_tensors.front().GetTensorMutableData<float>();
                    size_t emb_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

                    std::vector<float> emb_vec(emb_ptr, emb_ptr + emb_len);

                    if (has_nan(emb_vec)) {
                        std::cerr << "[ERROR] embedding 값에 NaN 포함. 등록 실패\n";
                        continue;
                    }

                    registered_embedding = emb_vec;
                    registered = true;

                    double min_val = *std::min_element(registered_embedding.begin(), registered_embedding.end());
                    double max_val = *std::max_element(registered_embedding.begin(), registered_embedding.end());
                    double avg_val = std::accumulate(registered_embedding.begin(), registered_embedding.end(), 0.0) / registered_embedding.size();

                    std::cout << "[INFO] ✅ 등록 완료 (embedding 길이: " << emb_len << ", min: " << min_val
                        << ", max: " << max_val << ", avg: " << avg_val << ")\n";

                    print_prompt();
                }

                catch (const Ort::Exception& e) {
                    std::cerr << "[ERROR] 등록 처리 실패: " << e.what() << "\n";
                }

            }

            catch (const Ort::Exception& e) {
                std::cerr << "[ERROR] 등록 처리 실패: " << e.what() << "\n";
            }

        }
        else if (key == 'c') {

            if (!registered) {
                std::cout << "[WARN] 등록된 사용자 없음. 먼저 [r]로 등록하세요.\n";
                continue;
            }

            Ort::Value input_tensor = mat_to_tensor(frame, mem_info, true);

            try {
                Ort::AllocatorWithDefaultOptions allocator;
                Ort::AllocatedStringPtr input_name_ptr = rec_sess->GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr output_name_ptr = rec_sess->GetOutputNameAllocated(0, allocator);

                const char* input_names[] = { input_name_ptr.get() };
                const char* output_names[] = { output_name_ptr.get() };

                try {
                    auto output_tensors = rec_sess->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
                    float* emb_ptr = output_tensors.front().GetTensorMutableData<float>();
                    size_t emb_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

                    std::vector<float> cur_emb(emb_ptr, emb_ptr + emb_len);

                    if (has_nan(cur_emb)) {
                        std::cerr << "[ERROR] 현재 frame embedding에 NaN 포함. 비교 불가\n";
                        continue;
                    }

                    double sim = cosine_similarity(registered_embedding, cur_emb);
                    std::cout << "[INFO] 유사도: " << sim << " -> " << (sim > 0.7 ? "✅ 인증 성공" : "❌ 인증 실패") << "\n";
                    print_prompt();
                }
                catch (const Ort::Exception& e) {
                    std::cerr << "[ERROR] 비교 처리 실패: " << e.what() << "\n";
                }

            }
            catch (const Ort::Exception& e) {
                std::cerr << "[ERROR] 비교 처리 실패: " << e.what() << "\n";
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

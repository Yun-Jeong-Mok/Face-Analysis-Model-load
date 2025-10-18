// 관련 라이브러리 curl, opencv4, onnx runtime 설치
// windows 64비트 운영체제 vcpkg 사용하여 라이브러리 설치
// 터미널 빌드 시 Developer Command Prompt for VS 2022에서 ( 기본 cmd, 파워쉘 불가 )
// cd <프로젝트 폴더명> 후 msbuild YourProject.sln /p:Configuration=Release /p:Platform=x64 실행
// R2 오브젝트 저장소에서 모델 메모리 로드 -> 이미지 조회, 등록 수행

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <curl/curl.h>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/core/utils/memory/stl/SimpleStringStream.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

// Read callback 구조체 (offset 관리)
struct BufferReader {
    std::vector<unsigned char> data;
    size_t offset = 0;
};

struct RegisteredUser {
    std::string id;            // 예: user_169...
    std::string image_url;     // R2 퍼블릭 URL
    std::vector<float> emb;    // (선택) embedding 캐시 — 필요 시 저장
};

// 등록 / 조회 시 비교 결과를 담기 위한 구조체
struct MatchResult {
    double best_sim = -1.0;
    std::string best_id;
};

std::vector<RegisteredUser> registry;

static size_t CurlReadFunc(void* ptr, size_t size, size_t nmemb, void* userdata) {
    BufferReader* br = reinterpret_cast<BufferReader*>(userdata);
    size_t maxCopy = size * nmemb;
    size_t remain = br->data.size() - br->offset;
    size_t toCopy = std::min<int>(maxCopy, remain);
    if (toCopy > 0) {
        memcpy(ptr, br->data.data() + br->offset, toCopy);
        br->offset += toCopy;
    }
    return toCopy;
}

namespace fs = std::filesystem;

// curl 다운로드 콜백 
static size_t WriteToVector(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::vector<unsigned char>* buffer = reinterpret_cast<std::vector<unsigned char>*>(userp);
    const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(contents);
    buffer->insert(buffer->end(), dataPtr, dataPtr + total);
    return total;
}

// aws sdk로 R2에 얼굴 이미지 등록
std::string upload_image_to_r2(const cv::Mat& img, const std::string& filename) {
    const std::string R2_ACCESS_KEY = "ee742980c2af4e584fa11b4d37477113";
    const std::string R2_SECRET_KEY = "7a87839f0cbda24b4770e814e141fd41541ea22097ce571eac4bec410f72532b";
    const std::string R2_ACCOUNT_ID = "1ecfa0016ba5bb85531224eacc1de5f2";
    const std::string BUCKET_NAME = "images";

    const std::string R2_S3_ENDPOINT = "https://" + R2_ACCOUNT_ID + ".r2.cloudflarestorage.com";
    const std::string R2_PUB_BASE = "https://pub-e0960a473cc94765a0d324cec3d9f0dd.r2.dev/images/faces/"; // GET 보낼 주소
    const std::string OBJECT_KEY = "faces/" + filename; // 버킷 내 실제 경로

    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", img, buf)) {
        std::cerr << "[ERROR] imencode 실패\n";
        return "";
    }

    Aws::Client::ClientConfiguration config;
    config.endpointOverride = R2_S3_ENDPOINT;
    config.scheme = Aws::Http::Scheme::HTTPS;

    config.region = "auto";

    Aws::Auth::AWSCredentials credentials(R2_ACCESS_KEY, R2_SECRET_KEY);
    Aws::S3::S3Client s3_client(credentials, config,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        false);

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(BUCKET_NAME);
    request.SetKey(OBJECT_KEY);
    request.SetContentType("image/jpeg");

    // cv::Mat 버퍼를 SDK 스트림으로 복사
    auto input_data = Aws::MakeShared<Aws::StringStream>("UploadStream");
    input_data->write(reinterpret_cast<const char*>(buf.data()), buf.size());
    request.SetBody(input_data);

    auto outcome = s3_client.PutObject(request);

    if (outcome.IsSuccess()) {
        std::string public_url = R2_PUB_BASE + filename;
        std::cout << "[INFO] R2 업로드 성공 (SDK): " << public_url << "\n";
        return public_url;
    }
    else {
        std::cerr << "[ERROR] R2 업로드 실패 (SDK): "
            << outcome.GetError().GetMessage() << "\n";
        return "";
    }
}

static size_t CurlWriteToVector(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::vector<unsigned char>* buf = reinterpret_cast<std::vector<unsigned char>*>(userp);
    const unsigned char* data = reinterpret_cast<const unsigned char*>(contents);
    buf->insert(buf->end(), data, data + total);
    return total;
}

cv::Mat download_image_from_r2(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ERROR] curl init 실패(다운로드)\n";
        return cv::Mat();
    }

    std::vector<unsigned char> buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteToVector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "[ERROR] 이미지 다운로드 실패: " << curl_easy_strerror(res) << "\n";
        return cv::Mat();
    }

    if (buffer.empty()) {
        return cv::Mat();
    }

    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return img;
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
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
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

std::vector<std::string> list_r2_faces() {
    // R2_ACCESS_KEY, R2_SECRET_KEY, R2_ACCOUNT_ID, BUCKET_NAME은
    // upload_image_to_r2 함수 내에 정의된 값과 동일하게 사용합니다.
    const std::string R2_ACCESS_KEY = "ee742980c2af4e584fa11b4d37477113";
    const std::string R2_SECRET_KEY = "7a87839f0cbda24b4770e814e141fd41541ea22097ce571eac4bec410f72532b";
    const std::string R2_ACCOUNT_ID = "1ecfa0016ba5bb85531224eacc1de5f2";
    const std::string BUCKET_NAME = "images";
    const std::string R2_S3_ENDPOINT = "https://" + R2_ACCOUNT_ID + ".r2.cloudflarestorage.com";
    const std::string OBJECT_PREFIX = "faces/"; // 조회할 경로

    Aws::Client::ClientConfiguration config;
    config.endpointOverride = R2_S3_ENDPOINT;
    config.scheme = Aws::Http::Scheme::HTTPS;
    config.region = "auto";

    Aws::Auth::AWSCredentials credentials(R2_ACCESS_KEY, R2_SECRET_KEY);
    Aws::S3::S3Client s3_client(credentials, config,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        false);

    Aws::S3::Model::ListObjectsV2Request request;
    request.SetBucket(BUCKET_NAME);
    request.SetPrefix(OBJECT_PREFIX); // "faces/" 경로 내의 객체만 조회

    auto outcome = s3_client.ListObjectsV2(request);
    std::vector<std::string> keys;

    if (outcome.IsSuccess()) {
        std::cout << "[INFO] R2 객체 목록 조회 성공.\n";
        for (const auto& object : outcome.GetResult().GetContents()) {
            keys.push_back(object.GetKey()); // 객체의 전체 키(예: faces/user_123.jpg)를 추가
        }
    }
    else {
        std::cerr << "[ERROR] R2 객체 목록 조회 실패: "
            << outcome.GetError().GetMessage() << "\n";
    }
    return keys;
}

int main() {
    Aws::SDKOptions options;
    Aws::InitAPI(options);

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

            std::string id = "user_" + std::to_string(std::time(nullptr));
            std::string filename = id + ".jpg";

            std::string public_url = upload_image_to_r2(frame, filename);
            if (public_url.empty()) {
                std::cerr << "[ERROR] 이미지 업로드 실패. 등록 취소\n";
                continue;
            }

            // (3) 모델로 embedding 생성 (기존 방식 재사용)
            Ort::Value input_tensor = mat_to_tensor(frame, mem_info, true);

            try {
                Ort::AllocatorWithDefaultOptions allocator;
                Ort::AllocatedStringPtr input_name_ptr = rec_sess->GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr output_name_ptr = rec_sess->GetOutputNameAllocated(0, allocator);
                const char* input_names[] = { input_name_ptr.get() };
                const char* output_names[] = { output_name_ptr.get() };

                auto output_tensors = rec_sess->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
                float* emb_ptr = output_tensors.front().GetTensorMutableData<float>();
                size_t emb_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
                std::vector<float> emb_vec(emb_ptr, emb_ptr + emb_len);

                if (has_nan(emb_vec)) {
                    std::cerr << "[ERROR] embedding 값에 NaN 포함. 등록 실패\n";
                    continue;
                }

                // (4) registry에 추가 (embedding은 캐시로 저장)
                RegisteredUser ru;
                ru.id = id;
                ru.image_url = public_url;
                ru.emb = emb_vec; // 선택: 캐시해두면 매번 R2 다운로드 안 해도 됨
                registry.push_back(std::move(ru));

                // (5) 사용자 알림
                std::cout << "[INFO] ✅ 등록 완료: " << id << " -> " << public_url << "\n";
                print_prompt(true);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "[ERROR] 등록 처리 실패: " << e.what() << "\n";
                continue;
            }
        }
        else if (key == 'c') {

            // 현재 프레임 임베딩 생성
            Ort::Value input_tensor = mat_to_tensor(frame, mem_info, true);
            std::vector<float> cur_emb;
            try {
                Ort::AllocatorWithDefaultOptions allocator;
                Ort::AllocatedStringPtr input_name_ptr = rec_sess->GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr output_name_ptr = rec_sess->GetOutputNameAllocated(0, allocator);
                const char* input_names[] = { input_name_ptr.get() };
                const char* output_names[] = { output_name_ptr.get() };

                auto output_tensors = rec_sess->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
                float* emb_ptr = output_tensors.front().GetTensorMutableData<float>();
                size_t emb_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
                cur_emb.assign(emb_ptr, emb_ptr + emb_len);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "[ERROR] 현재 프레임 임베딩 생성 실패: " << e.what() << "\n";
                continue;
            }

            if (has_nan(cur_emb)) {
                std::cerr << "[ERROR] 현재 frame embedding에 NaN 포함. 비교 불가\n";
                continue;
            }

            std::vector<std::string> object_keys = list_r2_faces();
            if (object_keys.empty()) {
                std::cout << "[INFO] 비교할 대상이 R2에 없습니다.\n";
                continue;
            }

            double best_sim = -1.0;
            std::string best_id;
            const std::string R2_PUB_BASE = "https://pub-e0960a473cc94765a0d324cec3d9f0dd.r2.dev/";

            for (const auto& key : object_keys) {
                std::string url = R2_PUB_BASE + key;
                cv::Mat ref_img = download_image_from_r2(url);
                if (ref_img.empty()) {
                    continue;
                }

                // (B) embedding 생성
                Ort::Value ref_tensor = mat_to_tensor(ref_img, mem_info, true);
                try {
                    Ort::AllocatorWithDefaultOptions allocator;
                    Ort::AllocatedStringPtr input_name_ptr = rec_sess->GetInputNameAllocated(0, allocator);
                    Ort::AllocatedStringPtr output_name_ptr = rec_sess->GetOutputNameAllocated(0, allocator);
                    const char* input_names[] = { input_name_ptr.get() };
                    const char* output_names[] = { output_name_ptr.get() };

                    auto out_t = rec_sess->Run(Ort::RunOptions{ nullptr }, input_names, &ref_tensor, 1, output_names, 1);
                    float* emb_ptr = out_t.front().GetTensorMutableData<float>();
                    size_t emb_len = out_t.front().GetTensorTypeAndShapeInfo().GetElementCount();
                    std::vector<float> ref_emb(emb_ptr, emb_ptr + emb_len);

                    if (has_nan(ref_emb)) {
                        continue;
                    }

                    double sim = cosine_similarity(ref_emb, cur_emb);

                    size_t last_slash = key.find_last_of('/');
                    std::string filename = (last_slash == std::string::npos) ? key : key.substr(last_slash + 1);
                    size_t last_dot = filename.find_last_of('.');
                    std::string user_id = (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);

                    if (sim > best_sim) {
                        best_sim = sim;
                        best_id = user_id;
                    }
                }
                catch (const Ort::Exception& e) {
                    continue;
                }
            } 

            double THRESH = 0.7;
            if (best_sim >= THRESH) {
                std::cout << "[INFO] 유사도 최고: " << best_sim << " -> ✅ 인증 성공 (ID: " << best_id << ")\n";
            }
            else {
                std::cout << "[INFO] 최고 유사도: " << best_sim << " -> ❌ 인증 실패\n";
            }

            print_prompt();
        }
    }

    cap.release();
    cv::destroyAllWindows();
    Aws::ShutdownAPI(options);
    return 0;
}

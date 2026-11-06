# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603401
- **Họ và tên:** Phạm Nguyễn Tất Đạt
- **SEED** (in ở ô đầu notebook): 5819

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.867 | 0.824 | 0.08 | 0 |
| 2. BoW + MLP | 0.947 | 0.947 | 4.99 | 0 |
| 3a. Encoder + kNN | 0.907 | 0.947 | 66.48 | 0 |
| 3b. Encoder + LogReg | 0.975 | 1.000 | 36.26 | 0 |
| 3c. Encoder + MLP | 0.975 | 1.000 | 32.93 | 0 |
| 4a. LLM nhỏ zero-shot (Groq 20b) | — | 0.952 | 2431 | 0 |
| 4b. LLM lớn zero-shot (Groq 120b) | — | 0.952 | 2443 | 0 |
| 4c. LLM nhỏ few-shot | — | 0.976 | 2597 | 0 |

## 3. Dự đoán và thực tế

1. `dl_curve`: đoán MLP F1 gần bằng logreg. Ra MLP 0.947, logreg 0.867. Lệch vì tôi nghĩ BoW đã chọn sẵn thì lớp ẩn không giúp, còn MLP vẫn học tổ hợp token tốt hơn trên test này.
2. `llm_models`: đoán model lớn F1 cao hơn nhỏ một chút. Ra 4a = 4b = 0.952. Lệch vì SUB 40 tin, 2 tin sai thì F1 hoà.
3. `tradeoff`: đoán BoW+LogReg và Emb+LogReg trên Pareto. Ra Pareto: BoW+LogReg, BoW+MLP, Emb+MLP. Emb+LogReg F1 bằng Emb+MLP nhưng p95 chậm hơn nên bị lấn át.

## 4. Phân tích 50 case khó

BoW+LogReg tụt nhiều nhất: F1 SUB 0.824 → F1 50 case 0.308, yếu nhất nhóm spam giọng tự nhiên (9%) và né từ khoá (14%).

1. `#2` ham: `FREE pizza in the office kitchen right now...` — 4d đoán spam vì chữ FREE + giục.
2. `#47` ham: `lol I just got a text saying I won £1000...` — 4d đoán spam vì trích từ won / call.

## 5. Thử thách cá nhân

Đề #6 (SEED 5819). Prompt JSON `{label, reason}` + parse, chạy 10 tin SUB, so bản 1 từ. Groq 20b.

| Bản | F1 | tok_in | tok_out |
| --- | ---: | ---: | ---: |
| 1 từ | 1.000 | 447 | 13 |
| JSON | 1.000 | 527 | 56 |

Input 1.18×, output 4.20×. Chỉ cần spam/ham thì bản 1 từ đủ. JSON đáng khi cần `reason`.

## 6. Phương án cho sản phẩm thật

Chọn Encoder+MLP: F1 test 0.975, p95 32.93 ms < 50 ms. BoW+MLP cũng đạt (F1 0.947, p95 4.99 ms). LLM p95 ~2431 ms, không đạt 10 triệu tin/ngày.

Đánh đổi: không LLM trên mọi tin. Router: model rẻ cho tin chắc, LLM chỉ tin lưỡng lự.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 2 | overfit, zero_grad, bỏ ReLU | val loss tăng = overfit; 2 Linear không ReLU = 1 Linear | tự viết `train_mlp`, assert 2.2 |
| Phần 3 | cosine vs dot, kNN vote | chia độ dài vector, argsort lớn nhất | tự viết cosine + kNN, assert 3.1–3.3 |
| Phần 4 | code `call_openai`, parse, batch; Gemini rồi Groq | hàm OpenAI-compatible, parse spam/ham | `not spam` trước `spam`; hết quota Gemini → `LLM_PROVIDER = groq`; assert 4.1–4.4 |

Accuracy ~0.97 dễ đẹp vì ham chiếm đa số, phải nhìn F1 spam. Test thường lạc quan: BoW F1 0.867 trên test, 0.308 trên 50 case khó. SUB 40 tin thì 4a và 4b hoà 0.952.

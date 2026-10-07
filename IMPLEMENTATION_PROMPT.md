# Prompt triển khai Day21 — Trần Quốc Vương, 2A202602522

Hoàn thiện bài Day21 theo README.md, HARDWARE-GUIDE.md và rubric.md; không chỉ viết kế hoạch. Chạy thí nghiệm thật, lưu evidence và viết báo cáo tiếng Việt khớp số đo. Không sửa dữ liệu/checker/test để nâng điểm; kết luận fine-tune không thắng cũng là kết quả hợp lệ.

## Hợp đồng triển khai

1. Kiểm tra repo, dependency, GPU/driver và các instructions trước khi viết. Máy hiện có RTX 3050 Ti 4 GiB; mặc định T4/4B không phù hợp. Ưu tiên chọn Qwen/Qwen3.5-0.8B theo quyền đổi BASE_MODEL của README, giữ cùng model và cấu hình phần cứng cho mọi nhánh. Nếu runtime không đủ, ghi rõ blocker và chuẩn bị Colab; không giả tạo training/evaluation artifacts.
2. Tạo môi trường Python riêng; không cần YesScale/OpenAI API key vì đây là fine-tuning model mở cục bộ. Pin dependency thực tế và ghi rõ các điều chỉnh cho Windows/driver; không thay môi trường Day19/Day20. Không commit .env, credentials, model cache hoặc full base weights.
3. Giữ corpus gốc và checksums. Chạy toàn bộ provided tests và smoke trước. NB1 tải tokenizer thật, giải mã loss mask, kiểm cả answer_is_supervised/question_is_masked, template thinking và p95; split 90/10 seed 42. Chạy check_mask_agreement trước training nếu đổi model.
4. NB2 chạy toàn bộ target/regression, không EVAL_LIMIT, cả naive và optimized prompt. Đóng băng baseline/corpus/prompt/config/timestamps trước NB3. Không làm yếu optimized prompt để adapter có vẻ thắng; không thay baseline sau khi thấy kết quả train.
5. NB3 train correct thật, assistant-only, hai epochs mặc định, cùng step budget NB4; lưu adapter ngay, loss curve, actual trainable counts, precision, peak VRAM và runtime. Mask train phải khớp NB1, training prompt phải khớp generation prompt. Báo cáo lựa chọn max_length so với p95.
6. NB4 train attn_only với matched_rank và chênh ngân sách dưới 5%, wrong_lr đổi duy nhất LR, qlora đổi precision/quantization theo thiết kế gốc. Cả bốn run cùng max_steps, nguồn data, seed và lịch optimizer. Không xếp hạng bằng train loss thay target. Nếu một nhánh lỗi, giữ log và sửa nguyên nhân hoặc báo chưa hoàn thành, không viết số giả.
7. NB5 chấm target/regression/format/latency cho base và adapter, autopsy cho cả ba contrasts tại precision đã train. Giữ cổng hồi quy gốc và raw completions. Phân tích ít nhất năm ví dụ, gồm hai ca FT thua nếu thực sự có; nếu không có đủ, báo số thật thay vì dựng ca thua.
8. NB6 optional chỉ thực hiện khi core hoàn tất: merge/score consistency và hot-swap hai adapters thật. Không nhận bonus dataset/rank/reasoning/HF nếu chưa có thí nghiệm tương ứng.
9. Viết submission/REPORT.md theo cấu trúc riêng: model/dataset/lý do, mask/template/p95, freeze evidence, ba baselines, bốn run/các câu hỏi autopsy, verdict >=100 từ, kết luận >=150 từ, định tính/raw evidence và limitations. Reflection phân biệt việc AI hỗ trợ với trải nghiệm cá nhân người học, không bịa lời tự thuật của người dùng.
10. Chạy scripts/verify.py, tests và audit secrets/data/experiment consistency. Tạo checklist với done/pending thật, hướng dẫn chạy lại từ máy sạch, package nộp có results và correct adapter; .gitignore gốc bỏ artifacts nên phải xử lý packaging/force-add có chọn lọc khi xuất bản. Không commit/push/nộp VLearn nếu chưa được yêu cầu riêng cho Day21.

## Điều kiện kết thúc

Core NB1–NB5 có artifacts được sinh thật, provided tests pass, gatekeeper được chạy và report khớp số. Nếu thiếu GPU/runtime hoặc external account thì hoàn thiện phần có thể kiểm chứng, chỉ rõ thiếu gì và cần thao tác nào; không coi missing training là verdict FAILED của một model đã được train.

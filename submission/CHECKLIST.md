# Checklist Day21 — Trần Quốc Vương / 2A202602522

## Core đã kiểm chứng

- [x] Đọc README, HARDWARE-GUIDE, rubric; tạo `IMPLEMENTATION_PROMPT.md` và thực hiện thay vì chỉ viết kế hoạch.
- [x] Môi trường Linux/CUDA riêng, GPU RTX 3050 Ti 4 GiB, model 0.8B khai báo rõ; dependency có pin và `pip check` pass.
- [x] NB1: tokenizer thật; template giữ thinking probe; mask 37/94, hai assert đúng; p95=98; split 225/25 seed 42.
- [x] Chạy mask-agreement diagnostic trước train; giữ FAIL của tokenizer mask và giải thích vì sao training dùng labels labkit đã chứng minh.
- [x] NB2 đầy đủ 50 target/15 regression; b=0,505 > a=0,000; optimized prompt không sửa; freeze trước NB3/NB4.
- [x] NB3 train thật, lưu `adapters/correct/`; 10.822.656 tham số; 58 bước thực tế; loss/runtime/peak VRAM có evidence.
- [x] NB4 cả attn_only, wrong_lr, qlora chạy thật; ngân sách tham số attention-only khớp 0%; cả bốn run cùng 58 actual steps.
- [x] NB5: ba baseline đủ bốn nhóm; cả ba đối chứng có target/format/latency; raw completions được lưu.
- [x] Verdict **FAILED**, target Δ=+0,480, regression Δ=−0,600; giữ gate gốc, không sửa threshold để biến thành PASS.
- [x] Báo cáo có model/data/lý do, mask/template, freeze, bảng bốn run, causal limitations, verdict trên 100 từ và kết luận trên 150 từ.
- [x] Định tính có năm ticket, gồm toàn bộ ba lỗi target, và **hai ca FT thua thật ở regression**. Target không có ca thua b: 48 thắng/0 thua/2 hòa; nếu rubric đòi riêng hai ticket thua thì tiêu chí đó chưa đạt, không bịa ví dụ.
- [x] Reflection mô tả evidence/AI assistance, không bịa trải nghiệm cá nhân của người học.
- [x] Dữ liệu/checksum/test/gate gốc không đổi; không dùng holdout để tuning; overlap input train/validation/target bằng 0.

## Kiểm tra cuối

- [x] `python -m pytest -rN`: **126 passed**; `submission/evidence/pytest.log`.
- [x] `python scripts/verify.py`: **26 passed · 1 warning · 0 failures**, exit 0, “Ready to submit”; warning duy nhất là model verdict FAILED, hợp lệ theo rubric.
- [x] `python scripts/audit_submission.py`: **21/21 checks OK**, chấm lại raw outputs; count tham số và steps thực tế khớp; merge/hot-swap cũng được kiểm.
- [x] `git diff --check` không có lỗi whitespace; `data/` không có thay đổi so HEAD; `.env`, cache, venv và merged weights bị ignore; không tìm thấy credential-shaped values trong file nộp.
- [x] Report, đầy đủ JSON/CSV, correct adapter và notebook/mã nguồn được chọn cho **Option A**; không kèm full base/merged weights.
- [x] ZIP **38,32 MiB** được giải nén trong thư mục tạm không có `.env`, cache hay state của máy tác giả: 126 tests pass, gatekeeper exit 0, audit exit 0; `submission/evidence/package-review.log`.

## Bonus và phạm vi

- [x] **B1 (+3 nếu reviewer duyệt)**: NB6 trên đủ 50 target, 0,985 trước merge → 0,990 sau merge, Δ=+0,005 trong tolerance; hot-swap 3 adapter trên một base.
- [ ] B2 custom dataset; B3 hai mask reasoning; B4 sweep rank; B5 public HF Hub: **không làm/không nhận điểm**.
- [x] Commit/push repo Day21 theo yêu cầu: https://github.com/Neon310304/Day21-Track3-Finetuning-Lab; chỉ kèm correct adapter, không kèm secret/cache/full weights.
- [ ] Nộp link/file vào VLearn: chưa thực hiện; người học cần nộp bằng tài khoản của mình.

Hướng dẫn chạy lại: `submission/REPRODUCE.md`. Gói nộp do `python scripts/package_submission.py` tạo tại `submission/lab21_2A202602522.zip`. Có thể xóa và dựng lại gói; không chạy đè các stage đã đo thành công để “làm đẹp” số liệu.

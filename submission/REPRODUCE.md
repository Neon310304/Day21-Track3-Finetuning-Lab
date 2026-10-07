# Chạy lại Day21 từ máy sạch

## Cấu hình của bài làm

- Người làm: Trần Quốc Vương — `2A202602522`.
- Base: `Qwen/Qwen3.5-0.8B`, snapshot Hugging Face `2fc06364715b967f1860aea9cf38778875588b17`.
- `COMPUTE_TIER=LAPTOP`, `BASE_MODEL=Qwen/Qwen3.5-0.8B`, `MASK_MODE=assistant-only`, `EPOCHS=2`; không đặt `EVAL_LIMIT`.
- Giữ recipe LAPTOP: `max_length=1024`, batch 1, accumulation 8. RTX 3050 Ti 4 GiB không đủ cho model mặc định 2B/4B, nên đổi **model**, không đổi dữ liệu/chấm điểm.
- Training thật chạy Linux trong Docker với GPU CUDA, Python 3.12.15, bf16. Không cần OpenAI/YesScale API key. Windows venv chỉ dùng kiểm tra CPU; không dùng nó để thay thế nhánh QLoRA Linux.
- `requirements-local-gpu.txt` pin stack trực tiếp; `submission/evidence/pip-freeze.txt` ghi toàn bộ dependency đã dùng. CUDA 11.8 tương thích driver 546.33 của máy. Không cài `torchao` trong venv sạch: yêu cầu gốc đó nhằm nâng phiên bản torchao cũ trên Colab; thí nghiệm này dùng bitsandbytes, không dùng torchao.

## Đọc và kiểm bài đã đo

Chạy tại gốc repo. Trước hết kiểm `submission/REPORT.md`, `results/`, `submission/evidence/` và `adapters/correct/`. Không lấy `.env` hoặc cache từ máy tác giả.

```powershell
$env:PYTHONIOENCODING = "utf-8"
docker build -f Dockerfile.local -t day21-lab:local .
docker run -d --name day21-review --gpus all --mount "type=bind,source=$((Get-Location).Path),target=/lab" --workdir /lab day21-lab:local sleep infinity
docker exec -e COMPUTE_TIER=LAPTOP -e BASE_MODEL=Qwen/Qwen3.5-0.8B -e MASK_MODE=assistant-only -e EPOCHS=2 day21-review python -m pytest tests/ -q
docker exec -e COMPUTE_TIER=LAPTOP -e BASE_MODEL=Qwen/Qwen3.5-0.8B -e MASK_MODE=assistant-only -e EPOCHS=2 day21-review python scripts/verify.py
docker exec day21-review python scripts/audit_submission.py
docker stop day21-review
```

`audit_submission.py` chấm lại **raw completions đã lưu**, không gọi model: kiểm score, verdict, số optimizer step thực tế, ngân sách tham số quan sát được, timestamp đóng băng và SHA256. Evidence `source-normalization.json` giữ cả SHA raw khớp manifest lúc thực thi và SHA sau chuẩn hóa CRLF→LF, để checkout Git khác hệ điều hành không bị coi là sửa logic. Không chuẩn hóa nội dung khác hay bỏ qua thay đổi code. Nó bổ sung bằng chứng, không thay thế gatekeeper gốc. Gatekeeper công nhận cả verdict FAILED nếu artifact và phép so sánh hợp lệ.

Không GPU: tạo venv, cài `requirements-cpu.txt` theo README rồi chạy pytest. Kiểm artifact/số liệu không cần CUDA; muốn huấn luyện lại cả QLoRA phải có Linux/CUDA. Docker cần NVIDIA Container Toolkit hoặc Docker Desktop/WSL2 có GPU passthrough.

## Huấn luyện lại trong một bản sao riêng

Không chạy đè bài đã đo: runner cố ý từ chối overwrite stage thành công. Tạo bản sao riêng của mã nguồn và dữ liệu gốc; bỏ các artifact sinh ra trong `results/`, `adapters/`, `data/split/`, `submission/evidence/` của **bản sao đó**, không xóa bản bài nộp. Đặt bốn biến cấu hình trên bằng environment hoặc `.env` riêng. Các folder sẽ được tạo lại.

Để không tải nhầm revision mới của nhánh Hub `main`, tải snapshot đã đo **trước** NB1, rồi đặt `BASE_MODEL` thành đường dẫn snapshot in ra. Ví dụ trong Linux container:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17'))"
export BASE_MODEL=/cache/huggingface-day21/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export HF_HUB_OFFLINE=1
```

Giữ `BASE_MODEL` đó cho **mọi** stage, kể cả tokenizer NB1. Manifest mới sẽ ghi local snapshot path thay cho Hub ID; đó là cùng trọng số, không phải đổi model giữa các nhánh. Trên Windows dùng đường dẫn snapshot thực tế và `$env:BASE_MODEL`, `$env:HF_HUB_OFFLINE`. Không đổi `BASE_MODEL` bên trong bài đã freeze; chỉ áp dụng ở bản sao thí nghiệm sạch.

```bash
python scripts/verify.py --smoke
python scripts/check_mask_agreement.py
python scripts/run_submission.py nb1 nb2
python scripts/run_submission.py nb3 nb4 nb5
python scripts/audit_submission.py
python scripts/verify.py
```

Đọc baseline (b) trước khi gọi NB3. Các run dùng cùng tokenizer/model, toàn bộ 50 target và 15 regression, split seed 42, hai epochs; ba đối chứng không được hưởng thêm step. Muốn thử prompt/dataset/epochs khác phải tạo **thí nghiệm khác**, không thay artifact đóng băng hiện tại. Bonus NB6 nếu chạy phải thực hiện sau core, không dùng nó để chỉnh training.

`check_mask_agreement.py` có thể in **FAIL** vì template Qwen không chứa `{% generation %}`: tokenizer mask rỗng, còn TRL tự vá sang mask khác. Đây là diagnostic được giữ nguyên. Đường training dùng `labkit.data.to_training_dataset()` với labels đã được NB1 chứng minh; không bật `assistant_only_loss` và không packing. Vì thế diagnostic đó không phải bằng chứng training dùng mask rỗng.

## LF và tính tái lập

`.gitattributes` ép LF cho JSONL/Python/notebook. Checksum trong lab đo **bytes**, nên checkout CRLF của Windows gây FAIL giả dù nội dung hiển thị không đổi. Bài này khôi phục đúng blob gốc bằng `git archive HEAD data`, không đổi dữ liệu hoặc checksums. Sau khôi phục, SHA của bốn file đều trùng bản gốc. Không mở/khai thác holdout để tuning.

Model revision trong `baselines_frozen.json` có thể là `null` vì Transformers không giữ `_commit_hash` ở config runtime. Revision thực tế được ghi trong tài liệu này và evidence cache; khi tái lập về sau, dùng đúng snapshot đó thay vì tin rằng nhánh `main` trên Hub luôn bất biến. Greedy decode giảm sampling noise nhưng CUDA/library/hardware có thể làm khác prediction ở trường hợp sát biên; latency và VRAM không phải hằng số giữa các máy.

## Phạm vi đóng gói

Option A: report, tất cả JSON/CSV trong `results/`, adapter chính, notebooks và mã nguồn để kiểm tra. Không nộp `.env`, `.venv`, model cache, base/merged weights hoặc adapter đối chứng lớn. Các output trên là số đo thật; không thay chúng bằng bảng mẫu trong docs của upstream.

```bash
python scripts/package_submission.py
```

Lệnh tạo `submission/lab21_2A202602522.zip`. `.gitignore` đã cho phép các result JSON/CSV và đúng ba file chính của `adapters/correct/`; các adapter khác và full merged model vẫn bị ignore. ZIP không được commit. Giữ adapter gốc trong bài nộp dù verdict FAILED: chính nó là vật chứng để reviewer kiểm lại thí nghiệm.

ZIP đã được giải nén vào thư mục tạm riêng, không `.env` hay model cache: pytest, verify và audit đều exit 0. Evidence kiểm tra gói nằm trong `submission/evidence/package-review.log` và `package-final-review.log`. Sau khi kiểm tra có thể dừng container dành riêng cho Day21; không cần Neo4j hay dịch vụ Day19/Day20 để kiểm bài này.

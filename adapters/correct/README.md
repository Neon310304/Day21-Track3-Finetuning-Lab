---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- vietnamese
---

# Day21 — Vietnamese ticket triage adapter

Trần Quốc Vương — MSSV 2A202602522. Adapter thực nghiệm cho Lab21, không phải bản được phê duyệt triển khai. **Regression gate FAILED** dù target accuracy tăng.

## Tác vụ và dữ liệu

Input: ticket CSKH tiếng Việt. Output: JSON có `intent`, `urgency`, `product`, `sentiment`. Corpus gốc của lab có 250 mẫu tổng hợp; split 225 train / 25 validation, seed 42. Không đánh giá checkpoint trên holdout, không sửa dữ liệu nguồn. Trọng số base và tokenizer không nằm trong gói adapter; tải đúng base từ Hugging Face.

## Recipe đã chạy

- Base revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- LoRA text-linear, r=16, alpha=32, dropout=0, LR=1e-4, cosine, warmup 6 bước.
- Batch 1, accumulation 8, hai epochs, **58 actual optimizer steps**, max_length ceiling 1024.
- Loss mask assistant-only từ labels tiền-token hóa đã kiểm bằng NB1; không packing.
- GPU RTX 3050 Ti 4 GiB; bf16 base/compute, saved adapter weights F32.
- 10.822.656 tham số trainable, 43.340.856 bytes safetensors; train 886,9 s, peak allocated 3,07 GiB.
- Torch 2.7.1+cu118, Transformers 5.15.1, TRL 1.10.0, PEFT 0.20.0.

## Đánh giá và giới hạn

50 target / 15 regression, greedy decode. Target field accuracy **0,985**, format key coverage **1,000**, regression keyword recall **0,0667**, latency **1181,7 ms/mẫu** (batch 4). Base với prompt tối ưu đạt target 0,505, regression 0,6667, latency 788,1 ms/mẫu. Gate thất bại vì regression Δ=−0,600, vượt tolerance 0,020.

Không dùng như trợ lý kiến thức tổng quát: câu hỏi ngoài miền bị trả thành JSON triage. Số đo trên dữ liệu tổng hợp nhỏ/một seed không chứng minh chất lượng ticket thực tế, độ an toàn, hoặc độ công bằng. Không dùng loss 0,3936 để tuyên bố model sẵn sàng ship. Full evidence và phân tích: [báo cáo](../../submission/REPORT.md), [cách chạy lại](../../submission/REPRODUCE.md).

## Tải để nghiên cứu

Chạy từ gốc repo với môi trường GPU trong hướng dẫn tái lập; không cần API key:

```python
import os
import sys

os.environ["BASE_MODEL"] = "Qwen/Qwen3.5-0.8B"
sys.path.insert(0, "src")

from labkit import generate
from labkit.config import get_tier
from peft import PeftModel

model, tokenizer = generate.load_base(get_tier("LAPTOP"))
model = PeftModel.from_pretrained(model, "adapters/correct")
model.eval()
outputs, latency = generate.generate_batch(
    model, tokenizer,
    ["Shop ơi, mình muốn trả lại chuột không dây. Gấp. Cảm ơn shop."],
    system=generate.NAIVE_PROMPT,
)
print(outputs[0])
```

Không có Hugging Face Hub upload trong bài này. Không dùng merged checkpoint để thay đổi số core cao hơn; bản adapter gốc này được giữ để kiểm chứng kết quả, kể cả verdict FAILED.

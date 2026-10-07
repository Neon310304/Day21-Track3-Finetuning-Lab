# Day21 — Prompt tốt có còn thua fine-tune không?

**Trần Quốc Vương · MSSV 2A202602522 · 07/10/2026.** Báo cáo do AI hỗ trợ soạn từ thí nghiệm chạy thật; số T4 mẫu của upstream không dùng làm kết quả bài nộp.

## 1. Thiết kế phép đo và lựa chọn thực tế

Tác vụ là chuyển ticket CSKH tiếng Việt thành JSON `intent`, `urgency`, `product`, `sentiment`. Chọn corpus gốc vì có nhãn khách quan và scorer từng trường, không cần LLM judge hay API trả phí. Giữ nguyên 250 mẫu nguồn, chia 225 train/25 validation seed 42; đánh giá trên toàn bộ 50 target và 15 regression. Validation chỉ được tách theo scaffold, không dùng để chọn checkpoint hay tune LR; holdout không được đọc để tối ưu.

Model là **Qwen/Qwen3.5-0.8B**, snapshot `2fc06364715b967f1860aea9cf38778875588b17`. RTX 3050 Ti laptop chỉ có **4 GiB** VRAM nên 2B/4B mặc định không phù hợp. README cho đổi `BASE_MODEL`; cả baseline và bốn adapter đều dùng cùng 0.8B, không lấy baseline của model lớn hơn. Recipe LAPTOP giữ nguyên batch 1, accumulation 8, effective batch 8, max_length 1024; EPOCHS=2, không EVAL_LIMIT. Precision thực tế bf16 trên sm86; QLoRA đổi base sang NF4 double quantization, compute vẫn bf16.

Training chạy Linux/Docker, Python 3.12.15, torch 2.7.1+cu118, Transformers 5.15.1, TRL 1.10.0, PEFT 0.20.0, bitsandbytes 0.50.2. Stack pin ở `requirements-local-gpu.txt` và freeze ở `submission/evidence/`. Windows CPU test không thay thế bằng chứng CUDA. Không dùng key YesScale/OpenAI. Checksum Windows ban đầu lệch vì CRLF; đã khôi phục bytes từ blob Git gốc, không sửa nhãn, corpus hay checker. `.gitattributes` tránh lặp lỗi trên máy khác.

## 2. NB1: chứng minh phần đang được học

`results/token_stats.json`: mean **93,1**, p50 **93**, p95 **98**, p99 **100**, max **101** trên 250 mẫu; gợi ý max_length **256**. Bài giữ ceiling **1024** của tier để không thêm một biến phần cứng so với recipe gốc. Vì tất cả mẫu ngắn hơn 1024, không bị truncation, batch 1 và padding động không pad mọi mẫu lên 1024; đây không phải tuyên bố rằng p95 đòi 1024. Nếu tối ưu deployment sau lab, 256 là ceiling nhỏ hơn có cơ sở đo.

`results/mask_proof.json` chứng minh mẫu kiểm có **37/94 token**, fraction **0,3936**, `answer_is_supervised=true`, `question_is_masked=true`. Phần được tính loss:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

System prompt, ticket và generation prefix `<think>\n\n</think>` bị mask; EOS của câu trả lời được học. Đối chứng `everything` trong NB1 supervise 94/94, là cách sai với mục tiêu này. Trên toàn train, **8.564/20.951 token (40,9%)** có loss; không phải proof trên một mẫu rồi train cả prompt.

`results/template_check.json` giữ nguyên khối `<think>` có nội dung của probe, `ok=true`. Tuy nhiên corpus triage chỉ có JSON, không có reasoning traces. `scripts/check_mask_agreement.py` phát hiện template không có `{% generation %}`: tokenizer trả **0/31** token supervise; TRL tự vá thành **13/31** gồm empty thinking block, khác **9/31** của labkit. Diagnostic in FAIL được giữ nguyên trong evidence. Training NB3/NB4 dùng `data.to_training_dataset()` với labels tiền-token hóa đúng mask NB1; không bật `assistant_only_loss`, packing hoặc padding_free. Mask đúng và prompt training/eval cùng `NAIVE_PROMPT` là điều kiện cần, không bảo đảm accuracy cao.

## 3. NB2: baseline đã mạnh hơn trước khi train

Prompt tối ưu **không sửa**; SHA rút gọn `719e74d3b6232053`, đầy đủ trong manifest. Naive prompt chỉ yêu cầu “Phân loại ticket sau.”; optimized prompt định nghĩa schema, vocabulary và ví dụ. Kết quả trước training: (a) target **0,000**, format **0,000**, regression **0,6667**, latency **3030,1 ms/mẫu**; (b) target **0,505**, format **1,000**, regression **0,6667**, latency **788,1 ms/mẫu**. Như vậy (b) thật sự hơn (a), không cần làm yếu baseline để adapter trông thắng.

`results/baselines_frozen.json` lưu toàn bộ điểm; `baseline_predictions.json` giữ completion của cả target/regression. `experiment_manifest.json` ghi thời gian đóng băng, SHA256 baseline/prompt/data và thời điểm bắt đầu từng stage. `scripts/run_submission.py` từ chối train khi chưa freeze, cấu hình đổi, baseline bị sửa hoặc stage đã thành công bị chạy đè. `model_revision=null` trong config runtime không được biến thành một revision giả; snapshot thực được xác nhận riêng bằng `submission/evidence/model-snapshot.json`.

Mốc đóng băng **2026-10-07 06:47:01.982954 UTC**, trước NB3 **06:47:49.836110 UTC**, trước NB4 **07:04:31.130149 UTC**. SHA256 file đóng băng: `9554d8b3108485088ad6ff6d06f15d803317aef534e17b260eb815dcae9e68e1`. Không thay prompt, dữ liệu hoặc recipe sau khi thấy điểm fine-tune.

## 4. NB3/NB4: cùng ngân sách, khác vị trí/LR/quantization

Model có 24 lớp: **18 linear-attention, 6 full-attention**. `resolve_target_modules()` lấy 12 loại linear trong text decoder, không vision tower/lm_head. LoRA dropout 0, alpha/r=2, seed 42, cosine scheduler và warmup 6 optimizer steps; cùng 225 mẫu và mask đã kiểm.

| Run | Vị trí | r / alpha | Tham số trainable | LR | Train loss trung bình | Target NB5 | Train giây | Peak VRAM GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 / 32 | 10.822.656 | 0,0001 | 0,3936 | **0,985** | 886,9 | 3,07 |
| attn_only | q,v | 271 / 542 | 10.822.656 | 0,0001 | 0,4343 | 0,945 | 741,7 | 3,08 |
| wrong_lr | text-linear | 16 / 32 | 10.822.656 | 0,00001 | 1,5419 | 0,330 | 874,5 | 3,07 |
| qlora | text-linear, NF4 | 16 / 32 | 10.822.656 | 0,0001 | 0,4239 | 0,950 | 914,7 | 2,29 |

Nguồn: `results/runs.csv`, `autopsy.json`, `training_*.json`. Cả **bốn run thực sự kết thúc ở 58/58 optimizer steps**, không chỉ khai báo budget bằng nhau. Tổng train time 3.417,8 giây; không bao gồm loading/eval/setup. NB3 chạy 2 epochs, NB4 dùng max_steps=58 có cùng điểm dừng. `observed_trainable_params` và số phần tử trong safetensors đều bằng 10.822.656, chênh ngân sách attention-only **0%**. Nếu giữ q,v r=16 thì chỉ có **638.976** tham số, ít hơn **16,9375 lần**; đó sẽ là đối chứng không công bằng.

`final_loss` trong CSV là `TrainOutput.training_loss` **trung bình cả run**, không phải loss của minibatch cuối. Mọi grad_norm/loss quan sát đều hữu hạn. Loss ở các mốc thật:

| Run | step 5 | step 20 | step 40 | step 55 |
|---|---:|---:|---:|---:|
| correct | 2,8057 | 0,1282 | 0,0239 | 0,0079 |
| attn_only | 2,7401 | 0,1827 | 0,0455 | 0,0201 |
| wrong_lr | 3,0250 | 1,8186 | 1,0173 | 0,7779 |
| qlora | 2,8857 | 0,1525 | 0,0345 | 0,0190 |

### 4.1 Vị trí so với rank

Theo target, **correct > qlora > attn_only > wrong_lr**; theo loss trung bình thấp hơn, thứ tự cũng như vậy. Không có đảo hạng giữa loss và target trong lần đo này, nên không kể câu chuyện “loss thấp hơn mà task tệ hơn” khi evidence không có. Nhưng loss vẫn không thể thay target hay regression: correct có loss 0,3936 mà vẫn làm hỏng nghiêm trọng câu hỏi phổ thông. Cùng 10.822.656 tham số, attn_only r=271 thua correct r=16 **4 điểm phần trăm target**; tăng rank ở q,v không bù được độ phủ các lớp linear/MLP/linear-attention trên tác vụ này. Đây là evidence cho **placement dưới ngân sách cố định**, không chứng minh rank luôn vô dụng hoặc mọi dataset đều cần all-linear. Chỉ có một seed/model nhỏ; chưa có sweep rank độc lập, nên không nhận bonus B4.

### 4.2 LR sai không đồng nghĩa loss phẳng

wrong_lr chỉ giảm LR từ 1e-4 xuống 1e-5; placement, rank, dữ liệu, mask và step giữ nguyên. Ở step 20 loss **1,8186** thay vì **0,1282**; step 55 vẫn **0,7779** thay vì **0,0079**, target chỉ **0,330** và thậm chí thua prompt (b) **0,505**. Loss của wrong_lr **có giảm**, không “phẳng từ đầu” như dự đoán đơn giản trong hướng dẫn; kết luận đúng là học quá chậm dưới budget 58 bước. Nếu không biết LR, có thể đổ oan cho khả năng LoRA hoặc cho rằng thêm rank chắc chắn giải quyết được. Chưa chạy wrong_lr với nhiều step hơn, nên không biết nó có đuổi kịp hay không; thêm step sẽ là một thí nghiệm khác.

### 4.3 QLoRA đổi tài nguyên lấy chất lượng và latency

Peak memory được đo bằng `torch.cuda.max_memory_allocated()`, đơn vị GiB: **3,07 → 2,29**, giảm **0,78 GiB (25,4%)** so correct; không phải tổng VRAM theo nvidia-smi. Đổi lại target **0,985 → 0,950**, giảm **3,5 điểm phần trăm**; train time tăng **3,1%**, latency **1181,7 → 1567,8 ms/mẫu**, tăng **32,7%**. Điểm target của QLoRA vẫn cao hơn attention-only 0,945; không gọi nó là nhánh tệ nhất. Trong giới hạn máy này, số đo ủng hộ bf16 LoRA nếu đủ RAM và ưu tiên quality/latency, nhưng QLoRA có thể đáng dùng khi 0,78 GiB quyết định chạy được hay OOM. Không suy rộng lời khuyên vendor sang mọi kích thước Qwen/hardware.

Giới hạn causal: bật k-bit khiến thư viện dùng LoRA weights **BF16**, trong khi các run 16-bit lưu LoRA weights **F32**; `adapter_evidence.json` xác nhận 372 tensors F32 ở correct so 372 BF16 ở qlora. “precision=bf16” trong CSV mô tả base/compute, không có nghĩa mọi tham số đều BF16. Đây là hệ quả của đường QLoRA chuẩn của TRL/PEFT, không phải một tham số thí nghiệm được sửa thêm; vì vậy không tách hoàn toàn quantization error khỏi khác biệt dtype adapter. NB5 chấm QLoRA trên base NF4 đúng như lúc train, không dùng base bf16 rồi gán lỗi mismatch cho quantization.

## 5. NB5: phán quyết bằng tác vụ và khả năng phổ thông

| Run | Target | Regression | Format | Latency ms/mẫu | n target |
|---|---:|---:|---:|---:|---:|
| (a) base + naive | 0,000 | 0,6667 | 0,000 | 3030,1 | 50 |
| (b) base + optimized | 0,505 | 0,6667 | 1,000 | 788,1 | 50 |
| (c) correct LoRA + naive | **0,985** | **0,0667** | 1,000 | 1181,7 | 50 |

Target là trung bình độ đúng của **4 trường**, không phải 98,5% ticket hoàn toàn đúng: correct đạt **197/200 trường**, **47/50 object đúng toàn bộ**. Format là tỷ lệ coverage của bốn khóa từ loose JSON parser, không phải chứng chỉ output JSON strict. Regression là keyword recall trên 15 câu, cùng prompt không system cho cả base và FT. Latency là tổng thời gian `model.generate()` chia số mẫu, batch 4, greedy, giới hạn 160 token target/96 regression; không phải p95 request đơn lẻ hay chi phí end-to-end có loading/tokenization. correct chậm hơn (b) **49,9%** trong lần đo; prompt ngắn không tự bảo đảm inference nhanh khi còn overhead adapter.

**Gate FAILED:** target Δ **+0,480**, regression Δ **−0,600**, tolerance giữ nguyên **0,020**. `valid_trace_rate=0,000`; đây không chứng minh reasoning-trace collapse vì corpus không train traces và generation tắt thinking. Không nhận bonus B3.

Diễn giải: Fine-tune đã học được mapping ticket→JSON rất tốt trên target, không chỉ khiến output parse được: format của baseline (b) vốn đã 1,000, nhưng target của nó mới 0,505. Sự tăng lên 0,985 vì thế là cải thiện nội dung bốn trường trên tập đang đo. Tuy nhiên, cổng yêu cầu đồng thời bảo toàn khả năng phổ thông, và điểm regression tụt từ hai phần ba xuống một phần mười lăm. Chênh lệch −0,600 lớn gấp 30 lần ngưỡng dung sai 0,020. Các completion cho câu sinh nhật và toán học bị thay bằng JSON triage sai tác vụ, nên đây không chỉ là nhược điểm keyword matching. Corpus train toàn ticket có thể làm adapter chuyên biệt hóa quá mạnh; hiện tượng phù hợp với quên/over-specialization, nhưng chưa có thí nghiệm replay để chứng minh can thiệp sẽ chữa được. Không deploy adapter này như trợ lý tổng quát, không nới gate hay thay prompt (b) sau training. Một hệ thống chỉ triage có router và base fallback có thể là hướng nghiên cứu khác, chưa được benchmark ở đây. FAIL là **kết quả của model**, khác với lỗi môi trường hay bài chưa train.

## 6. Định tính: cả thành công, lỗi target và ca thua thật

Index dưới đây là **0-based** trong bộ gốc. Mỗi tuple theo thứ tự `intent; urgency; product; sentiment`; full input, JSON và score được lưu trong `results/submission_audit.json` và hai file predictions. Bảng không chọn mỗi ca thắng hoàn toàn: có cả ba lỗi target duy nhất của correct.

| Index | Ticket rút gọn | Nhãn | Baseline (b) | Correct FT | Score b→FT và giải thích |
|---|---|---|---|---|---|
| 0 | Chuột không dây; “Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.” | doi_tra; cao; chuột không dây; tich_cuc | hoan_tien; cao; chuột không dây; tich_cuc | doi_tra; cao; chuột không dây; tich_cuc | 0,75→1,00; FT phân biệt trả hàng với hoàn tiền |
| 1 | Ốp lưng; “Hoàn tiền. Sớm nhé. Bực mình.” | hoan_tien; trung_binh; ốp lưng điện thoại; tieu_cuc | hoan_tien; cao; ốp lưng điện thoại; tich_cuc | hoan_tien; trung_binh; ốp lưng điện thoại; tieu_cuc | 0,50→1,00; b sai urgency/sentiment; khoảng trắng đầu product được scorer strip |
| 12 | Áo khoác gió; “Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.” | san_pham_loi; thap; áo khoác gió; tich_cuc | van_chuyen; cao; áo khoác gió; tich_cuc | doi_tra; thap; áo khoác gió; tich_cuc | 0,50→0,75; FT vẫn sai intent, dù hơn b |
| 18 | Máy xay; “Khi nào có tiền về. Mong shop phản hồi. Rất thất vọng.” | hoan_tien; trung_binh; máy xay sinh tố; tieu_cuc | hoan_tien; cao; máy xay sinh tố; tich_cuc | hoan_tien; thap; máy xay sinh tố; tieu_cuc | 0,50→0,75; FT hạ urgency sai |
| 23 | Nồi chiên; “Giao hàng chậm. Mong shop phản hồi. Lần cuối mua ở đây.” | van_chuyen; trung_binh; nồi chiên không dầu; tieu_cuc | van_chuyen; cao; nồi chiên không dầu; trung_tinh | van_chuyen; thap; nồi chiên không dầu; tieu_cuc | 0,50→0,75; cùng pattern urgency với #18 |

**Không bịa hai ca target thua:** kiểm tất cả 50 target cho kết quả **48 ca FT thắng, 0 ca thua, 2 ca hòa** theo scorer từng trường. Ba ca FT sai ở trên vẫn hơn b, nên không được đổi nhãn thành “FT thua”. `qualitative.json` upstream chỉ sort lỗi FT; muốn chứng minh thua phải đối chiếu completion b của cùng input. Hai ca thua thật sau thuộc **regression**, được nêu rõ phạm vi. Nếu reviewer yêu cầu riêng hai *ticket target* FT thua, tập đo này không cung cấp chúng; không sửa dữ liệu hoặc lựa chọn model hậu nghiệm để tạo ca đó.

### Hai ca FT thua baseline (b) ở regression

**Regression #3 — “Viết một câu chúc mừng sinh nhật bằng tiếng Việt.”**

- Keyword đúng: `sinh nhật`.
- Baseline b nguyên văn: `Chúc mừng sinh nhật! 🎉 Chúc bạn luôn tràn đầy năng lượng, hạnh phúc và những kỷ niệm đẹp nhất trong đời.`
- FT nguyên văn: `{"intent": "hoi_thong_tin", "urgency": "thap", "sentiment": "tieu_cuc"}`.
- Score **1→0**. FT không viết câu chúc mà áp schema ticket vào nhiệm vụ không liên quan; còn thiếu product, nhưng format target 1,000 không phát hiện được lỗi ngoài miền này.

**Regression #6 — “2 mũ 10 bằng bao nhiêu?”**

- Keyword đúng: `1024`.
- Baseline b có đoạn nguyên văn `$$2^{10} = 2 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 1024$$`; full completion trong predictions.
- FT nguyên văn: `{"intent": "hoi_thong_tin", "urgency": "thap", "urgency_score": 0}`.
- Score **1→0**. Không trả lời phép tính và tự sinh khóa mới `urgency_score`; đây là lệch tác vụ, không phải thiếu dấu tiếng Việt.

Có **10/15 câu regression FT thua b** theo keyword scorer. Đặc điểm chung là ưu tiên dạng JSON triage ngay cả khi user không yêu cầu phân loại. Ngược lại, hai lỗi urgency trên target cho thấy phrase “Mong shop phản hồi” chưa được học chắc ở mọi ngữ cảnh. Không dùng các lỗi vừa thấy để thay label hoặc retrain bài đã đóng băng.

## 7. Kết luận: tăng accuracy không đồng nghĩa sẵn sàng ship

Không nên deploy adapter correct hiện tại như một trợ lý dùng chung. Nó cải thiện tác vụ hẹp thêm 48 điểm phần trăm so với prompt tốt, nhưng trả giá bằng mất 60 điểm phần trăm regression và tăng gần 50% thời gian generate mỗi mẫu so với baseline (b). Quan trọng hơn, các câu hỏi ngoài miền bị chuyển thành JSON triage, nên rủi ro là hành vi hệ thống đã đổi, không chỉ một vài trường bị sai. Nếu sản phẩm chỉ nhận ticket đã được xác thực, có thể thử một tầng routing dùng adapter chuyên biệt và fallback model nền cho ngoài miền; đó là đề xuất, chưa phải cấu hình đạt gate hay kết quả triển khai đã kiểm.

Đòn bẩy quan sát được là LR đúng thang và placement phủ text-linear: giảm LR mười lần làm target xuống 0,330; tập trung toàn bộ ngân sách vào q,v dù rank 271 chỉ đạt 0,945 thay vì 0,985. Mask và prompt alignment là nền tảng để các so sánh ấy có nghĩa, chứ không phải biến đã ablate riêng trong bốn run. QLoRA giảm 25,4% allocated VRAM nhưng giảm target và chậm hơn; quyết định dùng nó phụ thuộc GPU có đủ chạy bf16 hay không. Corpus nhỏ, tổng hợp và chỉ một seed không bảo đảm generalization sang ticket thật. Trước khi triển khai, cần đánh giá dữ liệu thật sạch, bổ sung outside-domain/replay và đo lại bằng một thí nghiệm mới có freeze riêng. Không kết luận rằng LoRA luôn gây quên hoặc rằng replay chắc chắn giải quyết được vì chưa đo can thiệp đó. Bài học chính là phải biết **vì sao không ship**, ngay khi training loss rất đẹp.

Ba điều rút ra cụ thể: (1) proof mask phải dùng đúng đường labels đang train; (2) matched rank cần count thực tế, không dựa vào r bằng nhau; (3) score target/format không thay được kiểm tra hành vi ngoài miền. Nếu có thêm hai giờ, ưu tiên một thí nghiệm replay 1–5% riêng và external eval, không tăng rank để cố chữa một regression chưa chẩn đoán. Phản tư AI-assisted chi tiết nằm ở `submission/REFLECTION.md`.

## 8. Giới hạn phép đo và kiểm chéo

- Tất cả data checksum gốc giữ nguyên; overlap input chính xác train/validation, train/target, validation/target đều **0**. Tuy nhiên cùng nguồn template tổng hợp có thể khiến phân phối target dễ hơn dữ liệu thực; exact dedup không bảo đảm độc lập semantic.
- Regression keyword recall có thể cho điểm câu chứa keyword nhưng giải thích sai. Ví dụ b nhắc 1000 ở câu đổi km nhưng diễn giải rối; không coi 0,6667 là chứng nhận factual accuracy. Hai ca thua minh họa đã kiểm nội dung, không chỉ nhìn aggregate.
- Chỉ correct được chấm đủ regression; ba đối chứng chấm target/format/latency theo scaffold. Không suy ra qlora hoặc attn_only giữ khả năng phổ thông tốt hơn khi chưa đo.
- `scripts/audit_submission.py` chấm lại raw completions, kiểm artifact freeze, actual steps, count adapter và toàn bộ original checksums; không thay cổng hay dữ liệu. Report/số liệu được đối chiếu với JSON/CSV, không lấy con số của notebook mẫu.

## 9. Bonus NB6 và bài nộp

**B1 đã chạy thật:** trên đủ 50 target, trước merge **0,985**, sau merge **0,990**, Δ **+0,005**, không tụt quá tolerance **0,010**. Một trường đổi đúng không có nghĩa merge huấn luyện thêm hay chắc chắn cải thiện; phép cộng vào base bf16 làm rounding và token sát biên thay đổi. Giữ `merge_predictions.json` để kiểm lại, không thay score 0,985 của correct trong bảng core bằng score sau merge cao hơn. Merge đạt kiểm tra target, **chưa** chứng minh chữa được regression ngoài miền.

Hot-swap **correct, attn_only, qlora** trên cùng một base bf16, cùng ticket #0. `hot_swap.json` lưu completion: correct nhận doi_tra/cao, attn_only doi_tra/trung_binh, qlora hoan_tien/cao. Đây là demo chuyển adapter, không phải benchmark QLoRA đúng precision: adapter qlora ở demo được gắn vào base bf16; so sánh chính ở NB5 dùng NF4. NB6 giải phóng cả tham chiếu PEFT và merged trước reload để GPU 4 GiB không giữ hai base đồng thời. Không nhận bonus B2/B3/B4/B5 vì không có corpus riêng, hai mask reasoning, sweep rank hoặc upload HF Hub.

Bài được đóng gói **Option A**, adapter correct **43.340.856 bytes (~41,3 MiB)**, không cố khớp ví dụ ZIP 5–15 MB của rubric. Có toàn bộ JSON/CSV, notebook đã clear output, code/test, dependency pin, report/reflection, log và hướng dẫn chạy lại. Không đưa `.env`, `.venv`, cache, base/merged weights hoặc adapter đối chứng vào ZIP. Trạng thái kiểm tra cuối và phạm vi chưa thực hiện được ghi ở `submission/CHECKLIST.md`.

Kiểm cuối: **126 tests pass**; gatekeeper **26 passed, 1 warning, 0 failures**, exit 0; audit bổ sung **21/21 checks OK**. Warning của gatekeeper nhắc phân tích verdict FAILED, không phải lỗi khiến bài chưa nộp được. Output nguyên bản nằm trong `submission/evidence/pytest.log`, `verify.log`, `audit.log`; dữ liệu gốc không có Git diff. Bài làm được xuất bản theo yêu cầu tại https://github.com/Neon310304/Day21-Track3-Finetuning-Lab. Chưa nộp VLearn hoặc upload HuggingFace Hub tự động.

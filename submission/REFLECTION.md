# Reflection — Day21

Đây là phản tư kỹ thuật dựa trên evidence của bài làm, được soạn với AI assistant; không phải lời khẳng định về trải nghiệm hay niềm tin trước đây của người học mà AI không biết.

## 1. Điều đáng chú ý nhất

Trên cùng model chưa train, prompt ngây thơ đạt target 0,000 còn prompt tối ưu đạt 0,505, format từ 0,000 lên 1,000. Nếu chỉ so fine-tune với baseline (a), phần lớn “tiến bộ” có thể chỉ là model được cho biết schema và vocabulary. Baseline (b) là đối chứng bắt buộc, không phải một cột để trang trí.

## 2. Nút thắt thực tế

Setup mất công ở tương thích driver CUDA, môi trường Linux cho bitsandbytes và checksum CRLF/LF của Windows. GPU chỉ có 4 GiB nên phải chọn model 0.8B, thay vì giả định có T4 hoặc báo số model 4B. Cần phân biệt lỗi setup với kết quả model: checksum sai vì bytes đổi không có nghĩa là được phép sửa checker hoặc sửa nhãn eval.

## 3. Nhận định được kiểm tra lại

Không thể suy ra mask đúng từ một cờ `assistant_only_loss=True`: tokenizer của model này trả mask 0/31, TRL vá thành 13/31, còn mask đã kiểm của lab là 9/31 ở probe. Training phải thực sự dùng labels đã chứng minh, không chỉ in proof của một hàm rồi train bằng một hàm khác. Tương tự, loss giảm không tự chứng minh chất lượng tác vụ hay khả năng phổ thông được giữ lại.

## 4. AI được dùng vào đâu, sai ở đâu

AI đọc README/rubric, tạo `IMPLEMENTATION_PROMPT.md`, thiết lập môi trường riêng, chạy notebook thật, thêm log/raw predictions và kiểm chéo số liệu, rồi soạn báo cáo. Một lần resolver không pin chặt chọn torch/CUDA quá mới so với driver; quá trình đó được dừng và thay bằng stack CUDA 11.8 có pin, kiểm `pip check` và probe GPU thật. Dự đoán rằng checkout lại là đủ sửa CRLF cũng không đúng; phải lấy bytes chuẩn từ blob gốc. AI không tạo điểm training giả, không đọc holdout để tuning và không tự nới gate khi kết quả không thuận lợi.

## 5. Nếu làm cho khách hàng thật

Bước đầu là chốt hợp đồng output và chi phí của lỗi, rồi thu thập một eval sạch đại diện cho dữ liệu triển khai. Đo baseline prompt mạnh trước, đóng băng dữ liệu/prompt và bổ sung regression cho chức năng cần bảo toàn. Chỉ fine-tune khi đã thấy khoảng trống cụ thể; sau đó dùng quality, latency và regression để quyết định ship, không chọn adapter chỉ vì training loss nhỏ nhất. Các đề xuất thử thêm trong report là thí nghiệm tương lai, không phải kết quả đã làm.

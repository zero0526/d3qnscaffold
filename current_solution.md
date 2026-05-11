# Kiến Trúc và Giải Pháp Hiện Tại (Hierarchical RL 6G)

Tài liệu này mô tả ngắn gọn cách hệ thống Reinforcement Learning (D3QN + Mean Field) điều phối tài nguyên 6G đang được tổ chức, đặc biệt nhấn mạnh vào **Bộ 3 Cải Tiến Cốt Lõi** vừa được thiết lập nhằm vượt qua rào cản nổ Q-value, Overestimation và thói quen lười biếng (Lazy Agent).

## 1. Luồng Hoạt Động Cốt Lõi (Hierarchical Architecture)

Hệ thống hoạt động trên 2 tầng (Time-scale separation):
- **Tầng Upper (Global):** Quá trình Markov dài hạn (chạy theo Frame = 10 Slots). Đưa ra quyết định phân bổ CPU (Compute) và Tải dịch vụ AI (Placement) ở các Edge Nodes.
- **Tầng Lower (Terminals):** Quá trình Markov ngắn hạn (chạy liên tục mỗi Slot). Tại mỗi thiết bị đầu cuối chạy 1 Agent độc lập chọn Node và Mô hình thu gọn phù hợp. Phối hợp tương tác thông qua kỹ thuật **Mean-Field** kết hợp **D3QN** (Dueling Double DQN).

## 2. Các Nút Thắt Đã Được Giải Quyết Triệt Để & Giải Pháp Hiện Tại

### A. Lỗi Ảo Giác và Phình To Q-Value (Target Overestimation Bias)
- **Vấn đề:** Khi mạng NN cập nhật điểm, đôi khi thuật toán chui nhầm vào những Action cấm (hành động không được hỗ trợ bởi Node) do điểm Q ngẫu nhiên tàn dư chưa được cập nhật. Kéo theo toàn bộ giá trị Max ở Bellman Equation bị méo mó.
- **Giải Pháp Vàng (Target Action Masking):** 
    - Hạ cứng (`-1e10`) và triệt tiêu trước mọi Q-value rác. Trong quá trình Back-propagate (hàm `learn()` của `D3QNAgent`), thông số `next_mask` được lấy thẳng từ ReplayBuffer. Lệnh `masked_next_q.argmax()` đảm bảo Agent học một cách trong sáng, vĩnh viễn không bị lừa chọn Action ảo.

### B. Agent "Ngồi Chơi Xơi Nước" (Energy-Dominated Lazy Agents)
- **Vấn đề:** Hàm phạt điện năng trước đây mạnh gấp hằng chục ngàn lần việc dọn dẹp hàng chờ (Backlog). Kết quả là Agent "lười biếng": thà cố tình chặn tác vụ (gây Failed) để đỡ hao năng lượng CPU/Transmission, còn hơn gồng mình xử lý Task.
- **Giải Pháp (Reward Linear Re-scaling `O(1)`):**
    - Lôi toàn bộ **Energy**, **Queue Delay**, và **QoS Violations** về chung một hệ tọa độ `O(1)` bằng phép chia `/ 100.0`. 
    - Ở hệ mới, điểm phạt của Điện năng và của Hàng Chờ (Queue) có sức nặng tương đương nhau (`~1.0 - 2.0`). Từ đây Terminals có lợi ích (Reward) thực sự khi cố gắng khơi thông tắc nghẽn, đẩy Completion Rate thoát khỏi cảnh bế tắc ở 40-50%.

### C. Quy Trách Nhiệm Cá Nhân (Local Credit Assignment)
- **Vấn đề:** 20 Agents xâu xé 1 điểm Reward Tổng. Agent A ném một cục kẹo, Agent B vứt một bịch rác, nhưng cả 2 bị la chung 1 mức án.
- **Giải Pháp:** 
    - Mix 50-50: Tách riêng phần tiêu thụ Load của Node và Giao tiếp thành mảng Phạt Cục Bộ (`reward_nodes`). 
    - Reward cuối = `50% Global + 50% Node` giúp thiết bị rút tỉa kinh nghiệm rất nhanh từ chính vị trí thắt nút (Bottleneck) mà nó vừa trực tiếp gây ra bão tải.

### 3. Tín Hiệu Nhận Diện Hội Tụ Sớm
- **Avg Lower TD Loss**: Ổn định đều đặn (không bị nẩy lên hàng triệu).
- **Completion Rate**: Sẽ nhích vượt dần từ `60%` lên đến `~90%`.
- **Epsilon và Action**: Agent sẽ bớt rải đều workload và tập trung đập mạnh vào 1 số Node có Placement tốt trong lúc Epsilon tuyến tính giảm dần qua các chặng Annealing.
- **Quản lý Gradient**: Neural Network được kìm đai bằng `clip_grad_norm_ = 10.0` để đi bước lớn, đẩy nhanh tốc độ hội tụ.

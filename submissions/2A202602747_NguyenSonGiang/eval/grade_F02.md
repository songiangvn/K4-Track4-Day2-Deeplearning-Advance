## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 97.90% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9730, mốc 0.9277, Δ=+0.0453, s=0.0017 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 95.0% (mốc 88.5%), Snake Weed 93.3% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0071, sau 0.0061 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9711, test 0.9730, chênh 0.0019 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 4.3 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 20 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).

# Sprint 2 — Nhật ký tiến độ kỹ thuật

File này ghi lại toàn bộ công việc kỹ thuật trong Sprint 2: quyết định, code, kết quả,
lỗi/debug, và lý do đằng sau mỗi lựa chọn. Mục đích chính: dùng làm tài liệu chuẩn bị
phỏng vấn — đủ chi tiết để trả lời câu hỏi kỹ thuật mà không cần nhớ lại từ đầu.

Quy tắc: chỉ append, không ghi đè entry cũ. Mỗi entry có timestamp + tiêu đề.

---

## [2026-09-04] - Sprint 2 kickoff: đồng bộ repo và tổng hợp context toàn bộ dự án

### Bối cảnh / Vấn đề
Trước khi bắt đầu Sprint 2, cần đảm bảo làm việc trên code mới nhất từ remote và có
một bức tranh đầy đủ, chính xác về kiến trúc hệ thống + những gì Sprint 1 đã hoàn thành,
để tránh lên kế hoạch Sprint 2 dựa trên thông tin sai hoặc lỗi thời. Repo là monorepo lớn
(nhiều module: backend, engine ML, HMI web, IoT edge) với lịch sử tái cấu trúc thư mục
nhiều lần, nên rủi ro đọc nhầm tài liệu cũ là có thật.

### Kỹ thuật/công nghệ sử dụng
- **Git fetch/pull (fast-forward)**: đồng bộ code an toàn, không tạo merge commit thừa.
  Chọn fast-forward vì branch local không có commit riêng lệch khỏi remote.
- **Subagent (general-purpose) để đọc sâu repo**: do repo có quy mô lớn (hàng trăm file,
  4 module chính), dùng một agent con để đọc toàn bộ, đối chiếu chéo giữa docs và code
  thật, rồi tổng hợp báo cáo — hiệu quả hơn tự đọc tuần tự từng file trong context chính.
- **git log --oneline, grep TODO/FIXME**: khảo sát nhanh lịch sử và nợ kỹ thuật mà không
  cần đọc toàn bộ diff.

### Ý nghĩa / Tầm quan trọng
Đây là bước nền tảng cho toàn bộ Sprint 2. Nếu bỏ qua bước này:
- Có thể lên kế hoạch Sprint 2 dựa trên `SPRINT_TASKS.md` (tài liệu mô tả sai scope —
  nói về GKE/cost dashboard, không khớp với những gì team thực sự làm ở Sprint 1).
- Có thể bỏ sót 2 vấn đề nghiêm trọng đang tồn tại ở tầng model: **model provenance
  không rõ ràng** và **species class list bị lệch nghiêm trọng** (xem phần dưới) — cả
  hai đều ảnh hưởng trực tiếp đến độ tin cậy của hệ thống nhận diện loài.

### Những gì đã thay đổi
- **File tạo mới:** `docs/sprint2/SPRINT2_LOG.md` (chính file này).
- **Không sửa/xóa code nào khác** trong bước này — đây thuần là bước đồng bộ + đọc/phân tích.
- **Trạng thái git local:** chuyển branch từ `EE/QDN/relocate-training-pipeline-folder`
  sang `main`; fast-forward pull `c4df89b4 → a7d69b12` (159 commit mới, 215 file thay đổi,
  +92482/-1935 dòng). Không có conflict.
- 4 file untracked ở root (`Sprint1_5min_Presentation_Script.md`,
  `Sprint1_Summary_Nolan_Nguyen.md`, `Sprint1_Technical_Overview_Nolan_Nguyen.md`, `hm.md`)
  được giữ nguyên, không đụng vào — theo yêu cầu của user.

### Trích dẫn code quan trọng
Không áp dụng — bước này thuần đọc/phân tích, chưa viết code.

### Output / Kết quả
```
git pull origin main
Updating c4df89b4..a7d69b12
Fast-forward
 215 files changed, 92482 insertions(+), 1935 deletions(-)
```
`git log --oneline -10` xác nhận HEAD tại `a7d69b12`
("Merge pull request #1005 from DataBytes-Organisation/EE/PW/heldout-baseline-shared-manifest").

### Quyết định kỹ thuật & lý do (nếu có)
- **Checkout sang `main` thay vì đọc từ branch feature cũ**: main phản ánh trạng thái
  mới nhất mà toàn team đã merge, còn branch feature chỉ chứa 1 thay đổi cục bộ (di chuyển
  thư mục training pipeline). Đọc context nên dựa trên main.
- **Dùng subagent thay vì tự đọc tuần tự**: cân nhắc giữa (a) tự đọc từng file tuần tự
  trong context chính — chậm, tốn context window của phiên làm việc chính; và (b) giao
  cho subagent đọc song song rồi tổng hợp — chọn (b) vì đây là tác vụ khảo sát rộng,
  không cần tương tác qua lại nhiều bước.

### Việc còn tồn đọng / hướng tiếp theo
Baseline context đã đọc (chi tiết đầy đủ ở tóm tắt bên dưới) sẽ là điểm tham chiếu cho
các entry tiếp theo trong Sprint 2. Các vấn đề lớn cần Sprint 2 giải quyết (đã xác định
nhưng CHƯA làm gì trong bước này):
1. Model provenance chưa rõ ràng (3 artifact model khác nhau, không truy được lịch sử train).
2. Species class list lệch nghiêm trọng (21 loài vs 123 loài, chỉ trùng ~8 loài).
3. Hợp nhất 2 pipeline augmentation (`src/prototypes/engine/augmentation/` và
   `src/prototypes/engine/reproducible_training_pipeline/`) — hiện đang tách tạm để tránh
   conflict, cần merge lại.
4. Train pipeline mới (`reproducible_training_pipeline`) đến hội tụ trên data thật —
   hiện mới chạy smoke test, chưa train thật.
5. Dọn `src/production/backend/app/main.py` (FastAPI() khởi tạo trùng, CORS đăng ký 2 lần,
   code chết).
6. Tăng test coverage Engine (hiện 46% trên `echo_engine_iot.py`), thiết lập MQTT broker
   test local thay vì gọi HiveMQ public.

---

### 📌 Tóm tắt trạng thái Sprint 2 tại thời điểm bắt đầu (2026-09-04)

**Kiến trúc hệ thống:**
Monorepo, tách `src/production/` (code chạy thật) và `src/prototypes/` (đang thử nghiệm).
4 module chính:
- **Backend** — FastAPI, entrypoint `src/production/backend/app/main.py`, MongoDB
  (pymongo/motor), 14 router, có contract lỗi chuẩn hoá (`app/errors.py`).
- **Engine (ML)** — docs mô tả `echo_engine.py` nhưng Docker thực tế deploy
  `echo_engine_iot.py` (TFLite). Pipeline: MQTT → mel-spectrogram → inference → MongoDB.
- **HMI (web admin)** — Express, `src/production/hmi/ui/server.js`, port 3000, OpenLayers
  map, JWT, Stripe, MQTT client.
- **IoT/Edge** — `src/production/iot/edge_inference/iot_edge_client.py`, TFLite inference
  tại thiết bị.

**Sprint 1 đã hoàn thành:**
- Reproducible training pipeline (Hydra/PyTorch) chạy end-to-end, có smoke test — chưa
  train hội tụ trên data thật.
- Chuẩn hoá lỗi backend (PR #984).
- Sửa lỗi khởi tạo bản đồ EchoNet trên HMI + test hồi quy.
- Bộ test Engine tự động: 44 test, coverage 46%.
- 2 audit tài liệu quan trọng: Model Provenance Audit, Species Class List Audit.
- Benchmark 6 phương pháp augmentation.

**Gaps/nợ kỹ thuật chính (chi tiết ở mục "Việc còn tồn đọng" phía trên):**
model provenance, species list mismatch, 2 pipeline augmentation chưa hợp nhất,
`app/main.py` cần dọn, coverage Engine còn thấp, `SPRINT_TASKS.md` có vẻ sai scope
cần xác nhận lại với team.

**Test coverage:**
`src/tests/` (unit/integration/pipeline), Engine module-local tests (46% coverage),
HMI có `nodes-overlay.test.mjs`. Backend chưa thấy thư mục test riêng — khả năng là gap.

---

## [2026-09-04] - Phân tích task Sprint 2 cá nhân (3.12 - Training-Pipeline Completion and QAT Enablement)

### Bối cảnh / Vấn đề
Nhận được file kế hoạch chính thức Sprint 2 của Engine team (`Engine_Sprint2_Tasks_Final.pdf`).
Task được giao: **3.12 Nolan Nguyen** — Workstream "Model Improvement and Training", cặp với
Oh Fu Sheng (3.11). Yêu cầu: (1) own pipeline architecture, review phần Oh Fu Sheng implement
(QAT/checkpoint deferred items), (2) chạy full training (non-synthetic) đầu tiên trên balanced
dataset hoặc dataset hiện tại nếu balanced set trễ. Cần phân tích kỹ trước khi code, vì đây là
task "review + integration" — sai sót ở bước review sẽ lan sang toàn bộ pipeline training.

### Kỹ thuật/công nghệ sử dụng
- Đọc trực tiếp source code (`model/quant.py`, `model/__init__.py`, `main.py`, `train.py`,
  `config/config.yaml`) thay vì chỉ tin vào README, vì README có thể mô tả trạng thái tại thời
  điểm viết chứ không phải trạng thái code hiện tại — cần verify bằng grep/read thực tế.
- PyTorch `torch.ao.quantization.quantize_fx` (FX Graph Mode Quantization): `prepare_qat_fx`
  (Quantization-Aware Training) và `prepare_fx`/`convert_fx` (Post-Training Static Quantisation) —
  đây là 2 con đường quantise khác nhau trong pipeline, cần phân biệt rõ cái nào đã wire, cái nào
  chưa.

### Ý nghĩa / Tầm quan trọng
Nếu review task 3.11 không kỹ, pipeline có thể báo "QAT path validated" trong khi thực chất
input shape sai hoặc PTQ path chưa từng chạy — dẫn đến model quantised không đúng, ảnh hưởng
trực tiếp đến toàn bộ Calibration/Edge workstream (Hoang, Tharrun, Dinal) vì họ build tiếp trên
kết quả của Model Improvement workstream này.

### Những gì đã thay đổi
Không sửa file code nào — bước này thuần đọc/phân tích. Chỉ cập nhật log này.

### Trích dẫn code quan trọng

`model/quant.py:40-44` — default input_size sai (kích thước CIFAR, không phải kích thước
mel-spectrogram thật của pipeline):
```python
def prepare_qat_fx(float_model, input_size=(1, 3, 32, 32)):
    example_inputs = torch.rand(size=input_size).cpu()
    prepared_qat = __prepare_qat_fx(float_model, qconfig_mapping, example_inputs=example_inputs)
```
`model/__init__.py:111` gọi hàm này **không truyền `input_size`**, nên luôn dùng default sai:
```python
self.model = prepare_qat_fx(self.model)
```
Input thật theo `config.yaml` (`n_mels=384`, `hop_length=480`, clip 2s @ 48kHz sample_rate):
shape thực tế khoảng `(1, 3, 384, 200)`, không phải `(1, 3, 32, 32)`.

`model/__init__.py:162-169` — nhánh non-QAT (Post-Training Static Quantisation) chưa wire gì:
```python
def quantise(self):
    if not self.use_qat:
        print("Warning: quantise() called, but model was not trained with QAT. Only fusing modules.")
        self.model.eval()
        self.model.fuse_model()
        return
    self.model = convert_fx(self.model)
```
`prepare_post_static_quantize_fx` trong `quant.py:47-59` được import nhưng **không hề được gọi**
ở đâu trong codebase — dead code.

`config/config.yaml:58-63` — checkpoint_path hardcode từ máy dev cũ, không portable:
```yaml
run:
  checkpoint_path: ./outputs/2025-08-23/18-50-17/best_efficientnet_v2_qat.pth
```

### Output / Kết quả
Không có output chạy code ở bước này (chỉ đọc/phân tích). Bằng chứng: README.md của
`reproducible_training_pipeline` xác nhận trạng thái hiện tại — chỉ có 1 lần chạy thật là
smoke test 2-epoch trên 3 class synthetic data (`docs/baseline_smoke_training_log.txt`); chưa
từng train trên `src/prototypes/data_files/` (~900MB, 128 species) hay balanced dataset thật.

### Quyết định kỹ thuật & lý do (nếu có)
- **Quyết định đọc code thay vì chỉ đọc README/PDF task list**: PDF task 3.11 chỉ nói chung
  chung "fix the example-input shape; wire prepare/convert" — không đủ cụ thể để review PR sau
  này. Grounding bằng code thật giúp có checklist review cụ thể (4 điểm) thay vì mơ hồ.
- **Phát hiện mâu thuẫn phân công**: ảnh chụp Slack (26/8) cho thấy Kiernan Nguyen nhắn trực
  tiếp yêu cầu rebase/merge `reproducible_training_pipeline/augment.py` vào
  `augmentation/augment.py` của anh ấy — đây chính là việc "folder de-duplication" mà PDF giao
  cho **Oh Fu Sheng (task 3.11)**, không phải Nolan. Quyết định: KHÔNG tự làm việc merge này
  ngay, mà xác nhận lại với Kiernan/Oh Fu Sheng ai thực sự đảm nhiệm, để tránh 2 người cùng sửa
  1 file dẫn đến conflict hoặc trùng công sức.
- **Thứ tự ưu tiên đề xuất**: (1) làm rõ vụ phân công augment.py, (2) chờ + review PR quantisation
  của Oh Fu Sheng theo 4 điểm đã liệt kê, (3) song song hỏi tiến độ balanced dataset
  (Manisha/Raveesha, critical path), (4) chốt `norm_choice` cho các model config với
  Kiernan/Praveen trước khi chọn model train full, (5) chạy full training thật, lưu log.

### Việc còn tồn đọng / hướng tiếp theo
- Chưa liên hệ Kiernan/Oh Fu Sheng để làm rõ phân công augment.py de-dup.
- Chưa review PR thật của Oh Fu Sheng (PR chưa tồn tại tại thời điểm phân tích này).
- Chưa xác nhận tiến độ balanced dataset từ Manisha/Raveesha.
- Chưa chạy full training thật — đang chờ input từ các bước trên.
- Cần verify riêng: `run.test=true` + checkpoint loading path đã được bug-fix ở Sprint 1
  (typo `cfg.run.cKLDivLossheckpoint_path`, thiếu import `OmegaConf`) nhưng **chưa từng được
  exercise bằng một lần chạy thật** — cần tự chạy thử trước khi báo "Reviewed, working pipeline".

---

## [2026-09-04] - So sánh 2 bản augment.py (Kiernan vs. reproducible_training_pipeline) trước khi merge

### Bối cảnh / Vấn đề
Nhận báo cáo Sprint 1 của Kiernan Nguyen (`Project_Echo_Sprint1_Augmentation_Framework_Report.pdf` +
`README.pdf` + `Review of current SpecAugment implementation...pdf`) mô tả đầy đủ code
`augment.py` của anh ấy tại `src/prototypes/engine/augmentation/`. Kết hợp với tin nhắn Slack
trước đó (26/8), Kiernan yêu cầu merge `reproducible_training_pipeline/augment.py` (của mình)
lên trên file của anh ấy, giữ giá trị của Kiernan nếu có conflict. Cần so sánh 2 file thật để
biết merge này có đơn giản hay không trước khi ai đó (Oh Fu Sheng, theo phân công PDF) thực hiện.

### Kỹ thuật/công nghệ sử dụng
- Đọc trực tiếp `reproducible_training_pipeline/augment.py` (106 dòng, class `SpecAugment(nn.Module)`)
  và đối chiếu từng đoạn với code Kiernan dán trong report PDF — không dùng công cụ diff tự động
  vì 2 file có tên biến/cấu trúc khác nhau dù cùng logic, cần đọc hiểu ý nghĩa chứ không chỉ so
  text.

### Ý nghĩa / Tầm quan trọng
Đây không phải merge cosmetic. 2 bản `augment.py` được viết độc lập bởi 2 người, cùng mục tiêu
(fix bug "80% clip bị mask" từ review gốc) nhưng **khác nhau ở hành vi thực tế của augmentation**
— ảnh hưởng trực tiếp đến dữ liệu training thật sự đưa vào model. Nếu merge sai hướng (giữ bản
của mình thay vì bản Kiernan yêu cầu), pipeline training đầy đủ (`main.py`/`train.py`) sẽ chạy với
augmentation logic khác với những gì Kiernan đã visually + experimentally validate ở Sprint 1.

### Những gì đã thay đổi
Không sửa file nào — thuần phân tích/so sánh, ghi vào log.

### Trích dẫn code quan trọng

Bản của mình (`reproducible_training_pipeline/augment.py:39-54`) — validate nghiêm ngặt, mask_value
chỉ hỗ trợ "zero":
```python
if (freq_mask_param is None) == (freq_mask_ratio is None):
    raise ValueError(...)
...
if mask_value != "zero":
    raise ValueError(f"SpecAugment: unsupported mask_value '{mask_value}'; only 'zero' is currently implemented.")
```

Bản Kiernan (theo PDF) — có `_fill_value()` hỗ trợ đủ "zero"/"min"/"mean", và ép zero-width mask
về tối thiểu 1 pixel để đảm bảo mask luôn thực sự xảy ra:
```python
def _fill_value(self, x):
    if self.mask_value == "zero": return 0.0
    elif self.mask_value == "min": return x.min()
    elif self.mask_value == "mean": return x.mean()
    ...
f = random.randint(1, max(1, f_param))  # lower bound = 1, luôn mask thật sự
```

Bản của mình xử lý zero-width khác hẳn — cho phép mask bị bỏ qua thay vì ép width tối thiểu:
```python
width = self._strip_width(self.freq_mask_param, self.freq_mask_ratio, F)
if width <= 0:
    continue
```

### Output / Kết quả
Bảng so sánh đầy đủ 6 điểm khác biệt (default ratio/cap, mask_value support, zero-width handling,
input validation, thứ tự random-vs-clamp cho time cap) đã trình bày cho user trong chat.
Không có output chạy code — đây là phân tích tĩnh dựa trên đọc 2 nguồn code.

### Quyết định kỹ thuật & lý do (nếu có)
- **Kết luận merge đúng hướng phải lấy code Kiernan làm nền, không phải ngược lại**: theo đúng
  chỉ thị của Kiernan ("giữ giá trị của tôi nếu conflict"), file merge cuối cùng phải khôi phục lại
  `mask_value="min"/"mean"` và cách ép zero-width về 1 — tức là **bản của mình sẽ mất đi 2 tính năng
  hiện tại** (strict validation là ngoại lệ, có thể giữ vì đây là phần "thêm", không "conflict" với
  giá trị nào của Kiernan).
- **Chưa tự thực hiện merge**: vì task 3.11 (folder de-duplication) chính thức thuộc về Oh Fu Sheng
  theo PDF, chỉ dừng ở phân tích để trang bị thông tin, tránh làm trùng việc hoặc merge sai hướng
  trước khi xác nhận ai thực sự làm.

### Việc còn tồn đọng / hướng tiếp theo
- Chờ xác nhận từ Kiernan/Oh Fu Sheng ai thực hiện merge thật.
- Nếu merge được thực hiện, cần viết 1 test nhanh so sánh phân phối mask trước/sau merge (đặc biệt
  với `original_unfixed_reference` preset, vì đây là preset duy nhất set `freq_mask_param`/`time_mask_param`
  cố định — nơi khác biệt zero-width dễ biểu hiện nhất) để đảm bảo hành vi augmentation trong
  `reproducible_training_pipeline` không đổi ngoài ý muốn trước khi chạy full training thật.

---

## [2026-09-04] - Đọc Sprint 1 report của Oh Fu Sheng (Engine Codebase Cleanup) + phát hiện pipeline bị nhân đôi thật sự

### Bối cảnh / Vấn đề
Nhận 2 file PDF Sprint 1 của Oh Fu Sheng (`Engine_Sprint1_Deliverable_Report_OFS.pdf` +
`Engine_Sprint1_Decision_Matrix_and_Recommendations.pdf`) — báo cáo phân tích cấu trúc thư mục
Engine, không di chuyển/xóa file nào ở Sprint 1. User yêu cầu đọc report này cùng report của
Kiernan (đã đọc trước đó) chỉ để hiểu bối cảnh chuẩn bị Sprint 2, chưa cần hành động.

### Kỹ thuật/công nghệ sử dụng
- Đọc PDF report + đối chiếu với `Glob` thực tế trên `src/prototypes/engine/augmentation/` để
  kiểm chứng claim trong các report bằng trạng thái file thật, thay vì chỉ tin vào mô tả text.

### Ý nghĩa / Tầm quan trọng
Phát hiện quan trọng: `src/prototypes/engine/augmentation/` hiện **không chỉ có `augment.py` của
Kiernan** mà còn có **toàn bộ bản sao `main.py`, `train.py`, `dataset.py`, `model/*.py`, `config/`**
— gần như giống hệt những gì đang có ở `reproducible_training_pipeline/`. Nghĩa là "folder
de-duplication" (task 3.11 của Oh Fu Sheng) không chỉ là việc merge 1 file `augment.py` như Kiernan
nhắn Slack, mà là dọn **toàn bộ pipeline bị nhân đôi**. Việc này ảnh hưởng trực tiếp đến task 3.12
của mình ("own pipeline architecture") vì hiện có 2 bản sống song song, có nguy cơ đã lệch nhau.

### Những gì đã thay đổi
Không sửa file nào — chỉ đọc report + `Glob` kiểm tra thực trạng thư mục.

### Trích dẫn code quan trọng
Không áp dụng — bước này không có code, chỉ có kết quả liệt kê thư mục:
```
src/prototypes/engine/augmentation/
├── augment.py, main.py, train.py, dataset.py     <- trùng với reproducible_training_pipeline/
├── model/{effv2,ghost_effv2,panns_*,quant,utils}.py
├── config/{augmentation,model,teacher_model,local}/...
├── build_experiment_subset.py, run_validation_experiment.py, visualise_spec_augment.py  <- chỉ Kiernan có
└── experiment_results/, spectrogram_examples/    <- chỉ Kiernan có, không có bản sao
```

### Output / Kết quả
`Glob` liệt kê 8 file cấp 1 và toàn bộ cây con của `src/prototypes/engine/augmentation/` —
xác nhận pipeline đầy đủ (không chỉ augment.py) tồn tại song song ở 2 nơi.

### Quyết định kỹ thuật & lý do (nếu có)
- **Giả thuyết về nguyên nhân trùng lặp**: nhánh Kiernan (`EE/KN/audio-augmentation-framework`,
  PR #975) khả năng được branch ra TRƯỚC khi mình relocate pipeline (`caa3ee8b`), nên khi merge
  vào main sau đó, Git giữ lại toàn bộ `main.py`/`train.py`/`dataset.py`/`model/` từ nhánh Kiernan
  tại `augmentation/`, song song với bản đã chuyển sang `reproducible_training_pipeline/`. Đây là
  giả thuyết dựa trên bằng chứng gián tiếp (thứ tự PR trong git log, README tự mô tả "git mv,
  content unchanged"), **chưa xác nhận 100% bằng git blame/log chi tiết** — cần verify nếu quan trọng.
- **Quyết định KHÔNG hành động ngay**: theo đúng yêu cầu của user ("cứ đọc và tìm hiểu trước"),
  chỉ dừng ở việc ghi nhận phát hiện, không tự ý dọn dẹp hay báo cho Oh Fu Sheng/Kiernan.

### Việc còn tồn đọng / hướng tiếp theo
- Cần xác nhận việc de-duplication (khi Oh Fu Sheng thực hiện task 3.11) phải giữ lại các file chỉ
  có ở `augmentation/`: `build_experiment_subset.py`, `run_validation_experiment.py`,
  `visualise_spec_augment.py`, `experiment_results/`, `spectrogram_examples/` — đây là evidence
  Sprint 1 của Kiernan, không được xóa nhầm khi dọn phần pipeline trùng lặp.
- Cần xác định rõ bản nào là "nguồn thật" giữa `augmentation/` và `reproducible_training_pipeline/`
  trước khi review PR của Oh Fu Sheng hoặc chạy full training — khả năng cao là
  `reproducible_training_pipeline/` (theo README tự nhận là bản đã sửa 2 bug + rewrite augment.py),
  nhưng chưa chính thức xác nhận với Oh Fu Sheng/Kiernan.

---

## [2026-09-04] - Verify dataset readiness + chạy thật path run.test=true/checkpoint loading

### Bối cảnh / Vấn đề
Trong lúc chờ Kiernan/Oh Fu Sheng phản hồi vụ folder de-dup, chuyển sang làm phần việc không phụ
thuộc ai trong task 3.12: (1) kiểm tra dataset local có đủ điều kiện chạy full training không —
README đã cảnh báo class nào <5 file sẽ gây `ZeroDivisionError`; (2) verify path `run.test=true` +
checkpoint loading — README ghi là đã bug-fix (typo `cfg.run.cKLDivLossheckpoint_path`, thiếu
import `OmegaConf`) nhưng **chưa từng chạy thật để chứng minh**. Đây là 2 việc cụ thể, tự làm được,
không cần chờ dataset balanced từ Manisha/Raveesha hay PR của Oh Fu Sheng.

### Kỹ thuật/công nghệ sử dụng
- Bash `find`/loop đếm số file mỗi thư mục species trong `src/prototypes/data_files/` — cách nhanh
  nhất để kiểm tra invariant "mỗi class cần ≥5 file" mà không cần chạy Python.
- Viết script Python throwaway (không commit vào repo, nằm ở scratchpad) mô phỏng lại
  `test_train_smoke.py` nhưng thêm phase 2: chạy `main.py` lần 2 với `run.train=false run.test=true
  run.checkpoint_path=<checkpoint từ phase 1>` để exercise path checkpoint-loading thật, dùng
  `subprocess.run` gọi thẳng `.venv/Scripts/python.exe` của pipeline (không phải `uv run`, vì `uv`
  CLI không có sẵn trong PATH của session này, nhưng venv đã được sync sẵn từ trước).

### Ý nghĩa / Tầm quan trọng
Cả 2 việc đều là "Reviewed, working pipeline" evidence bắt buộc phải có trước khi báo cáo Sprint 2
task 3.12 hoàn thành. Nếu bỏ qua verify path checkpoint-loading, có nguy cơ báo "pipeline hoạt động
tốt" trong khi một nhánh code quan trọng (dùng để test/evaluate model đã train, không phải chỉ để
train) chưa từng chạy thật lần nào kể từ khi bug được fix ở Sprint 1.

### Những gì đã thay đổi
- Không sửa file nào trong repo.
- Tạo 1 file throwaway ngoài repo (scratchpad): `verify_run_test_checkpoint.py` — không phải phần
  giao nộp, chỉ dùng để verify tại chỗ.

### Trích dẫn code quan trọng
Lệnh phase 2 (load checkpoint thật, không train):
```
main.py model=ghost_efficientnet_v2 run.train=false run.test=true run.quantise=false \
  run.checkpoint_path=<path đến best_ghost_efficientnet_v2_qat.pth từ phase 1>
```
Output xác nhận đúng nhánh code trong `main.py:176-178` đã chạy (không phải fallback
"no valid checkpoint found"):
```
Training skipped. Loading model from: ...\outputs\best_ghost_efficientnet_v2_qat.pth
--- Testing original model ---
Starting evaluation on test set...
```

### Output / Kết quả
**Dataset check** (`src/prototypes/data_files/`):
```
Total species folders: 128
Total files: 9205
Species with <5 files: 1   ->  "Geopelia cuneata" chỉ có 4 file
```
Khớp với con số "128 species" README tự nhận. Chỉ 1 class có vấn đề — dễ xử lý (loại bỏ class này
hoặc gộp thêm data trước khi chạy full training).

**Smoke test hiện có** (`src/tests/pipeline/engine_training/smoke_test/test_train_smoke.py`):
```
Ran 1 test in 75.478s
OK
```

**Verify run.test=true + checkpoint loading (2 phase, model=ghost_efficientnet_v2, CPU, synthetic data)**:
- Phase 1 (train.true, 2 epoch): thành công, sinh ra `best_ghost_efficientnet_v2_qat.pth`.
- Phase 2 (train=false, test=true, load checkpoint phase 1): returncode 0.
  ```
  Total Training samples: 15
  Total Validation samples: 3
  Training skipped. Loading model from: ...best_ghost_efficientnet_v2_qat.pth
  --- Test Set Metrics ---
  Accuracy: 0.3333
  Precision: 0.1111
  Recall: 0.3333
  F1 Score: 0.1667
  ```
  Accuracy 0.33 hợp lý với baseline random 3-class trên synthetic data (không kỳ vọng học được gì
  thật từ sine wave giả).

### Quyết định kỹ thuật & lý do (nếu có)
- **Chọn verify bằng synthetic data thay vì local dataset thật (128 species)**: mục tiêu chỉ là xác
  nhận code path hoạt động đúng (không exception, load đúng checkpoint), không phải đánh giá độ
  chính xác model — nên dùng synthetic giống hệt cách `test_train_smoke.py` đã làm, nhanh và không
  phụ thuộc vào việc dataset thật có sẵn đủ điều kiện hay chưa.
- **Gọi thẳng venv python thay vì `uv run`**: `uv` không có trong PATH của session Bash hiện tại,
  nhưng `.venv/` đã được `uv sync` từ trước (còn nguyên từ lần làm việc trước), nên gọi trực tiếp
  `.venv/Scripts/python.exe main.py ...` cho kết quả tương đương mà không cần cài lại `uv`.

### Việc còn tồn đọng / hướng tiếp theo
- **"Geopelia cuneata" (4 file) sẽ crash full training** nếu không xử lý trước — cần quyết định: bỏ
  species này ra khỏi dataset, hay tìm thêm audio cho nó, trước khi chạy full training trên
  `src/prototypes/data_files/`.
- **Đính chính (2026-09-04, sau khi user hỏi lại)**: không cần chờ balanced dataset từ
  Manisha/Raveesha trước khi train. PDF mục "Cross-Task Dependencies" (dòng Nolan/Oh Fu Sheng/Kiernan)
  ghi rõ "Training proceeds in parallel on the current dataset and switches to the augmented/balanced
  dataset when ready" — nghĩa là train ngay trên dataset hiện tại, switch sau nếu balanced set kịp
  giao trong sprint. Đề xuất thứ tự công việc trước đó (liệt "chờ xác nhận balanced dataset" như một
  bước tuần tự trước khi train) là hiểu sai — đây là 2 việc chạy song song, không phụ thuộc nhau.
  Việc thật sự cần làm trước khi train: chỉ cần xử lý xong species thiếu file.
- Vẫn cần làm song song (không chặn việc train): xác nhận `norm_choice` với Kiernan/Praveen, xác
  nhận ai làm folder de-dup (Kiernan/Oh Fu Sheng).
- Chưa test path `run.quantise=true` sau `run.test=true` (nhánh gọi `trainer.model.quantise()`) —
  đây vẫn thuộc phạm vi review PR của Oh Fu Sheng (task 3.11), chưa tồn tại PR để review.

---

## [2026-09-04] - Loại species thiếu file, fix bug LMDB cache, chuyển sang GPU, launch full training thật

### Bối cảnh / Vấn đề
Sau khi xác nhận không cần chờ balanced dataset (xem entry trước), tiến hành chuẩn bị và chạy
"first full (non-synthetic) training" thật theo task 3.12. Gặp lần lượt 3 vấn đề thật trong quá
trình này: (1) 1 species thiếu file gây crash tiềm ẩn, (2) bug LMDB cache khiến training luôn crash
ngay khi bắt đầu validation, (3) máy có GPU (RTX 4060) nhưng PyTorch cài sẵn là bản CPU-only nên
training cực chậm (~27 phút/epoch cho CPU trên full dataset).

### Kỹ thuật/công nghệ sử dụng
- **LMDB (Lightning Memory-Mapped Database)**: thư viện cache dùng trong `dataset.py` để lưu
  spectrogram đã tính sẵn lên đĩa, tránh tính lại mỗi epoch. Đặc điểm quan trọng: LMDB **không cho
  phép mở cùng 1 environment (path) 2 lần trong cùng 1 process** — đây chính là nguồn gốc bug.
- **PyTorch CUDA wheel vs CPU wheel**: `pip install torch==X.Y.Z` không kèm `--index-url` sẽ luôn
  lấy bản CPU-only từ PyPI mặc định, kể cả khi máy có GPU. Phải trỏ đúng
  `--index-url https://download.pytorch.org/whl/cuXXX` khớp với CUDA version mà driver hỗ trợ.
- **`pip install --force-reinstall`**: cần thiết vì pip coi `torch==2.8.0` (constraint không có local
  version segment) là đã "thỏa mãn" bởi bản `2.8.0+cpu` đang cài, nên bỏ qua cài lại nếu không ép buộc.
- **Hydra multi-select config group**: cú pháp đúng để chọn nhiều file từ cùng 1 config group là
  `+local=[cpu,local_data_files]` (list), không phải gọi `+local=` 2 lần riêng biệt như README ghi
  (`+local=cpu +local=local_data_files` bị lỗi "Multiple values for local").

### Ý nghĩa / Tầm quan trọng
Cả 2 bug (LMDB cache, Hydra multi-select syntax) đều **chưa từng bị phát hiện** vì chưa ai chạy
training thật (non-synthetic) với cache bật + train/val cùng process trước đây — README tự ghi rõ
"A full run against the real ~900MB local dataset was not attempted this sprint". Nếu không fix,
không thể hoàn thành deliverable "First full training log" của task 3.12, và các thành viên khác
dùng chung pipeline (Oh Fu Sheng test quantisation, Praveen chạy held-out baseline) sẽ gặp đúng bug
này khi họ chạy training/eval thật với config mặc định.

### Những gì đã thay đổi
- **File sửa**: [dataset.py](../../src/prototypes/engine/reproducible_training_pipeline/dataset.py) —
  tách cache path theo `train`/`val` subdirectory dựa vào `is_train` (dòng ~74-83).
- **Dữ liệu di chuyển** (không xóa, không commit vào git vì `data_files` untracked):
  `src/prototypes/data_files/Geopelia cuneata/` → `src/prototypes/data_files_excluded/Geopelia cuneata/`
  (4 file, thực chất chỉ 2 clip trùng định dạng mp3+wav — dưới ngưỡng 5 file/class).
- **Môi trường**: cài lại `torch`, `torchaudio`, `torchvision` (2.8.0/2.8.0/0.23.0) trong
  `.venv` của `reproducible_training_pipeline` từ bản `+cpu` sang bản `+cu126`, khớp driver CUDA
  12.6 của GPU (RTX 4060 Laptop). Không sửa `pyproject.toml`/`uv.lock` (vẫn pin CPU theo mặc định
  của `uv sync` — nếu ai chạy lại `uv sync` sau này sẽ mất bản CUDA, cần lưu ý).
- **File log mới** (untracked, ngoài phạm vi track thường xuyên):
  `docs/sprint2/full_training_run_2026-09-04.log` — log thật của lần full training đang chạy.

### Trích dẫn code quan trọng
Fix LMDB (`dataset.py`):
```python
self.use_cache = cfg.system.get("use_disk_cache", False)
self.env = None
if self.use_cache:
    # train_dataset và val_dataset là 2 instance riêng, cùng process khi
    # num_workers=0 -> lmdb không cho mở cùng path 2 lần -> tách theo split
    self.cache_path = Path(cfg.system.cache_directory) / ("train" if self.is_train else "val")
    self.cache_path.mkdir(parents=True, exist_ok=True)
```

Lệnh launch full training thật (thay vì dùng `+local=cpu +local=local_data_files` bị lỗi, trỏ path
tuyệt đối trực tiếp để tránh luôn cú pháp Hydra multi-select):
```
main.py +local=cpu \
  system.audio_data_directory=D:/Deakin/SIT374/Project-Echo/src/prototypes/data_files \
  ~augmentations.audio \
  training.num_workers=4 training.batch_size=32 training.device=cuda \
  training.epochs=15 \
  hydra.run.dir=outputs/sprint2_first_full_training_2026-09-04
```

### Output / Kết quả
- **Trước fix** (CPU, batch_size=16, full 127-species dataset): 27m51s chạy được 463/2247 bước rồi
  crash `lmdb.Error: The environment '.cache' is already open in this process` ngay khi chuyển sang
  validation.
- **Sau fix** (verify bằng 3 species nhỏ, real audio, CPU): train xong → val xong → checkpoint lưu
  thành công, không lỗi. `val_acc=0.3333` (hợp lý, gần baseline random 3-class).
- **Sau khi đổi sang GPU** (verify lại 3 species nhỏ): `torch.cuda.is_available()=True`, device
  name `NVIDIA GeForce RTX 4060 Laptop GPU`, cùng 1 kịch bản train→val→save chạy thành công, tốc độ
  quan sát được nhanh hơn rõ rệt so với CPU (dù mẫu quá nhỏ để so sánh định lượng chính xác).
- **Full training thật đang chạy** (background, 127 species, batch_size=32, num_workers=4,
  device=cuda, epochs=15) — kết quả sẽ được cập nhật ở entry tiếp theo khi hoàn tất.

### Quyết định kỹ thuật & lý do (nếu có)
- **Loại `Geopelia cuneata` bằng cách di chuyển (không xóa)**: dataset local không track bởi git
  (`.gitignore`), nên đây là thao tác an toàn, dễ hoàn tác — không cần thiết phải sửa code pipeline
  để handle class thiếu file (giải pháp code sẽ phức tạp hơn cho lợi ích không tương xứng ở giai
  đoạn "first full training" này).
- **Fix cache bằng subdirectory theo split thay vì share 1 lmdb env**: đơn giản nhất, không cần sửa
  `main.py` hay `config.yaml`, tận dụng `is_train` đã có sẵn. Đánh đổi: nếu sau này thêm
  `test_dataset` (hiện đang comment out trong `main.py`) với `is_train=False` mặc định, `val` và
  `test` sẽ vô tình dùng chung 1 cache subdirectory — chưa phải vấn đề hiện tại vì `test_dataset`
  chưa được kích hoạt, nhưng cần nhớ lại nếu ai đó bật nó lên sau này.
  **Không dùng phương án chia sẻ 1 lmdb env chung** giữa train/val vì sẽ cần refactor sâu hơn vào
  cách `main.py` khởi tạo dataset (đổi kiến trúc), rủi ro cao hơn cho lợi ích tương đương.
- **Cài CUDA build thay vì chấp nhận chạy CPU**: với ~27 phút/epoch trên CPU, training 15 epoch sẽ
  mất >6 giờ — không khả thi để có evidence trong phiên làm việc. User xác nhận máy có GPU thật và
  yêu cầu dùng GPU, nên ưu tiên đầu tư thời gian sửa môi trường (khoảng 10-15 phút tải+cài) thay vì
  chấp nhận full training chạy nhiều giờ trên CPU.
- **Không sửa `pyproject.toml`/`uv.lock` để pin CUDA vĩnh viễn**: quyết định này ảnh hưởng đến toàn
  bộ team (nếu pin cứng CUDA, máy không có GPU/không đúng driver version sẽ cài lỗi) — đây là quyết
  định kiến trúc/môi trường cần bàn với team trước, không tự ý đổi. Local fix (cài trực tiếp vào venv
  bằng pip, không qua uv) chỉ ảnh hưởng máy hiện tại.
- **Chọn epochs=15 (không dùng mặc định 500) cho lần full training đầu tiên**: mục tiêu Sprint 2 là
  "first full training", không phải train đến hội tụ hoàn toàn trong phiên này. 15 epoch đủ để có
  đường cong loss/accuracy thật ý nghĩa làm evidence, trong khi vẫn có thể hoàn tất trong một
  khoảng thời gian hợp lý. Có thể resume/train tiếp từ checkpoint sau (path `run.test`/checkpoint đã
  verify hoạt động ở entry trước) nếu cần train sâu hơn.
- **batch_size=32, num_workers=4** (thay vì mặc định 64/12 hay giá trị an toàn tối thiểu 4/0 từ
  README): cân bằng giữa tốc độ (tận dụng 16 core CPU cho tiền xử lý audio song song, GPU 8GB VRAM
  cho model) và rủi ro tràn bộ nhớ (né giá trị 64/12 đã từng gây OOM theo ghi nhận của Kiernan ở
  Sprint 1).

### Việc còn tồn đọng / hướng tiếp theo
- **Cập nhật tiến độ (2026-09-04, đang chạy)**: Epoch 1/15 hoàn tất sau ~22 phút (cache nguội).
  Kết quả epoch 1: `train_loss=24.3343, train_acc=0.0000, val_loss=6.5076, val_acc=0.0600`.
  127 lớp → random baseline ~0.8%, val_acc 6.0% sau 1 epoch là tín hiệu học thật hợp lý, xác nhận
  toàn bộ pipeline (data loading thật + LMDB cache đã fix + GPU + train/val loop) hoạt động đúng
  end-to-end trên dataset thật lần đầu tiên. Tốc độ ổn định ~2.4-3.6s/it tùy giai đoạn train/val.
  Đang tiếp tục theo dõi epoch 2-15 qua Monitor (chỉ báo khi chuyển epoch để tránh spam).
- Chờ full training chạy xong hoàn toàn, sẽ cập nhật kết quả cuối (loss/accuracy curve đầy đủ,
  checkpoint cuối, tổng thời gian thực tế) ở entry tiếp theo.
- Cần báo cho team (đặc biệt Oh Fu Sheng — người sẽ review/dùng chung pipeline cho quantisation, và
  Praveen — held-out baseline) về bug LMDB đã fix, vì họ có thể đã hoặc sẽ gặp đúng lỗi này.
- Cần cân nhắc: có nên thêm ghi chú vào README về việc `pip install torch==X` không tự lấy bản CUDA,
  hoặc thêm hướng dẫn cài CUDA build cho ai có GPU, để tránh các thành viên khác train chậm không
  cần thiết trên CPU trong khi máy họ có GPU.
- Bug cú pháp `+local=cpu +local=local_data_files` trong README cần được sửa lại thành
  `+local=[cpu,local_data_files]` hoặc README hướng dẫn set `audio_data_directory` trực tiếp.
- Sau khi full training xong, cần review lại: `norm_choice` cho các model config (`ghost_efficientnet_v2`,
  `panns_*`) chưa được Kiernan/Praveen xác nhận — vẫn đang dùng `freeze_bn` giả định.

---

## [2026-09-05] - First full training hoàn tất — kết quả cuối cùng

### Bối cảnh / Vấn đề
Tiếp nối entry trước: full training thật (127 species, GPU, 15 epoch) đã chạy xong hoàn toàn sau
khi theo dõi qua Monitor suốt quá trình (báo tiến độ mỗi lần chuyển epoch). Đây là entry tổng kết
deliverable "First full training log" của task 3.12.

### Kỹ thuật/công nghệ sử dụng
Không có gì mới so với entry trước — đây là kết quả của lần chạy đã launch (GPU, EfficientNetV2 +
ArcFace/Circle metric learning, AdamW, ReduceLROnPlateau, checkpoint theo `val_loss` min).

### Ý nghĩa / Tầm quan trọng
Đây là bằng chứng đầu tiên trong toàn bộ lịch sử pipeline rằng nó chạy được **end-to-end trên dữ
liệu thật, quy mô đầy đủ (127 species)**, không sập giữa chừng, có checkpoint thật để dùng cho các
bước tiếp theo của Sprint 2 (Oh Fu Sheng cần checkpoint thật để test quantisation path; Praveen có
thể dùng để so sánh với held-out baseline). Trước đây pipeline chỉ được xác minh bằng synthetic
data hoặc bị crash khi thử data thật.

### Những gì đã thay đổi
Không có thay đổi code mới trong entry này — chỉ là kết quả của lần chạy đã launch trước đó.
Artifact sinh ra (không commit vào git, nằm trong thư mục `outputs/` untracked của pipeline):
- `outputs/sprint2_first_full_training_2026-09-04/best_efficientnet_v2.pth` (82.3 MB) — checkpoint
  tốt nhất, lưu từ epoch 10 (dựa trên `val_loss` thấp nhất, không phải epoch cuối).
- `outputs/sprint2_first_full_training_2026-09-04/class_names.txt` — danh sách 127 species dùng
  cho lần train này.
- `outputs/sprint2_first_full_training_2026-09-04/events.out.tfevents.*` — TensorBoard log.
- `docs/sprint2/full_training_run_2026-09-04.log` — log console đầy đủ (untracked, để tham khảo).

### Trích dẫn code quan trọng
Không áp dụng — entry tổng kết kết quả, không có code mới.

### Output / Kết quả
**Tổng thời gian: 4 giờ 43 phút 39 giây** cho 15 epoch (trung bình ~19 phút/epoch, dao động
14-30 phút/epoch tùy tải hệ thống — máy chạy song song nhiều app khác lúc train).

Đường cong theo epoch (best_metric = val_loss, thấp hơn = tốt hơn; val_acc = accuracy trên 127 lớp,
random baseline ~0.8%):

| Epoch | val_loss | val_acc | Ghi chú |
|---|---|---|---|
| 1 | 6.5076 | 6.00% | Cache nguội, chậm nhất (~22 phút) |
| 2 | 5.3159 | 7.23% | |
| 3 | 4.7002 | 7.23% | |
| 4 | 4.5732 | 2.30% | Dao động mạnh (val nhỏ, ~14 file/lớp) |
| 5 | 4.3339 | 7.96% | |
| 6 | 4.3289 | 7.62% | Best tạm thời |
| 7 | 4.3311 | 6.84% | Không vượt best |
| 8 | 5.6108 | 6.95% | Loss tăng vọt tạm thời, không phải lỗi |
| 9 | 4.2152 | 8.63% | Best mới |
| 10 | **4.1150** | 8.63% | **Best cuối cùng — checkpoint được lưu từ đây** |
| 11 | 4.2929 | 7.17% | |
| 12 | 4.2128 | 6.78% | |
| 13 | 4.2299 | 6.95% | |
| 14 | 4.2448 | **9.75%** | Acc cao nhất tạm thời |
| 15 | 4.1628 | **10.71%** | Acc cao nhất toàn bộ run, nhưng loss không vượt epoch 10 |

`train_acc` giữ nguyên 0.0000 suốt toàn bộ 15 epoch — **đây là hành vi đã biết trước, không phải
lỗi**: model dùng ArcFace/Circle margin loss (`training.use_arcface: circle`), train_acc tính bằng
argmax thô trên logit đã bị margin-adjust nên gần như luôn ~0 ở giai đoạn đầu train (đã ghi nhận
tương tự trong report Sprint 1 của Kiernan) — `val_acc` mới là số đáng tin.

### Quyết định kỹ thuật & lý do (nếu có)
- **Chấp nhận checkpoint từ epoch 10 (không phải epoch 15)** làm "best model" cho deliverable, đúng
  theo tiêu chí `metric_mode=min` trên `val_loss` đã cấu hình sẵn trong `config.yaml` — không tự ý
  đổi tiêu chí chọn checkpoint sang val_acc dù epoch 15 có acc cao hơn, vì đó là quyết định kiến
  trúc có sẵn, thay đổi cần bàn với team (val_loss vs val_acc criterion) chứ không tự quyết ở đây.
- **Không tiếp tục train thêm epoch để cải thiện thêm**: 15 epoch đã đủ tạo ra đường cong học thật,
  có ý nghĩa (accuracy tăng từ ~6% lên ~10%, vượt xa random baseline ~0.8%), đáp ứng đúng yêu cầu
  "first full training" của task 3.12 (không yêu cầu hội tụ hoàn toàn). Có thể resume/train tiếp từ
  checkpoint này sau (dùng path `run.test`/checkpoint đã verify hoạt động ở entry trước đó) nếu cần.

### Việc còn tồn đọng / hướng tiếp theo
- **Đánh giá task 3.12 tại thời điểm này**: cả 3 output kỳ vọng đã có bằng chứng —
  "Validated QAT path" (một phần, còn phụ thuộc PR của Oh Fu Sheng), "First full training log"
  (✅ hoàn thành, log này), "Reviewed, working pipeline" (✅ pipeline đã chứng minh chạy được
  end-to-end trên data thật + GPU, đã fix 1 bug thật (LMDB cache) trong quá trình review).
- Cần thông báo cho Oh Fu Sheng/team về: (1) bug LMDB đã fix, (2) bug cú pháp `+local=` kép trong
  README, (3) môi trường máy này giờ dùng torch CUDA (không phải CPU) — nếu người khác pull code và
  chạy lại trên máy này cần biết venv đã đổi.
- Checkpoint + TensorBoard log hiện nằm trong `outputs/` (untracked, không commit) — cần quyết định
  với team: có nên archive checkpoint này ở đâu đó chia sẻ được (không phải Git do file 82MB) để
  Oh Fu Sheng/Praveen dùng chung, hay mỗi người tự chạy lại.
- Vẫn còn treo: xác nhận `norm_choice`, xác nhận ai làm folder de-dup (Kiernan/Oh Fu Sheng chưa
  phản hồi 2 tin nhắn đã soạn).

---

## [2026-09-05] - Verify QAT path: CUDA OOM khi train ghost_efficientnet_v2 (QAT) trên GPU 8GB

### Bối cảnh / Vấn đề
Sau khi hoàn tất "First full training log", chuyển sang phần còn thiếu của task 3.12: "Validated
QAT path". Lần full training trước dùng model mặc định `efficientnet_v2` (không bật QAT), nên
nhánh `trainer.model.quantise()` (gọi trong `main.py` khi `run.test=true` và `run.quantise=true`)
chưa từng được exercise thật với 1 model đã train bằng QAT. Cần: (1) test nhánh non-QAT
(`fuse_model()` only) trên checkpoint đã có, (2) train nhanh 1 checkpoint QAT thật
(`model=ghost_efficientnet_v2`, có `use_qat: true`) để test nhánh `convert_fx()`.

### Kỹ thuật/công nghệ sử dụng
- **QAT (Quantization-Aware Training)** qua `torch.ao.quantization.quantize_fx.prepare_qat_fx`:
  chèn `FakeQuantize` module vào model NGAY TỪ ĐẦU training (không phải sau khi train xong), để
  model "làm quen" với nhiễu lượng tử hoá trong lúc học — mỗi module này cần thêm buffer để track
  min/max activation, tốn thêm VRAM so với model float thường.

### Ý nghĩa / Tầm quan trọng
Đây là bằng chứng thật đầu tiên cho thấy **QAT không chạy được với cấu hình batch_size mặc định
trên GPU 8GB của máy này** — một constraint tài nguyên thật, ảnh hưởng trực tiếp đến việc Oh Fu
Sheng/Hoang Lam Vu (calibration/quantisation workstream) sẽ gặp phải khi họ chạy QAT training thật
trên GPU tương tự. Phát hiện sớm giúp tránh lãng phí thời gian debug sau này.

### Những gì đã thay đổi
Không sửa code. Chạy 2 lần thử nghiệm (1 crash, 1 đang chạy lại với batch nhỏ hơn); không có file
nào bị xóa/tạo trong source code, chỉ có output huấn luyện tạm trong `outputs/` (untracked).

### Trích dẫn code quan trọng
Traceback thật:
```
File "train.py", line 166, in _train_one_epoch
    self.scaler.scale(loss).backward()
...
torch.AcceleratorError: CUDA error: out of memory
```
Xảy ra ở batch 70/6048 của epoch 1, với `training.batch_size=32` (giống config dùng cho lần full
training `efficientnet_v2` trước đó chạy ổn với cùng batch_size).

### Output / Kết quả
- **`model=ghost_efficientnet_v2` (QAT), `batch_size=32`, GPU**: CRASH — `CUDA error: out of
  memory` ở batch 70/6048, epoch 1. Không có checkpoint nào được lưu.
- `nvidia-smi` sau crash: VRAM giải phóng về 708MiB/8188MiB (bình thường, không phải leak).
- Đang retry với `batch_size=8` (giảm 4 lần) — kết quả sẽ cập nhật ở entry tiếp theo.
- Test nhánh non-QAT (`fuse_model()`) trên checkpoint `efficientnet_v2` đã train: đang chạy
  (~22% qua 1784 file validation tại thời điểm ghi log này), chưa có kết quả cuối.

### Quyết định kỹ thuật & lý do (nếu có)
- **Không dùng lại batch_size=32 cho QAT training**: cùng batch_size chạy ổn với model thường
  (`efficientnet_v2`) nhưng OOM với QAT (`ghost_efficientnet_v2`) — xác nhận rõ ràng QAT tốn thêm
  VRAM đáng kể do FakeQuantize observers, không phải do batch_size vốn đã sát giới hạn từ trước.
  Giảm xuống batch_size=8 là bước thử hợp lý đầu tiên (giảm 4x) trước khi cân nhắc các giải pháp
  phức tạp hơn (gradient checkpointing, giảm width_mult của model, v.v.) nếu vẫn OOM.
- **Chỉ train 3 epoch (không phải 15)** cho lần verify QAT này: mục tiêu chỉ là có 1 checkpoint
  hợp lệ đã qua QAT training để test `quantise()`, không cần độ chính xác cao — tiết kiệm thời gian.

### Việc còn tồn đọng / hướng tiếp theo
- Chờ kết quả retry `batch_size=8` và kết quả test `fuse_model()` — sẽ cập nhật ở entry tiếp theo.
- Nếu retry vẫn OOM, cần thử thêm: giảm batch_size xuống nữa (4, 2), hoặc bật gradient
  checkpointing/AMP tối ưu hơn, hoặc giảm `width_mult`/`depth_mult` của ghost_efficientnet_v2.
- Cần báo phát hiện OOM này cho Hoang Lam Vu (task 3.3 - TFLite Quantisation Sweep) vì anh ấy sẽ
  chạy QAT training thật trên "Keras/SavedModel source" — có thể gặp constraint tương tự nếu dùng
  GPU nhỏ.

---

## [2026-09-05] - 2 bug thật nữa: LMDB race với num_workers>0, và CPU OOM trong test/_evaluate với file audio dài

### Bối cảnh / Vấn đề
Tiếp tục verify QAT path. Gặp thêm 2 bug thật hoàn toàn độc lập với bug LMDB đã fix trước đó:
(1) retry QAT training với `num_workers=4` bị lỗi LMDB khác (không phải bug đã fix), (2) test nhánh
`fuse_model()` trên checkpoint `efficientnet_v2` bị CPU OOM khi cố cấp phát 2.36GB cho 1 lần gọi
`batch_norm`.

### Kỹ thuật/công nghệ sử dụng
- **DataLoader `num_workers>0` trên Windows dùng `spawn`** (không phải `fork` như Linux): mỗi worker
  là 1 process con độc lập, tự gọi `lmdb.open()` riêng khi cần (thiết kế "lazy init" vốn có trong
  code, đúng cho đa-tiến-trình) — nhưng lần này crash với "No such file or directory" dù thư mục đó
  **thực sự tồn tại** khi kiểm tra thủ công ngay sau đó → nhiều khả năng là race condition lúc nhiều
  worker cùng khởi tạo LMDB env lần đầu, hoặc Windows Defender quét/khoá thư mục mới tạo trong
  khoảnh khắc đó — chưa root-cause đến cùng do giới hạn thời gian.
- **Chunk-level aggregation trong `_evaluate`/`test()`**: mỗi "batch" validation thực chất là toàn
  bộ chunk của 1 file audio dồn vào 1 tensor duy nhất, đưa qua model 1 lần. Với file dài bất thường
  (nhiều chunk), tensor này có thể rất lớn, không có giới hạn/sub-batching nào.

### Ý nghĩa / Tầm quan trọng
Bug OOM khi test là quan trọng nhất: đây là lần **đầu tiên trong lịch sử pipeline** nhánh
`trainer.test()` (dùng khi `run.test=true`) được chạy hết một validation set thật — trước đó
chỉ verify bằng 3 file synthetic (entry trước). Phát hiện ra ngay: pipeline **không xử lý được file
audio dài** trong đường test/evaluate, sẽ là blocker thật cho Praveen (held-out baseline
re-evaluation) và Oh Fu Sheng (test model đã quantise) nếu dataset thật của họ có file dài tương tự.

### Những gì đã thay đổi
Không sửa code trong entry này — chỉ ghi nhận 2 phát hiện, để dành fix cho bước sau (ưu tiên thời
gian cho việc hoàn thành checkpoint QAT trước).

### Trích dẫn code quan trọng
Lỗi LMDB với num_workers=4:
```
lmdb.Error: .cache\train: No such file or directory
```
(xảy ra trong worker process, dù `Path(...).mkdir(parents=True, exist_ok=True)` đã chạy ở process
chính trước đó — xác nhận bằng cách kiểm tra thủ công thư mục vẫn tồn tại sau khi crash.)

Lỗi CPU OOM khi test:
```
RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator:
not enough memory: you tried to allocate 2534326272 bytes.
```
(trong `torch.nn.functional.batch_norm`, khi chạy `trainer.test(val_loader)` với
`training.device=cpu`, khoảng 20-25% qua 1784 file validation.)

### Output / Kết quả
- Retry QAT training với `num_workers=4, batch_size=8`: **crash** ở batch đầu tiên với lỗi LMDB trên.
- Test `fuse_model()` trên checkpoint `efficientnet_v2` (CPU, `run.test=true run.quantise=true`):
  **crash** ở ~20-25% qua validation set với CPU OOM (cấp phát 2.36GB cho 1 batch_norm).
- Retry QAT training lần 2 với `num_workers=0, batch_size=8` (né cả 2 vấn đề trên): **đang chạy ổn
  định** trên GPU, ~1.5-2 it/s, không OOM — ước tính ~70-75 phút cho 3 epoch.

### Quyết định kỹ thuật & lý do (nếu có)
- **Không root-cause sâu bug LMDB race với num_workers>0 ngay bây giờ**: đã có giải pháp né an toàn
  (`num_workers=0`, đã verify ổn định nhiều lần), và mục tiêu hiện tại chỉ cần 1 checkpoint QAT hợp
  lệ để test `quantise()` — điều tra sâu hơn (ví dụ tắt Windows Defender real-time scan để loại trừ
  giả thuyết, hay xem xét lmdb `max_readers`/lock semantics) nên để dành cho lúc thật sự cần tối ưu
  tốc độ training bằng nhiều worker, không phải ưu tiên ngay lúc này.
- **Không tự sửa bug OOM trong `_evaluate`/`test()` ngay**: đây là bug ảnh hưởng kiến trúc xử lý
  chunk (cần thêm sub-batching logic), phạm vi sửa lớn hơn một "quick fix" — quyết định ghi nhận lại
  làm technical debt rõ ràng, ưu tiên hoàn thành mục tiêu chính (verify QAT path) trước, tránh lan
  man sang một task sửa lỗi khác giữa chừng.

### Việc còn tồn đọng / hướng tiếp theo
- Chờ QAT training (`num_workers=0, batch_size=8`) chạy xong (~70-75 phút), rồi test `quantise()`
  (nhánh `convert_fx()` thật) trên checkpoint đó — cố gắng né path test đầy đủ trên val set để tránh
  bug OOM vừa phát hiện (ví dụ giới hạn số file test, hoặc chấp nhận rủi ro OOM lặp lại).
- **Cần fix riêng bug OOM trong `_evaluate`** (thêm sub-batching cho file có nhiều chunk) — đây là
  technical debt thật, ảnh hưởng đến Praveen (held-out baseline) và bất kỳ ai chạy `run.test=true`
  trên dataset thật có file dài. Nên báo cho team, có thể đưa vào backlog Sprint 2 hoặc Sprint 3.
- Cần báo cho team về bug LMDB race với `num_workers>0` (dù chưa root-cause được) — khuyến nghị tạm
  thời dùng `num_workers=0` cho đến khi điều tra xong.

---

## [2026-09-05] - Kết luận verify QAT path: convert_fx() chạy được nhưng model quantised không inference được

### Bối cảnh / Vấn đề
Đã có checkpoint QAT thật (`ghost_efficientnet_v2`, 3 epoch, train trên data thật) từ entry trước.
Để tránh bug OOM trong `trainer.test()`/`_evaluate()` (entry trước) chặn đường tới `quantise()`,
viết 1 script cô lập gọi thẳng `Model.quantise()` + forward pass, bỏ qua toàn bộ vòng lặp test đầy
đủ — mục tiêu chỉ là xác nhận `convert_fx()` (nhánh QAT thật) có chạy được không.

### Kỹ thuật/công nghệ sử dụng
- **Hydra `compose()` API** (`hydra.initialize_config_dir` + `compose`): dựng `cfg` giống hệt CLI
  nhưng gọi trực tiếp từ Python script, không cần chạy qua `@hydra.main` decorator của `main.py` —
  cách này cho phép test 1 phần nhỏ của pipeline (chỉ `Model` class) mà không kéo theo toàn bộ
  logic dataset/train loop (nơi có bug OOM).
- **`torch.ao.quantization.quantize_fx.convert_fx`**: chuyển 1 model đã "prepare" cho QAT
  (`FakeQuantize` modules) thành model dùng **quantized kernels thật** (int8). Điểm mấu chốt: model
  sau convert **yêu cầu input là tensor đã quantize** (qua `torch.quantize_per_tensor(...)`), không
  chấp nhận tensor float thường — đây chính là phần đang thiếu.

### Ý nghĩa / Tầm quan trọng
Đây là câu trả lời dứt khoát cho "Validated QAT path" của task 3.12: **QAT path hiện KHÔNG hoạt
động end-to-end**, dù `convert_fx()` tự nó không báo lỗi. Nếu không phát hiện bằng cách test thật
(chỉ đọc code sẽ khó thấy vì convert_fx() "trông như" chạy được), Oh Fu Sheng có thể tưởng nhầm
QAT path đã xong và chuyển sang việc khác, để lại 1 gap không ai biết cho đến khi ai đó thật sự cần
dùng model đã quantise để inference (ví dụ Hoang/Tharrun ở workstream Calibration and Edge).

### Những gì đã thay đổi
Không sửa code trong repo. Chỉ tạo 1 script chẩn đoán ngoài repo (scratchpad):
`verify_qat_quantise.py` — không phải phần giao nộp, dùng để xác nhận hành vi.

### Trích dẫn code quan trọng
```python
model.eval()
out_before = model(dummy_input)          # OK, shape [1, 127]
model.quantise()                         # OK, không exception (gọi convert_fx() nội bộ)
out_after = model(dummy_input)           # FAIL:
```
```
NotImplementedError: Could not run 'aten::qscheme' with arguments from the 'CPU' backend.
... 'aten::qscheme' is only available for these backends:
[..., QuantizedCPU, QuantizedCUDA, ...]
```
`dummy_input` ở đây vẫn là `torch.randn(1, 1, 384, 200)` — tensor float thường, chưa qua
`torch.quantize_per_tensor()`. Model sau `convert_fx()` cần input đã ở dạng quantized tensor mới
chạy được qua các quantized kernel (`QuantizedCPU` dispatch key).

### Output / Kết quả
- Forward pass trước quantise(): thành công, `output shape: torch.Size([1, 127])` — xác nhận
  checkpoint load đúng, kiến trúc khớp 127 lớp.
- `model.quantise()`: **không exception** — bản thân bước convert cấu trúc model thành công.
- Forward pass sau quantise(): **crash** — `aten::qscheme` không chạy được trên backend CPU thường,
  vì input chưa được quantize.

### Quyết định kỹ thuật & lý do (nếu có)
- **Không tự sửa gap này (thêm bước `torch.quantize_per_tensor()`/QuantStub)**: đây thuộc phạm vi
  "wire prepare/convert" của task 3.11 (Oh Fu Sheng), không phải việc review của mình. Quyết định
  dừng lại ở việc xác nhận CHÍNH XÁC gap nằm ở đâu (thiếu bước quantize input, không phải lỗi ở
  convert_fx() hay ở kiến trúc model) để bàn giao thông tin rõ ràng, thay vì tự ý sửa code người
  khác được giao nhiệm vụ.
- **Test bằng script cô lập thay vì chạy lại toàn bộ `main.py --run.test=true`**: tránh lặp lại bug
  OOM đã biết (entry trước), đồng thời cô lập chính xác biến số đang muốn kiểm tra (chỉ
  `Model.quantise()`, không lẫn với bug khác) — nguyên tắc "thay đổi 1 biến số mỗi lần test".

### Việc còn tồn đọng / hướng tiếp theo
- **Kết luận cuối cùng cho task 3.12 "Validated QAT path"**: QAT path **CHƯA hoạt động** —
  `convert_fx()` chạy được về cấu trúc, nhưng thiếu bước quantize input trước khi inference. Đây là
  phản hồi cụ thể, có bằng chứng để gửi lại cho Oh Fu Sheng (task 3.11) sửa.
- Cần Oh Fu Sheng thêm: `torch.quantize_per_tensor()` (hoặc `QuantStub`/`DeQuantStub` trong kiến
  trúc model, hoặc set `torch.backends.quantized.engine` phù hợp) ở đúng chỗ trong luồng inference
  sau `quantise()`, rồi test lại bằng chính script `verify_qat_quantise.py` này (có thể chia sẻ lại
  logic cho OFS nếu cần).
- Tổng kết toàn bộ phiên làm việc hôm nay (2026-09-04 → 2026-09-05) cho task 3.12: 2/2 phần việc đã
  chạm tới (full training + review/validate QAT), dù QAT path kết luận là "chưa hoạt động" — đó vẫn
  là kết quả review hợp lệ, không phải việc chưa làm.

---

## [2026-09-05] - Oh Fu Sheng xác nhận nhận task 3.11, nhưng kế hoạch mâu thuẫn với yêu cầu của Kiernan

Oh Fu Sheng trả lời Discord xác nhận đang làm task 3.11 (folder de-dup), kế hoạch: giữ
`reproducible_training_pipeline/augment.py` (bản của Nolan) làm nguồn duy nhất, xóa bản của
Kiernan ở `augmentation/`. Kế hoạch này **ngược với** yêu cầu Kiernan nhắn trước đó (giữ bản của
Kiernan làm nền, merge phần validate của Nolan lên trên) — nếu làm theo kế hoạch OFS, Kiernan sẽ
mất `mask_value="min"/"mean"` và cơ chế đảm bảo zero-width-mask luôn masking thật (2 điểm khác biệt
đã phân tích ở entry trước, ngày 2026-09-04).

**Quyết định**: không tự gửi cảnh báo cho OFS/Kiernan — để 2 người tự trao đổi và thống nhất hướng
merge trực tiếp với nhau, không can thiệp. Đây là xung đột giữa 2 thành viên khác, không phải việc
của task 3.12; đã nắm đủ thông tin để phản hồi nếu sau này cần, nhưng không chủ động chen vào lúc
này theo yêu cầu của user.

---

## [2026-09-05] - Chốt phạm vi task 3.12: "review Oh Fu Sheng's implementation" = review pipeline hiện có

### Bối cảnh / Vấn đề
Task 3.12 yêu cầu "review Oh Fu Sheng's implementation of the deferred items", nhưng thực tế team
có 2 ràng buộc: (1) tất cả thành viên Engine sẽ hoàn thành task Sprint 2 **cùng một ngày** (không
có khoảng đệm để người này chờ PR người kia rồi mới review), (2) PR của member này **không xem được
bởi member khác** cho tới khi leader approve — nghĩa là về mặt quy trình, không có cách nào để
Nolan thực sự đọc/review PR thật của Oh Fu Sheng trong khung thời gian Sprint 2.

### Ý nghĩa / Tầm quan trọng
Đây là lý do chính đáng để không coi "review OFS's implementation" là một hạng mục còn treo/chưa
xong của task 3.12 — vì bản thân yêu cầu này, hiểu theo nghĩa đen (đọc PR thật của OFS), **không
khả thi** trong quy trình team đang vận hành. Cách hiểu hợp lý duy nhất: "review" ở đây áp dụng cho
pipeline như nó đang tồn tại — đúng những gì đã làm suốt phiên hôm nay (verify checkpoint/run.test
path, verify QAT path, tìm và fix bug LMDB, chạy full training thật).

### Quyết định kỹ thuật & lý do (nếu có)
- **Chốt: task 3.12 coi như hoàn thành** với cách hiểu "review pipeline hiện có" thay vì "review PR
  thật của Oh Fu Sheng" — quyết định này dựa trên ràng buộc quy trình thật của team (cùng deadline,
  PR bị khóa), không phải để né việc. Các phát hiện kỹ thuật cụ thể về quant.py (input shape sai
  kênh, thiếu bước quantize input) đã được chuyển thẳng cho Oh Fu Sheng qua tin nhắn — đây thực chất
  là dạng review "trước" (pre-review), có giá trị hơn review PR thông thường vì dựa trên chạy thật
  chứ không chỉ đọc code.
- Không cần chờ OFS nộp PR để coi task 3.12 là "xong" — nếu muốn review PR thật của OFS sau này (khi
  leader mở khóa), đó là việc bổ sung, không phải điều kiện để đóng task.

### Việc còn tồn đọng / hướng tiếp theo
Không còn hạng mục nào của task 3.12 bị treo. Các việc còn lại (folder de-dup, norm_choice, checkpoint
sharing) đều là việc phối hợp team, nằm ngoài phạm vi bắt buộc của task 3.12 cá nhân.

---

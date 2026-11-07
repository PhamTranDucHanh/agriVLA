# Duyệt dataset AgriVLA trên localhost — bước 4.5

Bản v0 ở data/manifests/v0 và preview tĩnh vẫn giữ nguyên. Bản hiện hành ở **data/manifests/wording_v1**. Câu “tưới cả 3 vườn” phải kèm mức 1/2/3; không tự đoán mức. “Tắt hết” là STOP_ALL.

Trang HTML này đọc **toàn bộ 5.040 record của wording_v1** (3.600 train / 720 val / 720 test), không giới hạn 72 mẫu như preview tĩnh bước 4. Có thêm tab xem **1.892 ảnh nguồn**, gồm 104 ảnh nhãn 5 vẫn bị quarantine. Ảnh/nhãn/splits/manifests và dataset lock không bị sửa khi duyệt.

## 1. Chạy server

Mở terminal trên máy chứa repo, chạy:

```bash
cd /home/hanhpham-duc/agriVLA
bash agri-soybean-vla/scripts/browse.sh
```

Không cần activate môi trường, npm, Torch hay GPU. Script dùng đúng agri-soybean-vla/.venv. Server kiểm dataset lock, manifests và ảnh trước khi phục vụ; chờ terminal in:

```text
Dataset QA PASS · 5,040 mẫu · 1,892 ảnh nguồn
Mở trình duyệt: http://127.0.0.1:8765/
Giữ terminal này mở. Nhấn Ctrl+C để dừng server.
```

Mở **http://127.0.0.1:8765/** trong Chrome/Firefox/Edge. Giữ terminal chạy trong lúc duyệt. Nếu URL đang mở được, nhìn badge wording_v1 và tổng 5.040 mẫu để xác nhận đúng bản. Nếu vẫn thấy 4.200 mẫu hoặc vừa sửa source, chạy `bash agri-soybean-vla/scripts/browse.sh --replace` rồi Ctrl+Shift+R. Lệnh replace chỉ thay browser AgriVLA cùng repo/user, sau khi QA PASS; không dừng dịch vụ khác.

Nếu port 8765 đang được dùng bởi chương trình khác:

```bash
bash agri-soybean-vla/scripts/browse.sh --port 8766
```

Sau đó mở http://127.0.0.1:8766/. Dừng server bằng Ctrl+C trong terminal đã chạy lệnh. Khởi động lại bằng cùng lệnh, không sinh lại dataset.

Kiểm tra trước, không mở port và không ghi file:

```bash
bash agri-soybean-vla/scripts/browse.sh --dry-run
bash agri-soybean-vla/scripts/browse.sh --help
```

Nếu thiếu .venv, chạy setup data rồi thử lại:

```bash
bash agri-soybean-vla/scripts/setup.sh --mode data
```

## 2. Xem tất cả mẫu

Chọn tab **Mẫu tổng hợp**. Ban đầu trang hiện 1–12 / 5.040 mẫu, theo thứ tự dòng JSONL: train → val → test. Mỗi card có:

- Số thứ tự toàn dataset và dòng trong manifest.
- Split, ngôn ngữ VI/EN, seen/unseen.
- Câu lệnh, ý định được giải thích, params và loại ca.
- Ba ảnh đúng thứ tự vùng 1/2/3; mỗi vùng có teacher wilt và target level.
- Target vector [zone1, zone2, zone3].

Cuộn để xem các card; chọn 12/24/48 mẫu mỗi trang. Dùng **Sau / Trước / Đầu / Cuối**, hoặc nhập số trang rồi bấm **Đi**. Với 12 mẫu/trang có 420 trang; với 48 mẫu/trang có 105 trang. **Tới mẫu số** nhận số 1–5.040 và mở thẳng chi tiết mẫu đó, không cần cuộn qua các trang trước.

## 3. Lọc và tìm kiếm

Các bộ lọc kết hợp với nhau. Mỗi lần đổi lọc, trang tự quay về trang đầu của kết quả.

| Muốn xem | Thao tác | Kỳ vọng |
|---|---|---|
| Tưới bắt buộc cả ba | Intent = WATER_ALL | 840 record toàn dataset; target [level,level,level], không xét ngưỡng |
| Tắt hết | Tìm `tắt hết` | Chỉ STOP_ALL, target [0,0,0] |
| Toàn bộ test | Split = Test | 720 record |
| Lệnh Anh chưa thấy cách diễn đạt khi train | Split = Validation, ngôn ngữ = English, wording = Unseen | 180 record |
| STOP của nhóm trên | Thêm ý định STOP_ALL | 20 record, mọi target [0,0,0] |
| Action toàn mức cao | Mở Lọc thêm → Action = Toàn HIGH | Các target [3,3,3] |
| Hòa mức héo cao nhất | Lọc thêm → Loại ca = Hòa mức héo cao nhất | Panel wilt [4,4,1] với nhiều intent |
| Ảnh từ một plot/camera | Lọc thêm → Camera–plot | Mẫu có ít nhất một ảnh của nhóm đó |
| Có một vùng wilt 4 | Nhãn héo = Wilt 4 | Mẫu có ít nhất một ảnh được nguồn gán 4 |
| Câu Việt hoặc tên ảnh cụ thể | Gõ ô tìm kiếm | Tìm theo câu, sample/scene/case/family ID, tên ảnh và nhóm nguồn |

Tìm kiếm tiếng Việt chấp nhận bỏ dấu: thử `tat het`. Xóa bộ lọc để về đủ dataset. Thông báo không có kết quả là trạng thái bình thường nếu tổ hợp lọc không có mẫu; không có nghĩa là mất dataset.

## 4. Xem một mẫu, cặp dịch và đối chiếu lệnh

Nhấn **Chi tiết**:

1. Đọc instruction và target; ba ảnh vẫn giữ đúng thứ tự zone.
2. Bảng bản dịch hiện 2 dòng cho train hoặc 4 dòng cho val/test. Các dòng cùng case phải có cùng ảnh/teacher/target, khác ngôn ngữ và/hoặc wording.
3. Mở **Ảnh gốc, đường dẫn local & checksum** để xem camera/plot/timestamp, absolute path và SHA256 từng ảnh.
4. Mở **Toàn bộ record JSON & provenance** để xem mọi field.
5. **Tải JSON mẫu** tải đúng record gốc, không thêm trường của UI. **Copy link mẫu** tạo link localhost mở lại đúng mẫu đó khi server đang chạy.
6. Dùng **Đóng** hoặc phím Esc để quay lại danh sách.

Hai nút đối chiếu có ở cả card và chi tiết:

- **Bản dịch & wording:** chỉ hiện các record trong cùng case.
- **Cùng ảnh · lệnh khác:** hiện mọi case dùng đúng bộ ba ảnh có thứ tự đó, tiện đối chiếu AUTO/WORST/ZONE/CAP/EXCLUDE/STOP/WATER_ALL.

## 5. Mở đúng ảnh và thư mục gốc

Nhấp vào **ảnh** trên card để mở ảnh nguyên gốc trong tab mới. URL /original/... được server tra từ source manifest và kiểm encoded checksum; không phải ảnh giả hoặc đường dẫn tải bên ngoài. Thumbnail chỉ để trang tải nhanh; không thay ảnh gốc.

**Thư mục gốc** mở danh sách file thật qua localhost. Ví dụ:

- http://127.0.0.1:8765/files/images/ — 1.892 ảnh gốc.
- http://127.0.0.1:8765/files/manifests/ — train/val/test JSONL, lock, QA và card.
- http://127.0.0.1:8765/files/raw/ — CSV, ZIP và liên kết thư mục ảnh.
- http://127.0.0.1:8765/files/source-splits/ — các source manifests đã khóa.

Tab **Thư mục & tệp gốc** hiển thị absolute paths thật và nút copy. Đây là danh sách file qua HTTP local, không phải lệnh tự mở file manager của hệ điều hành. Trình duyệt thường hạn chế liên kết file:// từ trang HTTP; dùng các link HTTP đã có để mở file ổn định.

Nếu muốn mở bằng cửa sổ Files của Linux, chạy trong terminal máy chứa dữ liệu:

```bash
xdg-open /home/hanhpham-duc/agriVLA/agri-soybean-vla/data/raw/zenodo_8256382/images
xdg-open /home/hanhpham-duc/agriVLA/agri-soybean-vla/data/manifests/wording_v1
```

## 6. Duyệt ảnh nguồn

Tab **Ảnh nguồn** hiển thị đầy đủ 1.892 ảnh, khác với 1.685 ảnh thực sự được chọn vào scene dataset. Có thể lọc split, nhóm nguồn, nhãn 0–5 và trạng thái:

- **Đủ điều kiện v0:** 1.788 ảnh miền 0–4.
- **Quarantine · nhãn 5:** 104 ảnh, không thuộc train/val/test của v0 hoặc wording_v1.
- **Eligible chưa được chọn:** 103 ảnh; không phải ảnh hỏng, chỉ chưa được chọn vào panel val/test.

Mỗi ảnh hiện số record tái sử dụng. Nút **Xem các mẫu dùng ảnh** lọc scene dataset theo source ID; không hiện nút này nếu ảnh chưa được chọn hoặc bị quarantine.

## 7. Cách hiểu và xử lý tình huống thường gặp

- Scene là bộ ba ảnh có thứ tự; case là scene + intent/params; record là một câu thuộc một ngôn ngữ. Tổ hợp/bản dịch không làm tăng số ảnh độc lập.
- State [0,0,0] là giả định mọi van ban đầu đóng. Teacher wilt/structured intent/provenance chỉ là metadata, không là features model.
- Action 0/1/2/3 = OFF/LOW/MEDIUM/HIGH, chưa phải lượng nước hiệu chuẩn. Nhóm lệnh theo héo cần ngưỡng 2; WATER_ALL ép mức chỉ định bất kể héo, STOP_ALL luôn toàn 0. Tab Cách đọc mẫu nhắc lại ngay trên trang.
- Nếu không kết nối được: chờ terminal báo URL, kiểm port đúng và server còn chạy; Ctrl+C rồi chạy lại nếu cần. Server chỉ bind 127.0.0.1.
- Nếu dùng SSH/VS Code Remote và trình duyệt ở máy khác: forward port 8765 về máy bạn (VS Code: Ports → Forward a Port → 8765). localhost phải trỏ tới máy đang chạy server hoặc port đã forward.
- Nếu dataset mới thay đổi, khởi động lại server để đọc snapshot/QA mới. Checksum/lock sai sẽ bị báo lỗi, không tự sửa raw hoặc labels.
- Không mở trực tiếp src/agri_vla/web/browser.html bằng file manager: giao diện đầy đủ cần API local, hãy dùng URL localhost sau khi chạy server.

Source: browser.py, web/browser.html/browser.css/browser.js; CLI browse-revision và scripts/browse.sh; legacy v0 vẫn có CLI browse. Không cần dependency runtime mới. Bước 4.5 không convert/train, không commit/push. Kết quả test và screenshot thật được ghi trong AGRIVLA_IMPLEMENTATION_STATUS.md và reports/browser/step45/.

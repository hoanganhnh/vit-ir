# Hướng dẫn Thu thập Ảnh Vệ tinh Chữ cái (Manual)

## Mục tiêu
Thu thập 50-100 ảnh vệ tinh cho MỖI chữ cái (A-Z) từ hình dạng tự nhiên/nhân tạo.

## Trạng thái hiện tại (từ NASA scraper)
- **A**: ⚠️ 0/50 (cần thêm 50)
- **B**: ⚠️ 0/50 (cần thêm 50)
- **C**: ⚠️ 0/50 (cần thêm 50)
- **D**: ⚠️ 0/50 (cần thêm 50)
- **E**: ⚠️ 0/50 (cần thêm 50)
- **F**: ⚠️ 0/50 (cần thêm 50)
- **G**: ⚠️ 0/50 (cần thêm 50)
- **H**: ⚠️ 0/50 (cần thêm 50)
- **I**: ⚠️ 0/50 (cần thêm 50)
- **J**: ⚠️ 0/50 (cần thêm 50)
- **K**: ⚠️ 0/50 (cần thêm 50)
- **L**: ⚠️ 0/50 (cần thêm 50)
- **M**: ⚠️ 0/50 (cần thêm 50)
- **N**: ⚠️ 0/50 (cần thêm 50)
- **O**: ⚠️ 0/50 (cần thêm 50)
- **P**: ⚠️ 0/50 (cần thêm 50)
- **Q**: ⚠️ 0/50 (cần thêm 50)
- **R**: ⚠️ 0/50 (cần thêm 50)
- **S**: ⚠️ 0/50 (cần thêm 50)
- **T**: ⚠️ 0/50 (cần thêm 50)
- **U**: ⚠️ 0/50 (cần thêm 50)
- **V**: ⚠️ 0/50 (cần thêm 50)
- **W**: ⚠️ 0/50 (cần thêm 50)
- **X**: ⚠️ 0/50 (cần thêm 50)
- **Y**: ⚠️ 0/50 (cần thêm 50)
- **Z**: ⚠️ 0/50 (cần thêm 50)

## Nguồn thu thập bổ sung

### 1. Google Earth Pro (FREE)
- Download: https://www.google.com/earth/versions/#earth-pro
- Mở Google Earth → Tìm các hình dạng tự nhiên → Screenshot → Crop 224x224

### 2. NASA Worldview
- URL: https://worldview.earthdata.nasa.gov/
- Tìm kiếm theo khu vực → Screenshot

### 3. Sentinel Hub EO Browser
- URL: https://apps.sentinel-hub.com/eo-browser/
- Free tier, ảnh Sentinel-2 resolution ~10m

## Gợi ý hình dạng cho từng chữ cái

| Chữ | Hình dạng tự nhiên | Hình dạng nhân tạo |
|-----|--------------------|---------------------|
| A | Núi nhọn, tam giác delta | Tháp, cầu dây |
| B | Hai hồ nước cạnh nhau | Sân vận động đôi |
| C | Vịnh cong, bờ biển cong | Đường cong cao tốc, dam |
| D | Hồ hình bán nguyệt | Sân bay, nhà kho |
| E | Nhánh sông song song | Bến cảng, đường sắt |
| F | Nhánh sông chữ F | Giao lộ, cấu trúc nhà máy |
| G | Bờ biển vòng cung | Bến tàu vòng cung |
| H | Hai con sông + cầu nối | Đường giao nhau, cánh đồng |
| I | Sông thẳng, khe núi | Đường cao tốc thẳng, đập |
| J | Sông uốn cong 1 đầu | Bến cảng hình móc |
| K | Nhánh sông phân kỳ | Giao lộ phức tạp |
| L | Sông gấp khúc 90° | Góc đường, bờ kè |
| M | Dãy núi zigzag | Cầu dây văng |
| N | Sông zigzag | Đường núi ngoằn ngoèo |
| O | Hồ tròn, miệng núi lửa | Sân vận động, đảo san hô |
| P | Bán đảo + cổ đất | Bến cảng chữ P |
| Q | Hồ tròn + suối chảy ra | Sân bay vòng xoay |
| R | Sông phân nhánh | Đường phân nhánh |
| S | Sông uốn khúc | Đường serpentine |
| T | Hợp lưu sông chữ T | Giao lộ chữ T, đường băng |
| U | Thung lũng hình chữ U | Đập nước, sân vận động |
| V | Thung lũng hẹp, delta ngược | Đường giao nhau góc nhọn |
| W | Bờ biển zigzag | Đường núi, nhà kho |
| X | Giao điểm hai sông | Giao lộ, đường sắt |
| Y | Hợp lưu ba nhánh sông | Ngã ba đường |
| Z | Sông zigzag hình Z | Đường zigzag núi |

## Quy tắc khi thu thập
1. **Kích thước**: Crop ảnh về 224×224 pixels hoặc tỉ lệ 1:1
2. **Chất lượng**: Ảnh rõ ràng, hình dạng chữ cái dễ nhận ra
3. **Đa dạng**: Thu thập từ nhiều vùng địa lý, nhiều loại hình dạng
4. **Đặt tên**: `{nguồn}_{chữ_cái}_{số_thứ_tự}.jpg`
   - Ví dụ: `ge_S_001.jpg` (Google Earth, chữ S, ảnh thứ 1)
5. **Lưu vào**: `dataset/satellite_letters/raw/google_earth/{chữ_cái}/`

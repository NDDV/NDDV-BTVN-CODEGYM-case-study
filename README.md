# NDDV-BTVN-CODEGYM-case-study
[Case Study] Phân tích các nhân tố ảnh hưởng tới giá xe hơi
Như vậy trong quá trình phân tích người học có thể nhận thấy các thông tin sau:
-	Bộ dữ liệu gồm 205 dòng, 26 thuộc tính.
-	Tên của một số hãng ô tô bị viết sai chính tả → thực hiện việc sửa sai
-	Giá ô tô có sự biến động rất lớn: trung bình khoảng 10,000 usd/chiếc, giá loại thấp nhất hơn 5.000 usd/chiếc, giá loại cao nhất hơn 45.000 use/chiếc. Phần lớn ô tô được thiết kế với ở phân khúc giá rẻ (5-10 nghìn use/chiếc) và trung cấp (10-20 nghìn usd/chiếc)
-	Về thông tin xe được yêu thích
-	Toyota có vẻ là hãng xe chiếm thị phần cao nhất.
-	Số lượng ô tô chạy bằng nhiên liệu khí nhiều hơn động cơ diesel.
-	Sedan là loại xe được ưa chuộng nhất.
-	Những yếu tố ảnh hưởng tới giá xe dựa trên biểu đồ phân tích
-	Động cơ ohcv có giá cao nhất
-	Hai hãng Jaguar and Buick dường như có giá xe cao nhất
-	Động cơ diesel có giá cao hơn
- Loại bánh xe cũng có mối liên hệ với giá
-	Những thuộc tính quan trọng trong mô hình hồi quy dự báo giá xe:
-	Có 15 yếu tố khi phân tích định tính thấy rằng có thể ảnh hưởng tới giá xe: 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'curbweight',
'enginetype', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'fueleconomy', 'carlength', 'carwidth', 'carsrange
-	8 yếu tố ảnh hưởng tới giá xe nhiều nhất qua phân tích định lượng bằng mô hình hồi quy tuyến tính: 'wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'fueleconomy', 'carlength', 'carwidth'

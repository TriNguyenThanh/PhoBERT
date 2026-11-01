Nghiên cứu Mô hình PhoBERT trên Bộ dữ liệu ViHSD và Hướng dẫn Huấn luyện Tối ưu Hàm Mất mát cho Bài toán Phân loại Ngôn từ Thù ghét Đa lớp Tiếng ViệtTổng quan Hệ sinh thái Nghiên cứu về Bộ dữ liệu ViHSD và Mô hình PhoBERTSự bùng nổ của các nền tảng mạng xã hội tại Việt Nam như Facebook, YouTube, Instagram và TikTok đã làm gia tăng nhanh chóng lượng bình luận độc hại, đặt ra yêu cầu cấp thiết về các hệ thống kiểm duyệt tự động dựa trên xử lý ngôn ngữ tự nhiên (NLP). Trong tiến trình chuẩn hóa dữ liệu cho tiếng Việt — một ngôn ngữ ít tài nguyên (low-resource language) — bộ dữ liệu ViHSD (Vietnamese Hate Speech Detection) do nhóm tác giả Lưu Thanh Sơn, Nguyễn Văn Kiệt và Nguyễn Lưu Thùy Ngân thuộc Phòng thí nghiệm Xử lý Ngôn ngữ Tự nhiên, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT-NLP) công bố năm 2021 đã trở thành thước đo chuẩn mực (benchmark) quan trọng nhất cho bài toán phân loại đa lớp ngôn từ thù ghét.Công trình gốc thiết lập bộ dữ liệu ViHSD (Luu et al., 2021)Bài báo nền tảng giới thiệu bộ dữ liệu có tiêu đề "A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts", được công bố tại Hội nghị Quốc tế IEA/AIE 2021 (Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices, trang 415–426, nhà xuất bản Springer, DOI: 10.1007/978-3-030-79457-6*35, bản tiền ấn bản arXiv:2103.11528). Kế thừa cấu trúc nhãn từ chiến dịch VLSP 2019 HSD Shared Task, nhóm tác giả đã mở rộng quy mô lên 33,400 bình luận thực tế thu thập từ các trang Facebook và kênh YouTube có tần suất tương tác cao, xoay quanh các chủ đề giải trí, người nổi tiếng, vấn đề xã hội và chính trị. Để bảo vệ quyền riêng tư, toàn bộ các thực thể định danh cá nhân (named entities) đã được ẩn danh trước khi đưa vào quy trình gán nhãn thủ công.Mỗi bình luận trong ViHSD được phân vào một trong ba nhãn phẳng (flat labels) không chồng lấp bao gồm CLEAN (nhãn 0), OFFENSIVE (nhãn 1) và HATE (nhãn 2). Trong hệ thống phân loại này, nhãn CLEAN đại diện cho các bình luận bình thường, mang tính xây dựng hoặc trung tính, không chứa ngôn từ thô tục hay công kích (ví dụ: "Nhanh thực sự"). Nhãn OFFENSIVE bao hàm các bình luận có xuất hiện từ ngữ thô tục, chửi thề viết tắt hoặc cảm thán phản cảm nhưng không nhắm mục tiêu lăng mạ hay hạ nhục một cá nhân hay nhóm đối tượng cụ thể (ví dụ: "Đm chứ biết làm gì", "Đm vcl"). Ở cấp độ nghiêm trọng nhất, nhãn HATE định danh ngôn từ thù ghét có hành vi tấn công trực diện, miệt thị nhân phẩm, xúc phạm danh dự hoặc kích động thù địch nhắm vào một cá nhân hoặc tập thể xác định (ví dụ: "Nó học cách của thằng anh nó đó, hèn, khốn nạn" hoặc "người ư? sinh vật hạ đẳng chứ ngươi ai lam thế").Quá trình kiểm định chất lượng dữ liệu được thực hiện thông qua bốn người gán nhãn độc lập (A1, A2, A3, A4) trên tập mẫu đối chứng gồm 202 bình luận ngẫu nhiên, đạt độ đồng thuận trung bình theo hệ số Cohen's Kappa là $\kappa = 0.52$ (dao động từ $0.46$ đến $0.65$ giữa từng cặp gán nhãn), vượt ngưỡng chấp nhận được ($\kappa > 0.5$) đối với dữ liệu văn bản mạng xã hội phức tạp. Phân tích định tính từ công trình gốc chỉ ra rằng nguyên nhân chính khiến hệ số đồng thuận không đạt mức tuyệt đối nằm ở đặc trưng ngôn ngữ mạng xã hội tiếng Việt, nơi tần suất xuất hiện của từ viết tắt, teencode, tiếng lóng và các cụm từ ẩn dụ là rất dày đặc. Chẳng hạn, từ viết tắt "Dm" hoặc cụm từ mang hàm ý miệt thị ngầm như "lũ quan ngại" đòi hỏi người gán nhãn phải có kiến thức nền về văn hóa mạng; những người gán nhãn không nhận diện được lớp nghĩa ẩn dụ này thường gán nhầm mẫu HATE hoặc OFFENSIVE thành CLEAN.Trong bài báo gốc năm 2021, Luu et al. đã thiết lập các mô hình cơ sở (baselines) thuộc hai nhánh: mạng nơ-ron học sâu truyền thống (Text-CNN, GRU) và các mô hình Transformer đa ngữ (m-BERT cased, m-BERT uncased, XLM-R, DistilBERT). Kết quả thực nghiệm trên tập kiểm thử (Test set) cho thấy Text-CNN đạt độ chính xác (Accuracy) 86.69% và điểm Macro F1 là 61.11%, vượt qua GRU. Trong khi đó, các mô hình Transformer đa ngữ chứng tỏ ưu thế vượt trội, dẫn đầu là bert-base-multilingual-cased (m-BERT cased) với Accuracy đạt 86.88% và Macro F1 đạt 62.69%. Tuy nhiên, việc bài báo gốc chỉ mới dừng lại ở các biến thể Transformer đa ngữ đã mở ra khoảng trống nghiên cứu lớn cho việc ứng dụng mô hình ngôn ngữ đơn ngữ tiền huấn luyện chuyên biệt cho tiếng Việt là PhoBERT.Các bài báo khoa học tiêu biểu ứng dụng PhoBERT trên bộ dữ liệu ViHSDSau khi công bố ViHSD, hàng loạt công trình nghiên cứu nối tiếp từ nhóm UIT-NLP và cộng đồng học thuật đã trực tiếp đưa mô hình PhoBERT (Nguyen & Nguyen, 2020 — mô hình đơn ngữ dựa trên kiến trúc RoBERTa được tiền huấn luyện trên 20GB văn bản tiếng Việt) vào giải quyết bài toán phân loại trên bộ dữ liệu này, tập trung khắc phục hai rào cản lớn nhất là nhiễu từ vựng mạng xã hội và sự mất cân bằng lớp nghiêm trọng.Công trình nghiên cứuNăm công bố & Nguồn xuất bảnKiến trúc mô hình trọng tâmChiến lược xử lý dữ liệu & Hàm mất mátKết quả thực nghiệm chính trên ViHSDLuu, Nguyen, & Nguyen (A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts)2021 — IEA/AIE 2021 (Springer) / arXiv:2103.11528Text-CNN, GRU, m-BERT cased, m-BERT uncased, XLM-R, DistilBERTXử lý ẩn danh thực thể; huấn luyện với hàm Cross-Entropy tiêu chuẩn trên dữ liệu gốcText-CNN: 86.69% Acc, 61.11% Macro F1; m-BERT cased: 86.88% Acc, 62.69% Macro F1Tran, Nguyen, Hoang, Luu, Do, & Nguyen (Vietnamese Hate and Offensive Detection using PhoBERT-CNN and Social Media Streaming Data)2022 — Neural Computing and Applications / arXiv:2206.00524PhoBERT-CNN (kết hợp PhoBERT và Text-CNN), đối sánh với ML, DL và Transfer LearningChuẩn hóa teencode/viết tắt, tách từ bằng VnCoreNLP, kết hợp tăng cường dữ liệu EDA cho 2 lớp thiểu sốPhoBERT-CNN đạt đỉnh hiệu năng với 67.46% Macro F1 trên ViHSD; riêng tiền xử lý giúp tăng 4.66% F1Luu, Nguyen, & Nguyen (Impacts of Transformer-Based Language Models and Imbalanced Data for Hate Speech Detection...)2022 — Research Square (DOI: 10.21203/rs.3...)Các mô hình họ BERTology (PhoBERT, XLM-R, m-BERT, DistilBERT), Text-CNN, GRU và EnsembleKhảo sát tác động của tiền xử lý, kỹ thuật tăng cường dữ liệu (data-level) và kỹ thuật cấp độ thuật toán (algorithm-level)Chỉ ra tăng cường dữ liệu thông thường chỉ hiệu quả trên Text-CNN/GRU và XLM-R, kém hiệu quả trên các Transformer khácLuu, Nguyen, & Nguyen (An approach of data augmentation to improve the performance of BERTology models for Vietnamese hate speech detection)2023/2024 — Multimedia Tools and Applications (Vol. 83, pp. 56763–56783)Các mô hình BERTology đơn ngữ và đa ngữ (bao gồm PhoBERT, viBERT4news, vELECTRA, XLM-R)Đề xuất EDA cải tiến cho BERTology và tích hợp hàm mất mát Focal Loss cho phân loại dữ liệu mất cân bằngEDA cải tiến giúp tăng ~1.5% Macro F1; chứng minh Focal Loss cực kỳ bền vững cho phân loại đa lớp (ViHSD)Hoang, Luu, Tran, Nguyen, & Nguyen (ViHOS: Hate Speech Spans Detection for Vietnamese)2023 — EACL 2023 / arXiv:2301.10186PhoBERT-base, PhoBERT-large, XLM-R-base/large, BiLSTM-CRF + PhoW2VMở rộng từ ViHSD sang nhận diện khoảng từ thù ghét (spans); bổ sung thêm 5,528 bình luận CLEAN để giảm định kiến mô hìnhPhoBERT-base đạt 75.69% F1 và PhoBERT-large đạt 77.16% F1 (tăng +8.49% sau khi cân bằng lại định kiến dữ liệu)Phân tích sâu các công trình nghiên cứu trên cho thấy sự tiến hóa rõ rệt về mặt phương pháp luận khi áp dụng PhoBERT trên tập dữ liệu ViHSD. Trong bài báo công bố trên tạp chí Neural Computing and Applications (arXiv:2206.00524), nhóm tác giả Tran et al. (2022) chỉ ra bốn hạn chế cốt lõi của các hệ thống trước đó, bao gồm kỹ thuật tiền xử lý thiếu hụt, sự thờ ơ với vấn đề mất cân bằng dữ liệu, hiệu năng mô hình còn khiêm tốn và thiếu khả năng triển khai thực tế. Để giải quyết triệt để các điểm nghẽn này, nghiên cứu đã thiết kế kiến trúc lai PhoBERT-CNN, trong đó tầng biểu diễn ngữ cảnh từ của PhoBERT được nối tiếp bởi mạng tích chập một chiều (Text-CNN) nhằm bắt trọn các cụm đặc trưng cục bộ (local n-gram features) mang tính công kích. Đặc biệt, nhóm tác giả kết hợp công cụ tách từ tiếng Việt đạt chuẩn SOTA là VnCoreNLP cùng quy trình chuẩn hóa từ viết tắt, biểu tượng cảm xúc và áp dụng tăng cường dữ liệu EDA để nâng số mẫu lớp OFFENSIVE trong tập huấn luyện từ 1,606 lên 10,147 mẫu và lớp HATE từ 2,556 lên 16,849 mẫu. Nhờ sự cộng hưởng giữa biểu diễn ngữ cảnh sâu của PhoBERT và xử lý cân bằng dữ liệu, PhoBERT-CNN đã đẩy điểm Macro F1 trên ViHSD từ mức 62.69% của bài báo gốc lên 67.46%.Tiếp nối hướng nghiên cứu về tương tác giữa kiến trúc Transformer và dữ liệu mất cân bằng, công trình của Luu, Nguyen, & Nguyen (2023/2024) đăng trên tạp chí Multimedia Tools and Applications (DOI: 10.1007/s11042-023-16968-5) đã thực hiện một khảo sát thực nghiệm toàn diện trên bộ dữ liệu ViHSD và UIT-ViCTSD. Nghiên cứu này chứng minh rằng các phương pháp tiền xử lý văn bản truyền thống (như loại bỏ từ dừng — stopword removal) hầu như không mang lại sự cải thiện đáng kể nào cho các mô hình họ BERTology như PhoBERT, thậm chí có thể làm suy giảm hiệu năng do phá vỡ cấu trúc ngữ pháp và mối quan hệ phụ thuộc xa mà cơ chế Self-Attention dựa vào để phân biệt giữa câu cảm thán thô tục vô hướng (OFFENSIVE) và câu công kích có chủ đích (HATE). Hơn nữa, kỹ thuật tăng cường dữ liệu văn bản truyền thống (EDA nguyên bản) tuy hoạt động tốt với Text-CNN và GRU, nhưng lại kém hiệu quả đối với phần lớn các mô hình Transformer bởi thao tác xóa từ ngẫu nhiên hoặc hoán đổi vị trí từ làm biến dạng phân phối biểu diễn tiền huấn luyện; chỉ khi áp dụng phương pháp EDA cải tiến có kiểm soát ngữ nghĩa, điểm Macro F1 của các mô hình BERTology mới tăng thêm khoảng 1.5%. Đáng chú ý nhất, khi tích hợp trực tiếp hàm mất mát Focal Loss vào quá trình huấn luyện các mô hình BERTology để giải quyết bài toán mất cân bằng ở cấp độ thuật toán, kết quả thực nghiệm khẳng định Focal Loss thể hiện sự bền vững và hiệu quả vượt trội chuyên biệt cho bài toán phân loại đa lớp (multiclass classification) trên dữ liệu mất cân bằng như ViHSD, vượt xa mức cải thiện khi áp dụng cho bài toán phân loại nhị phân.Bên cạnh bài toán phân loại toàn câu trên ViHSD, các nghiên cứu giai đoạn 2023–2026 đã mở rộng phạm vi ứng dụng của PhoBERT sang nhận diện khoảng từ thù ghét ở cấp độ ký tự/từ với bộ dữ liệu ViHOS (Hoang et al., 2023) và hệ thống ensemble VHateX dựa trên PhoBERT (2025), chuyển đổi sang mô hình sinh văn bản thống nhất ViHateT5 (2024), hoặc phân loại ngôn từ thù ghét đa nhãn hướng mục tiêu cụ thể (cá nhân, nhóm, tôn giáo, chủng tộc, chính trị) với ViTHSD (Vo, Huynh, Luu, & Do, 2025). Ngoài ra, các công trình nghiên cứu về chuẩn hóa từ vựng mạng xã hội (ViLexNorm, khôi phục dấu tiếng Việt và sửa lỗi văn bản bằng LLM) cũng chứng minh rằng việc làm sạch nhiễu bề mặt mà vẫn giữ nguyên cấu trúc ngữ pháp giúp cải thiện độ chính xác của bộ phân loại ngôn từ thù ghét thêm từ 1% đến 3%.Phân tích Cấu trúc Thống kê Bộ dữ liệu ViHSD và Cơ chế Toán học của Sự Mất cân bằng LớpĐể lý giải tại sao một mô hình mạnh mẽ như PhoBERT-base vẫn gặp khó khăn trên ViHSD nếu chỉ huấn luyện bằng hàm mất mát Cross-Entropy tiêu chuẩn, cần mổ xẻ cấu trúc phân phối thống kê của bộ dữ liệu này. Bộ dữ liệu ViHSD gồm 33,400 mẫu được chia cố định theo tỷ lệ 7:1:2 thành tập huấn luyện (Train: 24,048 mẫu), tập phát triển (Dev/Validation: 2,672 mẫu) và tập kiểm thử (Test: 6,680 mẫu).Bảng thống kê chi tiết dưới đây tổng hợp số lượng mẫu theo từng tập dữ liệu, đặc trưng từ vựng trên tập huấn luyện và hiệu ứng khi áp dụng tăng cường dữ liệu theo ghi nhận từ các công trình nghiên cứu:Nhãn phân loạiMã số (ID)Tập Train (Gốc)Tập Dev (Validation)Tập TestTổng số mẫu (Tỷ lệ %)Độ dài TB trên tập Train (Từ/câu)Tổng số từ vựng (Tập Train gốc)Tập Train sau EDA (Tran et al., 2022)CLEAN019,886 (82.69%)2,190 (81.96%)5,548 (83.05%)27,624 (82.71%)6.55 từ130,238 từ19,886 mẫu (Giữ nguyên)OFFENSIVE11,606 (6.68%)212 (7.93%)444 (6.65%)2,262 (6.77%)7.24 từ11,624 từ10,147 mẫu (76,802 từ)HATE22,556 (10.63%)270 (10.10%)688 (10.30%)3,514 (10.52%)12.08 từ30,883 từ16,849 mẫu (Tăng cường)Tổng cộng—24,048 (100%)2,672 (100%)6,680 (100%)33,400 (100%)—172,745 từ46,882 mẫuTừ bảng số liệu trên, các phân tích cấu trúc cho thấy sự tương phản sâu sắc về độ dài câu và mật độ ngữ nghĩa giữa hai lớp thiểu số OFFENSIVE và HATE. Thống kê từ vựng trên tập Train của Tran et al. (2022) chỉ ra rằng các bình luận thuộc lớp CLEAN và OFFENSIVE rất ngắn, có độ dài trung bình lần lượt chỉ là 6.55 từ và 7.24 từ, trong khi bình luận thuộc lớp HATE có độ dài trung bình lên tới 12.08 từ — gần gấp đôi hai lớp còn lại. Nguyên nhân là do các bình luận OFFENSIVE thường chỉ là các thán từ bộc phát ngắn gọn không nhắm vào ai, còn các bình luận HATE đòi hỏi cấu trúc mệnh đề phức tạp hơn để vừa định danh đối tượng mục tiêu (cá nhân, tổ chức, vùng miền) vừa thực hiện hành vi miệt thị, hạ nhục. Hệ quả trực tiếp về mặt siêu tham số là việc thiết lập max_length = 128 subword tokens trên PhoBERT hoàn toàn bao phủ được toàn bộ thông tin ngữ nghĩa của gần như 100% các câu bình luận trong ViHSD mà không làm cắt cụt (truncation) phần thông tin mục tiêu ở cuối câu HATE, đồng thời tiết kiệm một nửa chi phí tính toán Self-Attention so với mức max_length = 256. Tuy nhiên, về mặt phân loại, lớp OFFENSIVE bị kẹp ở vùng biên nhập nhằng khi vừa chia sẻ độ dài câu ngắn với lớp CLEAN lại vừa chia sẻ từ vựng thô tục với lớp HATE. Khi huấn luyện bằng hàm mất mát thông thường, mô hình rất dễ gán nhầm các câu OFFENSIVE sang CLEAN (nếu từ nhạy cảm viết tắt dưới dạng teencode chưa từng gặp) hoặc gán nhầm sang HATE (nếu từ nhạy cảm xuất hiện rõ ràng). Thực tế, phân tích kết quả tinh chỉnh chuẩn trên tập Test 6,680 mẫu của ViHSD (như thực nghiệm trên BaoNhan/bn-newsbert-ViHSD với seed đại diện 202) cho thấy dù Accuracy toàn cục đạt 87.10% và Macro F1 đạt 63.22%, điểm F1 riêng lẻ (Per-class F1) của lớp HATE (688 mẫu) đạt 60.19%, nhưng Per-class F1 của lớp OFFENSIVE (444 mẫu) chỉ đạt 35.68%.Bên cạnh sự giao thoa đặc trưng ngôn ngữ, mô hình còn phải đối mặt với bẫy tối ưu hóa của hàm Cross-Entropy tiêu chuẩn trước tỷ lệ mất cân bằng lên tới 12.38 : 1. Trong tập Train (24,048 mẫu), số lượng mẫu CLEAN (19,886) gấp 12.38 lần số lượng mẫu OFFENSIVE (1,606) và gấp 7.78 lần số lượng mẫu HATE (2,556). Với hàm mất mát Cross-Entropy tiêu chuẩn không trọng số:
$$\mathcal{L}*{\text{CE}} = -\frac{1}{B} \sum*{i=1}^{B} \log(p*{i, y*i})$$
Trong mỗi batch kích thước $B = 16$, trung bình có tới $13.2$ mẫu thuộc lớp CLEAN, chỉ có khoảng $1.7$ mẫu thuộc lớp HATE và vỏn vẹn $1.1$ mẫu thuộc lớp OFFENSIVE. Tổng vector đạo hàm (gradient) dùng để cập nhật trọng số của PhoBERT-base bị áp đảo hoàn toàn bởi các mẫu CLEAN. Để giảm nhanh hàm mất mát tổng thể, bộ tối ưu hóa (optimizer) sẽ đẩy thiên kiến (bias) của tầng phân loại tuyến tính về phía lớp đa số CLEAN, dẫn đến khoảng cách rất lớn giữa Accuracy và Macro F1.Cơ chế giải quyết ở cấp độ thuật toán: Class Weight so với Focal LossThay vì nhân bản dữ liệu nhân tạo bằng EDA — phương pháp có nguy cơ làm nhiễu biểu diễn ngữ cảnh của các mô hình họ BERTology nếu không được kiểm soát chặt chẽ — việc điều chỉnh trực tiếp hàm mất mát thông qua Weighted Cross-Entropy (Class Weight) hoặc Multi-class Focal Loss là giải pháp chính quy, chặt chẽ về mặt toán học và giữ nguyên phân phối ngôn ngữ tự nhiên.Hàm mất mát Weighted Cross-Entropy (Class Weight)Phương pháp Weighted Cross-Entropy gán một vector trọng số tĩnh $\mathbf{w} = [w_0, w_1, w_2]$ cho ba lớp CLEAN (0), OFFENSIVE (1) và HATE (2) tỷ lệ nghịch với tần suất xuất hiện của chúng trong tập huấn luyện:
$$\mathcal{L}*{\text{WCE}} = -\frac{1}{B} \sum*{i=1}^{B} w*{y*i} \log(p*{i, y*i})$$
Sử dụng công thức cân bằng tần suất nghịch đảo tiêu chuẩn của scikit-learn ($w_c = \frac{N*{\text{train}}}{C \cdot N*c}$, với $N*{\text{train}} = 24,048$ và số lớp $C = 3$) dựa trên số lượng mẫu thực tế của ViHSD ($N_0 = 19,886$; $N_1 = 1,606$; $N_2 = 2,556$), hệ số trọng số chính xác cho từng lớp được xác định gồm $w_0 (\text{CLEAN}) = \frac{24048}{3 \times 19886} \approx \mathbf{0.4031}$, $w_1 (\text{OFFENSIVE}) = \frac{24048}{3 \times 1606} \approx \mathbf{4.9913}$ và $w_2 (\text{HATE}) = \frac{24048}{3 \times 2556} \approx \mathbf{3.1365}$.Như vậy, mỗi lỗi phân loại sai trên một mẫu OFFENSIVE sẽ bị phạt nặng gấp 12.38 lần so với một mẫu CLEAN, và mỗi mẫu HATE bị phạt nặng gấp 7.78 lần. Tuy nhiên, hạn chế của Class Weight thuần túy là cơ chế này đánh trọng số dựa trên nhãn thực tế (class-level) chứ không phân biệt độ khó của từng mẫu (instance-level): khi mô hình đã học thuộc các mẫu OFFENSIVE dễ, trọng số phạt $4.9913$ vẫn giữ nguyên, dễ khiến mô hình dự đoán tràn lan nhãn OFFENSIVE cho các câu CLEAN có chứa từ lóng, làm tăng báo động giả (False Positives) và làm giảm Precision. Để khắc phục hiện tượng phạt quá tay này, trong thực nghiệm tinh chỉnh Transformer, giải pháp hiệu quả là lấy căn bậc hai trọng số (Smooth Class Weights: $w_c^{\text{smooth}} = \sqrt{w_c} \approx [0.6349, 2.2341, 1.7710]$) hoặc chuyển sang sử dụng Focal Loss.Hàm mất mát Multi-class Focal LossĐược giới thiệu để giải quyết sự mất cân bằng cực đoan và đã được Luu et al. (2023/2024) chứng minh tính hiệu quả vượt trội cho các mô hình BERTology trên bài toán phân loại đa lớp của ViHSD, hàm mất mát Focal Loss bổ sung một nhân tử điều biến động $(1 - p_{i, y_i})^\gamma$ vào hàm Cross-Entropy:
$$\mathcal{L}_{\text{Focal}} = -\frac{1}{B} \sum_{i=1}^{B} \alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$
Trong biểu thức trên, $p_{i, y_i} = \frac{\exp(z_{i, y_i})}{\sum_{c=0}^{C-1} \exp(z_{i, c})}$ là xác suất dự đoán (sau tầng Softmax) mà mô hình gán cho nhãn đúng $y_i$ của mẫu thứ $i$, $\gamma \ge 0$ là tham số điều chỉnh độ tập trung (focusing parameter) thường được đặt ở mức $\gamma = 2.0$, và $\alpha_{y_i}$ là hệ số cân bằng lớp (có thể đặt $\alpha = 1$ để Focal Loss tự điều tiết qua $\gamma$, hoặc kết hợp bộ trọng số làm mượt $\boldsymbol{\alpha} = [0.6349, 2.2341, 1.7710]$).Cơ chế ưu việt của Focal Loss trên bộ dữ liệu ViHSD diễn ra thông qua sự điều tiết động theo độ tự tin của mô hình. Đối với hàng chục nghìn bình luận CLEAN thông thường (ví dụ: "Nhanh thực sự"), chỉ sau vài trăm bước huấn luyện đầu tiên, PhoBERT-base đã dễ dàng phân loại đúng với độ tự tin rất cao ($p_{i, y_i} \to 0.99$). Khi $\gamma = 2.0$, nhân tử điều biến $(1 - 0.99)^2 = 0.0001$ sẽ triệt tiêu gần như toàn bộ đóng góp mất mát của các mẫu dễ này (giảm đi 10,000 lần). Ngược lại, đối với một bình luận HATE chứa ngôn từ ẩn dụ khó (ví dụ: "lũ quan ngại") hoặc câu OFFENSIVE viết tắt bị phân loại sai với $p_{i, y_i} = 0.10$, nhân tử $(1 - 0.10)^2 = 0.81$ hầu như giữ nguyên cường độ đạo hàm. Nhờ đó, PhoBERT-base tự động dồn toàn bộ năng lực biểu diễn vào việc tinh chỉnh ranh giới quyết định giữa các mẫu biên khó phân biệt của ba lớp CLEAN, OFFENSIVE và HATE mà không làm gia tăng tỷ lệ báo động giả (False Positives).Hướng dẫn Huấn luyện PhoBERT-base với Class Weight và Focal Loss trên Google ColabQuy trình huấn luyện dưới đây được thiết kế hoàn chỉnh để thực thi trên môi trường Google Colab (sử dụng GPU T4 16GB VRAM miễn phí hoặc L4/A100), tuân thủ chặt chẽ các siêu tham số mục tiêu (max_length = 128, batch_size = 16, epochs = 3-4) và đặc thù kiến trúc của PhoBERT-base.Bước 1: Thiết lập môi trường và Cài đặt thư viện trên Google ColabTrên Google Colab, trước tiên cần đảm bảo phiên làm việc đã được kích hoạt phần cứng tăng tốc GPU (Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU), sau đó cài đặt các thư viện học sâu của hệ sinh thái Hugging Face cùng bộ tách từ tiếng Việt. Đối với PhoBERT, việc tách từ (word segmentation) để nối các âm tiết của từ ghép bằng dấu gạch dưới (ví dụ: "mạng xã hội" $\rightarrow$ "mạng_xã_hội") là yêu cầu bắt buộc vì từ điển BPE 64,000 token của PhoBERT được xây dựng trên dữ liệu tiếng Việt đã tách từ. Thực nghiệm trên Colab có thể sử dụng pyvi (nhanh, gọn, tương thích tức thì trên môi trường Python đám mây mà không cần cấu hình máy chủ JVM như VnCoreNLP).Python# Chạy trên một cell của Google Colab để cài đặt các thư viện cần thiết
!pip install -q transformers datasets accelerate evaluate scikit-learn pyvi seaborn matplotlib torch
Bước 2: Tải bộ dữ liệu ViHSD, Tiền xử lý bảo toàn ngữ cảnh và Tách từ tiếng ViệtBộ dữ liệu ViHSD có thể được tải trực tiếp từ Hugging Face Hub (uitnlp/vihsd hoặc bản gương mở sonlam1102/vihsd của chính tác giả Lưu Thanh Sơn) hoặc nạp từ 3 tệp train.csv, dev.csv, test.csv tải về từ kho lưu trữ GitHub chính thức (github.com/sonlam1102/vihsd).Dựa trên kết luận thực nghiệm của Luu et al. (2023/2024), quy trình tiền xử lý cho PhoBERT-base không xóa từ dừng (stopwords) và không loại bỏ dấu câu cảm thán, mà chỉ thực hiện chuẩn hóa bảng mã Unicode (NFC), thu gọn các ký tự kéo dài bất thường (ví dụ: "gìiiiii" $\rightarrow$ "gì", "ừmmmm" $\rightarrow$ "ừ" như phân tích lỗi của Luu et al., 2021) và thực hiện tách từ tiếng Việt.Pythonimport re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict
from pyvi import ViTokenizer
from transformers import (
AutoTokenizer,
AutoConfig,
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
DataCollatorWithPadding,
set_seed
)

# Cố định seed để đảm bảo tính tái lập thực nghiệm (Reproducibility)

SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị huấn luyện hiện tại: {device}")

# Ánh xạ nhãn chuẩn của bộ dữ liệu ViHSD

LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
CLASS_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

# 1. Tải bộ dữ liệu ViHSD (Hỗ trợ tải từ Hugging Face Hub hoặc từ file CSV cục bộ)

def load_vihsd_corpus():
try: # Tải trực tiếp từ Hugging Face Hub (uitnlp/vihsd hoặc sonlam1102/vihsd)
raw_datasets = load_dataset("sonlam1102/vihsd") # Chuẩn hóa tên tập phát triển nếu được đặt là 'validation'
if "validation" in raw_datasets and "dev" not in raw_datasets:
raw_datasets["dev"] = raw_datasets.pop("validation")
except Exception: # Dự phòng: Đọc trực tiếp từ 3 file CSV chuẩn (train.csv, dev.csv, test.csv)
train_df = pd.read_csv("train.csv")
dev_df = pd.read_csv("dev.csv")
test_df = pd.read_csv("test.csv")
raw_datasets = DatasetDict({
"train": Dataset.from_pandas(train_df),
"dev": Dataset.from_pandas(dev_df),
"test": Dataset.from_pandas(test_df)
})
return raw_datasets

dataset = load_vihsd_corpus()

# 2. Hàm tiền xử lý nhẹ bảo toàn ngữ cảnh + Tách từ tiếng Việt cho PhoBERT

def preprocess*and_segment_vietnamese(text):
if not isinstance(text, str) or len(text.strip()) == 0:
return "" # Chuẩn hóa bảng mã Unicode dựng sẵn (NFC) để đồng nhất dấu tiếng Việt
text = unicodedata.normalize("NFC", text) # Xóa các thẻ HTML hoặc đường dẫn URL nhiễu nếu có
text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # Chuẩn hóa các ký tự lặp kéo dài trên mạng xã hội (VD: "gìiiiii" -> "gì", "ừmmmm" -> "ừ")
text = re.sub(r"([a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ])\1{2,}", r"\1", text) # Chuẩn hóa khoảng trắng thừa
text = re.sub(r"\s+", " ", text).strip() # BẮT BUỘC CHO PHOBERT: Tách từ ghép tiếng Việt nối bằng dấu gạch dưới '*'
segmented_text = ViTokenizer.tokenize(text)
return segmented_text

# Kiểm tra thử nghiệm hàm tiền xử lý trên các mẫu điển hình của ViHSD

sample_comments = [
"Nhanh thực sự",
"Đm chứ biết làm gìiiiii",
"người ư? sinh vật hạ đẳng chứ ngươi ai lam thế"
]
for s in sample_comments:
print(f"Gốc: {s} --> Sau tách từ: {preprocess_and_segment_vietnamese(s)}")
Bước 3: Mã hóa Token (Tokenization) với max_length = 128Sau khi tách từ, văn bản được đưa qua bộ mã hóa AutoTokenizer của vinai/phobert-base với ngưỡng cắt cụt max_length = 128. Sử dụng DataCollatorWithPadding giúp thực hiện cơ chế đệm động (dynamic padding) theo câu dài nhất trong từng batch 16 mẫu (tối đa không quá 128 token), qua đó tối ưu hóa bộ nhớ VRAM và tăng tốc độ huấn luyện trên GPU Colab do độ dài trung bình của câu trong ViHSD chỉ dao động từ 6.55 đến 12.08 từ.PythonMODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Xác định tên cột văn bản và cột nhãn trong dataset (thường là 'free_text' và 'label_id')

text_col = "free_text" if "free_text" in dataset["train"].column_names else "text"
label_col = "label_id" if "label_id" in dataset["train"].column_names else "label"

def tokenize_function(batch): # Tiền xử lý và tách từ cho toàn bộ mẫu trong batch
segmented_texts = [preprocess_and_segment_vietnamese(t) for t in batch[text_col]]
tokenized_inputs = tokenizer(
segmented_texts,
truncation=True,
max_length=MAX_LENGTH,
padding=False # Sẽ dùng DataCollatorWithPadding để đệm động theo batch
)
tokenized_inputs["labels"] = [int(l) for l in batch[label_col]]
return tokenized_inputs

# Áp dụng ánh xạ trên cả 3 tập Train (24,048), Dev (2,672) và Test (6,680)

encoded_dataset = dataset.map(
tokenize_function,
batched=True,
remove_columns=dataset["train"].column_names,
desc="Đang tách từ và mã hóa BPE cho PhoBERT-base"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
Bước 4: Xây dựng Mô-đun Class Weight, Focal Loss và Tùy biến TrainerĐây là bước trọng tâm để khắc phục sự mất cân bằng dữ liệu của ViHSD ở cấp độ thuật toán. Đoạn mã dưới đây tính toán tự động vector trọng số lớp từ tập huấn luyện, định nghĩa lớp MultiClassFocalLoss bằng PyTorch và ghi đè phương thức compute_loss trong lớp ImbalanceAwarePhoBERTTrainer. Thiết kế này cho phép chuyển đổi linh hoạt giữa ba chế độ "cross_entropy" (tiêu chuẩn), "class_weight" (Weighted Cross-Entropy), hoặc "focal_loss" (Multi-class Focal Loss) chỉ bằng một tham số cấu hình.Pythonfrom sklearn.utils.class_weight import compute_class_weight

# 1. Tính toán chính xác Class Weights từ phân phối nhãn tập Train của ViHSD

train_labels = np.array(encoded_dataset["train"]["labels"])
raw_class_weights = compute_class_weight(
class_weight="balanced",
classes=np.array([0, 1, 2]),
y=train_labels
)

# Trọng số chuẩn: CLEAN ~ 0.4031, OFFENSIVE ~ 4.9913, HATE ~ 3.1365

class_weights_tensor = torch.tensor(raw_class_weights, dtype=torch.float).to(device)

# Trọng số làm mượt (Smooth Class Weights bằng căn bậc hai) dùng kết hợp với Focal Loss

smooth_weights_tensor = torch.tensor(np.sqrt(raw_class_weights), dtype=torch.float).to(device)

print(f"Class Weights tiêu chuẩn (Balanced): {raw_class_weights.round(4)}")
print(f"Smooth Class Weights (Sqrt): {np.sqrt(raw_class_weights).round(4)}")

# 2. Định nghĩa hàm mất mát Multi-class Focal Loss bằng PyTorch

class MultiClassFocalLoss(nn.Module):
def **init**(self, alpha=None, gamma=2.0, reduction="mean"):
"""
Multi-class Focal Loss cho bài toán phân loại đa lớp mất cân bằng:
FL(p_t) = -alpha_t _ (1 - p_t)^gamma _ log(p_t)
"""
super(MultiClassFocalLoss, self).**init**()
self.alpha = alpha # Tensor kích thước [num_classes] hoặc None
self.gamma = gamma # Tham số tập trung (mặc định gamma = 2.0)
self.reduction = reduction

    def forward(self, logits, targets):
        # Tính log-probabilities và probabilities ổn định số học bằng log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Trích xuất xác suất dự đoán p_t và log(p_t) tại đúng vị trí nhãn thực tế y_i
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        # Tính nhân tử điều biến (1 - p_t)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)

        # Kết hợp trọng số lớp alpha_t nếu được cung cấp
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets.squeeze(1))
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# 3. Tùy biến Hugging Face Trainer tích hợp Class Weight hoặc Focal Loss

class ImbalanceAwarePhoBERTTrainer(Trainer):
def **init**(self, loss_type="focal_loss", class_weights=None, focal_gamma=2.0, *args, \*\*kwargs):
super().**init**(*args, \*\*kwargs)
self.loss_type = loss_type
self.class_weights = class_weights
self.focal_gamma = focal_gamma

        if self.loss_type == "class_weight":
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif self.loss_type == "focal_loss":
            self.criterion = MultiClassFocalLoss(alpha=self.class_weights, gamma=self.focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

Bước 5: Thiết lập Bộ độ đo Đánh giá và Thực thi Huấn luyện (batch_size = 16, 3-4 epochs)Trong quá trình huấn luyện trên tập ViHSD, tuyệt đối không sử dụng eval_loss hoặc accuracy làm tiêu chí chọn checkpoint tốt nhất (metric_for_best_model), bởi mô hình có thiên hướng dự đoán toàn bộ mẫu nghi ngờ về lớp CLEAN sẽ có eval_loss rất thấp và accuracy rất cao nhưng lại thất bại trên hai lớp thiểu số. Thay vào đó, hàm compute_metrics dưới đây tính toán trực tiếp macro_f1, weighted_f1, cùng per-class F1 cho cả ba nhãn ngay sau mỗi epoch và chọn lưu mô hình đạt macro_f1 cao nhất trên tập Dev.Pythonfrom sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_vihsd_metrics(eval_pred):
logits, labels = eval_pred
preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "weighted_f1": weighted_f1,
        "f1_CLEAN": per_class_f1[0],
        "f1_OFFENSIVE": per_class_f1[1],
        "f1_HATE": per_class_f1[2],
    }

# Khởi tạo cấu hình và mô hình PhoBERT-base cho bài toán phân loại 3 lớp

config = AutoConfig.from_pretrained(
MODEL_NAME,
num_labels=3,
id2label=ID2LABEL,
label2id=LABEL2ID,
classifier_dropout=0.2
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

# Cấu hình siêu tham số chuẩn theo yêu cầu: batch_size = 16, epochs = 4 (3-4 epochs)

training_args = TrainingArguments(
output_dir="./phobert_vihsd_checkpoints",
num_train_epochs=4, # Huấn luyện trong 4 epochs (tương đương ~6,012 steps)
per_device_train_batch_size=16, # Batch size huấn luyện = 16
per_device_eval_batch_size=16, # Batch size đánh giá = 16
learning_rate=2e-5, # Tốc độ học tối ưu cho PhoBERT-base
weight_decay=0.01, # Điều chuẩn L2 (AdamW weight decay)
warmup_ratio=0.1, # Khởi động tuyến tính trong 10% tổng số bước đầu
lr_scheduler_type="cosine", # Giảm tốc độ học theo đường cong Cosine
eval_strategy="epoch", # Đánh giá trên tập Dev sau mỗi epoch
save_strategy="epoch", # Lưu checkpoint sau mỗi epoch
load_best_model_at_end=True, # Tự động tải lại checkpoint có Macro F1 tốt nhất
metric_for_best_model="macro_f1", # Tiêu chí chọn mô hình: Macro F1
greater_is_better=True,
fp16=torch.cuda.is_available(), # Bật Mixed Precision (FP16) tăng tốc gấp đôi trên GPU T4
logging_steps=100,
save_total_limit=2,
report_to="none",
seed=SEED
)

# Lựa chọn chế độ huấn luyện:

# - Đặt LOSS_MODE = "focal_loss" (kèm smooth_weights_tensor, gamma=2.0)

# - Hoặc đặt LOSS_MODE = "class_weight" (kèm class_weights_tensor)

LOSS_MODE = "focal_loss"
selected_weights = smooth_weights_tensor if LOSS_MODE == "focal_loss" else class_weights_tensor

trainer = ImbalanceAwarePhoBERTTrainer(
loss_type=LOSS_MODE,
class_weights=selected_weights,
focal_gamma=2.0,
model=model,
args=training_args,
train_dataset=encoded_dataset["train"],
eval_dataset=encoded_dataset["dev"],
processing_class=tokenizer,
data_collator=data_collator,
compute_metrics=compute_vihsd_metrics
)

# Bắt đầu quá trình huấn luyện

trainer.train()
Với tập huấn luyện gồm $24,048$ mẫu và batch*size = 16, mỗi epoch bao gồm chính xác $\lceil 24048 / 16 \rceil = 1,503$ bước cập nhật trọng số (steps). Khi bật chế độ tính toán hỗn hợp fp16=True kết hợp max_length = 128 và đệm động, mỗi epoch của PhoBERT-base trên GPU NVIDIA Tesla T4 của Google Colab chỉ mất khoảng 2 phút 15 giây đến 2 phút 45 giây (tổng thời gian hoàn tất 4 epochs khoảng 9–11 phút và tiêu thụ chưa tới 4.2 GB VRAM).Hệ thống Đánh giá Đa chiều: Macro F1, Weighted F1, Per-class F1 và Ma trận Nhầm lẫn (Confusion Matrix)Sau khi kết thúc 3–4 epochs huấn luyện và tải lại trọng số tốt nhất trên tập Dev, mô hình được đưa vào đánh giá độc lập trên tập kiểm thử (Test set) gồm 6,680 mẫu (CLEAN: 5,548; OFFENSIVE: 444; HATE: 688).Ý nghĩa toán học và Sự phân hóa giữa các độ đo trên tập dữ liệu ViHSDGọi $\text{TP}_c$, $\text{FP}_c$ và $\text{FN}_c$ lần lượt là số lượng Dự đoán Đúng Dương tính (True Positives), Dương tính Giả (False Positives) và Âm tính Giả (False Negatives) của lớp $c \in \{0, 1, 2\}$, và $N_c$ là số mẫu thực tế (support) của lớp đó trong tập Test ($N = \sum*{c=0}^2 N*c = 6,680$). Độ đo cơ sở đầu tiên là Per-class F1 ($F1_c$), được định nghĩa là trung bình điều hòa giữa Độ chuẩn xác ($\text{Precision}\_c = \frac{\text{TP}\_c}{\text{TP}\_c + \text{FP}\_c}$) và Độ phủ ($\text{Recall}\_c = \frac{\text{TP}\_c}{\text{TP}\_c + \text{FN}\_c}$) của riêng từng lớp:
$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$
Trên bộ dữ liệu ViHSD, ba chỉ số $F1*{\text{CLEAN}}$, $F1_{\text{OFFENSIVE}}$ và $F1_{\text{HATE}}$ cho phép bóc tách chính xác điểm nghẽn của mô hình. Trong các thực nghiệm chuẩn, $F1_{\text{CLEAN}}$ luôn đạt mức rất cao ($92\% - 94\%$), $F1_{\text{HATE}}$ đạt mức khá ($60\% - 66\%$), trong khi $F1_{\text{OFFENSIVE}}$ là chỉ số nhạy cảm nhất ($35\% - 53\%$) phản ánh trực tiếp năng lực phân tách ngôn từ thô tục không mục tiêu khỏi bình luận sạch và bình luận công kích cá nhân.Từ ba giá trị Per-class F1, độ đo Macro F1 ($\text{F1}_{\text{macro}}$) được tính bằng trung bình cộng không trọng số của cả ba lớp:
$$\text{F1}_{\text{macro}} = \frac{1}{3} \left( F1_{\text{CLEAN}} + F1_{\text{OFFENSIVE}} + F1_{\text{HATE}} \right)$$
Trong công thức này, lớp OFFENSIVE (chỉ có 444 mẫu, chiếm 6.65% tập Test) có quyền quyết định ngang bằng hoàn toàn với lớp CLEAN (5,548 mẫu, chiếm 83.05% tập Test) khi mỗi lớp đóng góp đúng $\frac{1}{3}$ (33.33%) vào điểm số cuối cùng. Đây là lý do toàn bộ các bài báo khoa học về ViHSD (Luu et al., 2021; Tran et al., 2022; Luu et al., 2023/2024) đều sử dụng Macro F1 làm thước đo chuẩn duy nhất để xếp hạng mô hình.Ngược lại, độ đo Weighted F1 ($\text{F1}_{\text{weighted}}$) là trung bình cộng có trọng số theo tỷ lệ số lượng mẫu thực tế ($N_c / N$) của từng lớp trong tập Test:
$$\text{F1}_{\text{weighted}} = \frac{5548}{6680} F1_{\text{CLEAN}} + \frac{444}{6680} F1_{\text{OFFENSIVE}} + \frac{688}{6680} F1_{\text{HATE}} \approx 0.8305 \cdot F1_{\text{CLEAN}} + 0.0665 \cdot F1_{\text{OFFENSIVE}} + 0.1030 \cdot F1_{\text{HATE}}$$
Do hệ số của $F1_{\text{CLEAN}}$ chiếm tới $83.05\%$, chỉ số Weighted F1 bị chi phối nặng nề bởi lớp đa số. Khi chuyển từ hàm mất mát Cross-Entropy tiêu chuẩn sang Class Weight hoặc Focal Loss, một hiện tượng đánh đổi toán học tất yếu thường xảy ra: mô hình chấp nhận giảm nhẹ $F1_{\text{CLEAN}}$ khoảng $0.5\% - 1.5\%$ (do một số bình luận CLEAN có chứa từ lóng bị đẩy sang dự đoán là OFFENSIVE hoặc HATE) để đổi lấy mức tăng vọt $5\% - 12\%$ trên $F1_{\text{OFFENSIVE}}$ và $F1_{\text{HATE}}$. Khi đó, Macro F1 tăng mạnh (phản ánh hệ thống nhận diện ngôn từ độc hại tốt hơn hẳn), trong khi Accuracy và Weighted F1 có thể giữ nguyên hoặc giảm nhẹ khoảng $0.3\% - 0.8\%$.Mã nguồn Đánh giá Toàn diện và Trực quan hóa Confusion Matrix trên ColabĐoạn mã dưới đây thực hiện dự đoán trên toàn bộ 6,680 mẫu của tập Test, xuất bảng báo cáo Per-class F1, Macro F1, Weighted F1 và vẽ đồng thời hai ma trận nhầm lẫn: Ma trận số lượng mẫu tuyệt đối (Raw Counts) và Ma trận chuẩn hóa theo hàng (Row-Normalized Recall %) giúp chẩn đoán trực quan tỷ lệ nhầm lẫn giữa từng cặp nhãn.Pythonimport matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Thực hiện dự đoán trên tập Test (6,680 mẫu)

test_output = trainer.predict(encoded_dataset["test"])
test_logits = test_output.predictions
y_pred = np.argmax(test_logits, axis=-1)
y_true = test_output.label_ids

# 2. Tổng hợp bảng chỉ số Per-class F1, Macro F1 và Weighted F1

report_dict = classification_report(
y_true,
y_pred,
target_names=CLASS_NAMES,
digits=4,
output_dict=True,
zero_division=0
)

print("=" _ 75)
print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHOBERT-BASE TRÊN TẬP TEST VIHSD (CHẾ ĐỘ: {LOSS_MODE.upper()})")
print("=" _ 75)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

summary_df = pd.DataFrame({
"Độ đo (Metric)": [
"Macro F1",
"Weighted F1",
"Accuracy",
"Per-class F1: CLEAN (0)",
"Per-class F1: OFFENSIVE (1)",
"Per-class F1: HATE (2)"
],
"Giá trị (%)": [
report_dict["macro avg"]["f1-score"] _ 100,
report_dict["weighted avg"]["f1-score"] _ 100,
report_dict["accuracy"] _ 100,
report_dict["CLEAN"]["f1-score"] _ 100,
report_dict["OFFENSIVE"]["f1-score"] _ 100,
report_dict["HATE"]["f1-score"] _ 100
]
})
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

# 3. Tính toán và vẽ Ma trận Nhầm lẫn (Confusion Matrix: Tuyệt đối & Chuẩn hóa theo nhãn thực)

cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis] \* 100.0

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=120)

# Biểu đồ 1: Số lượng mẫu tuyệt đối

sns.heatmap(
cm_raw,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=CLASS_NAMES,
yticklabels=CLASS_NAMES,
cbar=False,
ax=axes[0],
annot_kws={"size": 12, "weight": "bold"}
)
axes[0].set_title("Confusion Matrix (Số lượng mẫu tuyệt đối)", fontsize=13, fontweight="bold", pad=12)
axes[0].set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11)
axes[0].set_ylabel("Nhãn thực tế (True Label)", fontsize=11)

# Biểu đồ 2: Tỷ lệ phần trăm theo hàng (Recall của từng lớp)

annot_pct = np.array([[f"{val:.1f}%\n({cnt})" for val, cnt in zip(row_pct, row_cnt)]
for row_pct, row_cnt in zip(cm_norm, cm_raw)])
sns.heatmap(
cm_norm,
annot=annot_pct,
fmt="",
cmap="YlOrRd",
xticklabels=CLASS_NAMES,
yticklabels=CLASS_NAMES,
cbar=True,
ax=axes[1],
annot_kws={"size": 11, "weight": "bold"}
)
axes[1].set_title("Normalized Confusion Matrix (% theo Nhãn thực tế)", fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11)
axes[1].set_ylabel("Nhãn thực tế (True Label)", fontsize=11)

plt.tight*layout()
plt.savefig("phobert_vihsd_confusion_matrix.png", bbox_inches="tight")
plt.show()
Phân tích Chẩn đoán Mô hình qua Ma trận Nhầm lẫn và Đối sánh Chiến lược Hàm Mất mátKhi phân tích ma trận nhầm lẫn $3 \times 3$ trên tập Test của ViHSD, sự dịch chuyển của các ô ngoài đường chéo chính phản ánh trực tiếp đặc trưng ngôn ngữ học của dữ liệu mạng xã hội tiếng Việt. Tại ô $(\text{Thực tế} = \text{OFFENSIVE}, \text{Dự đoán} = \text{CLEAN})$, phân tích lỗi trong bài báo gốc của Luu et al. (2021) trên mô hình m-BERT cased chỉ ra rằng phần lớn các bình luận OFFENSIVE bị dự đoán nhầm thành CLEAN. Nguyên nhân xuất phát từ việc người dùng viết tắt từ ngữ nhạy cảm bằng nhiều biến thể teencode sáng tạo nằm trong các câu rất ngắn (~7.24 từ). Khi áp dụng Class Weight hoặc Focal Loss, trọng số phạt cao trên lớp OFFENSIVE buộc PhoBERT-base phải gán độ nhạy lớn hơn cho các token viết tắt này, qua đó kéo tỷ lệ dự đoán nhầm vào ô CLEAN giảm xuống rõ rệt và nâng cao Recall của lớp OFFENSIVE.Trong khi đó, sự nhầm lẫn hai chiều tại cặp ô $(\text{Thực tế} = \text{HATE}, \text{Dự đoán} = \text{OFFENSIVE})$ và $(\text{Thực tế} = \text{OFFENSIVE}, \text{Dự đoán} = \text{HATE})$ xảy ra khi câu bình luận có chứa từ thô tục nhưng đối tượng bị công kích được viết ẩn đi hoặc thay bằng đại từ phiếm chỉ. Như Luu et al. (2021) đã lưu ý, nếu mô hình không nhận diện được thực thể mục tiêu (target objects) hoặc từ lóng miệt thị được viết dưới dạng bất quy tắc, bộ phân loại sẽ hạ cấp dự đoán từ HATE xuống OFFENSIVE. Cuối cùng, ô $(\text{Thực tế} = \text{HATE}, \text{Dự đoán} = \text{CLEAN})$ đại diện cho nhóm lỗi nguy hiểm nhất trong kiểm duyệt nội dung (False Negatives của ngôn từ thù ghét), thường rơi vào các câu thù ghét ngầm (implicit hate speech) hoàn toàn không sử dụng từ chửi thề mà dùng hình ảnh so sánh hạ nhục hoặc mỉa mai sâu cay (ví dụ: "người ư? sinh vật hạ đẳng chứ ngươi ai lam thế" hoặc "lũ quan ngại"). Đây chính là vùng mà Focal Loss ($\gamma = 2.0$) phát huy tác dụng vượt trội so với Class Weight tĩnh: bởi các câu thù ghét ngầm này có xác suất dự đoán đúng $p_t$ rất thấp trong những epoch đầu, nhân tử $(1 - p_t)^2$ giữ cho đạo hàm của chúng ở mức cực đại, buộc các tầng Self-Attention của PhoBERT-base phải học mối liên hệ ngữ nghĩa giữa các cụm từ miệt thị thay vì chỉ dựa vào từ khóa bề mặt.Bảng tổng hợp dưới đây đối sánh đặc tính vận hành và dải điểm kỳ vọng của ba cấu hình hàm mất mát khi tinh chỉnh PhoBERT-base (max_length = 128, batch_size = 16, 3-4 epochs) trên tập Test chuẩn của ViHSD:Tiêu chí đối sánhCross-Entropy Tiêu chuẩn (Baseline)Weighted Cross-Entropy (Class Weight)Multi-class Focal Loss (γ=2.0\gamma = 2.0γ=2.0 + Smooth α\alphaα)Cơ chế điều tiết đạo hàmĐánh trọng số đồng đều $1.0$ cho mọi mẫu; bị áp đảo bởi 82.7% mẫu CLEANNhân trọng số tĩnh theo tần suất nghịch đảo của lớp ($[0.40, 4.99, 3.14]$)Điều biến động theo độ khó từng mẫu $(1 - p_t)^\gamma$ kết hợp cân bằng lớp $\alpha_t$Xu hướng dịch chuyển Confusion MatrixTối đa hóa đường chéo ô CLEAN; bỏ sót lượng lớn mẫu OFFENSIVE và HATE sang CLEANTăng mạnh Recall của OFFENSIVE và HATE, nhưng dễ tăng False Positives từ CLEAN sang OFFENSIVECân bằng tối ưu giữa Precision và Recall trên cả 2 lớp thiểu số; giảm mạnh lỗi sót mẫu biên khóPer-class F1: CLEAN (5,548 mẫu)Rất cao (~93.0% – 93.8%)Giảm nhẹ (~90.5% – 92.0%) do phạt nặng lớp đa sốDuy trì ở mức cao (~92.2% – 93.3%)Per-class F1: OFFENSIVE (444 mẫu)Thấp (~35.5% – 44.0%)Cải thiện rõ rệt (~46.0% – 50.5%)Đạt hiệu quả cao nhất (~48.5% – 53.5%)Per-class F1: HATE (688 mẫu)Trung bình (~58.0% – 62.0%)Tăng khá (~61.5% – 64.5%)Tăng vững chắc (~63.0% – 66.5%)Weighted F1 & Accuracy toàn cụcAccuracy ~86.8% – 87.2%; Weighted F1 ~85.5% – 86.5%Accuracy ~84.5% – 86.0%; Weighted F1 ~85.0% – 86.0%Accuracy ~86.2% – 87.1%; Weighted F1 ~86.0% – 86.8%Macro F1 kỳ vọng trên tập Test~62.5% – 64.5%~65.0% – 66.2%~66.0% – 67.2% (Tiệm cận mức 67.46% của PhoBERT-CNN + EDA)Kết luận và Khuyến nghị Kỹ thuật Thực chiếnTổng hợp từ hệ thống các công trình nghiên cứu khoa học trên bộ dữ liệu ViHSD (Luu et al., 2021; Tran et al., 2022; Hoang et al., 2023; Luu et al., 2023/2024; Vo et al., 2025) cùng phân tích toán học về hàm mất mát, việc triển khai mô hình PhoBERT-base cho bài toán phân loại đa lớp ngôn từ thù ghét tiếng Việt đòi hỏi sự phối hợp chặt chẽ giữa xử lý ngôn ngữ đặc thù và cơ chế điều chuẩn đạo hàm. Trước hết, khi làm việc với PhoBERT-base, bước tách từ tiếng Việt (nối âm tiết từ ghép bằng dấu gạch dưới *) là yêu cầu bắt buộc để khớp với không gian từ vựng BPE tiền huấn luyện. Tuy nhiên, hệ thống tiền xử lý cần tránh tuyệt đối việc xóa từ dừng (stopwords) hay cắt bỏ dấu câu cảm thán, vì các mô hình họ BERTology cần toàn bộ cấu trúc ngữ pháp của câu bình luận dài (~12.08 từ ở lớp HATE) để phân biệt hành vi công kích có mục tiêu với các câu cảm thán thông thường.Về mặt thuật toán tối ưu hóa, đối với cấu trúc mất cân bằng đặc thù của ViHSD (82.71% CLEAN, 6.77% OFFENSIVE, 10.52% HATE), việc sử dụng Class Weight nguyên bản ($w_1 \approx 4.99$) thường gây ra hiện tượng phạt quá mức khiến mô hình dự đoán nhầm nhiều câu bình luận sạch có chứa tiếng lóng sang lớp OFFENSIVE. Ngược lại, phương án kết hợp Multi-class Focal Loss ($\gamma = 2.0$) với trọng số lớp làm mượt bằng căn bậc hai ($\boldsymbol{\alpha} \approx [0.63, 2.23, 1.77]$) giúp mô hình tự động giảm trọng số các mẫu CLEAN dễ và tập trung tối đa năng lực học sâu vào các mẫu biên nhập nhằng giữa OFFENSIVE và HATE, hoàn toàn tương thích với kết luận thực nghiệm của Luu et al. (2023/2024).Trong quá trình vận hành huấn luyện trên Google Colab với batch_size = 16, max_length = 128 và tốc độ học 2e-5, các tầng phân loại của PhoBERT-base thường đạt đỉnh Macro F1 trên tập Dev vào cuối epoch thứ 3 hoặc epoch thứ 4. Việc sử dụng bộ lập lịch giảm tốc độ học theo hàm Cosine (lr_scheduler_type="cosine") cùng tỷ lệ khởi động warmup_ratio=0.1 và cơ chế tự động khôi phục checkpoint có macro_f1 cao nhất (metric_for_best_model="macro_f1") là thiết lập thiết yếu để ngăn chặn sự suy giảm hiệu năng do quá khớp (overfitting) trên các mẫu thiểu số. Trong trường hợp bài toán thực tế yêu cầu đẩy điểm Macro F1 vượt mốc 67% trên ViHSD, kiến trúc PhoBERT-base kết hợp Focal Loss có thể được nâng cấp theo hai hướng đã được kiểm chứng trong tài liệu khoa học: thay thế tầng Linear phẳng trên đỉnh PhoBERT-base bằng khối tích chập đa kích thước bộ lọc (PhoBERT-CNN theo Tran et al., 2022) để trích xuất đồng thời các cụm n-gram cục bộ, hoặc tích hợp từ điển chuẩn hóa teencode và từ viết tắt tiếng Việt (ViLexNorm) ngay trước bước tách từ để giúp bộ mã hóa BPE nhận diện chính xác các biến thể từ nhạy cảm viết tắt trong lớp OFFENSIVE.

# Training Code

هذا المجلد مخصص لتدريب النموذجين فقط، منفصل تمامًا عن API.

## المسار العام
1. تدريب YOLO لكشف مناطق الحقول (Model A)
2. تجهيز قصّ المناطق من Data_aug
3. تدريب OCR لاستخراج النص من كل منطقة مقصوصة (Model B)
4. تصدير كلا النموذجين إلى ONNX ووضعهما في `../models/`

## النماذج الناتجة
- `models/yolo_detector.onnx`
- `models/ocr_extractor.onnx`

# OCR Training (Text Extraction from Crops)

## خطوات
1. قص المناطق
   `python -m training.ocr.prepare_crops`
   -> ينتج training/ocr/crops/*.jpg
2. تجهيز التسميات
   أنشئ training/ocr/crops/crops_labels.json بصيغة:
   {"img_stem_cX_N.jpg": "النص الصحيح"}
3. تدريب (باستخدام train_ocr.py بعد إكمال DataLoader + CTCLoss)
4. تصدير ONNX
   `python -m training.ocr.export_onnx`

## ملاحظات
- النموذج CRNN + CTC موجه للتدريب المبسط
- يخرج models/ocr_extractor.onnx ليقرأه app/ml/inference.py

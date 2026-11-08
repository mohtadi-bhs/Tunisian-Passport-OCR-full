# YOLO Training (Detection/Localization)

## خطوات
1. تجهيز البيانات
   `python -m training.yolo.prepare_split`
2. تدريب
   `python -m training.yolo.train_yolo`  (يمكن تغيير weights/epochs/batch)
3. تصدير ONNX
   `python -m training.yolo.export_onnx`

## ملاحظات
- يستخدم Data_aug/images و Data_aug/labels بصيغة YOLO polygon/normalized
- يخرج models/yolo_detector.onnx

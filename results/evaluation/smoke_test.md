# Smoke test — โมเดลจริงกับภาพตัวอย่าง 6 ภาพ

> [!warning] นี่ไม่ใช่ผล precision/recall/mAP
> ภาพทั้ง 6 ภาพไม่มีป้ายกำกับ ground truth จึงใช้ตรวจว่า pipeline ทำงานได้
> และดูแนวโน้มผลกระทบของการปรับภาพเท่านั้น ตัวเลขสรุปผลสุดท้ายต้องรันกับ
> validation/test ที่มีป้ายกำกับ

## เงื่อนไข

- โมเดล: `best.pt` (YOLO11n, 2 คลาส: `With Helmet`, `Without Helmet`)
- ภาพ: `images/testModel1.jpg` – `testModel6.jpg`
- Confidence: `0.25`
- ค่า `imgsz`: ค่าเริ่มต้นของ Ultralytics (640)
- เปรียบเทียบ: ภาพเดิม / CLAHE / Median
- ผลดิบ: `results/prediction_comparison/prediction_smoke_test.csv`
- ภาพเทียบ: `results/prediction_comparison/*_detections.jpg`

## จำนวนกรอบที่ตรวจพบ

| ภาพ | ภาพเดิม | CLAHE | Median | หมายเหตุ |
|---|---:|---:|---:|---|
| testModel1 | 2 | 2 | 2 | คงที่ |
| testModel2 | 5 | 5 | 4 | CLAHE เปลี่ยนสัดส่วนคลาส; Median ลบกรอบ `Without Helmet` ออก |
| testModel3 | 2 | 2 | 2 | คงที่ |
| testModel4 | 3 | 3 | 3 | คงที่ |
| testModel5 | 4 | 4 | 4 | คงที่ |
| testModel6 | 12 | 11 | 11 | CLAHE เพิ่ม `Without Helmet` 1 กรอบ; Median ลด `With Helmet` 1 กรอบ |
| **รวม** | **28** | **27** | **26** | ยังสรุปความแม่นยำไม่ได้ |

## ตัวอย่างที่ดูด้วยตาแล้วถูก

- `testModel1` — คนขับและคนซ้อนไม่สวมหมวก โมเดลตรวจ `Without Helmet` 2 กรอบทั้งก่อนและหลัง CLAHE
- `testModel3` — ผู้ขับและคนซ้อนสวมหมวก โมเดลตรวจ `With Helmet` 2 กรอบทั้งก่อนและหลัง CLAHE

## ตัวอย่างที่ต้องตรวจกับป้ายจริง

- `testModel2` — หลัง CLAHE มีกรอบซ้อนกันบนคนคนเดียวกันและให้คลาสขัดกัน:
  `With Helmet 0.281` กับ `Without Helmet 0.273` ในตำแหน่งใกล้กัน
  ส่วน Median เหลือ `With Helmet` 4 กรอบและไม่มี `Without Helmet`
  ต้องดูว่าคนในภาพสวมหมวกจริงหรือไม่จึงจะบอกได้ว่าวิธีไหนดีขึ้น
- `testModel6` — ภาพฝูงชน ภาพเดิมตรวจ `With Helmet` 12 กรอบ
  หลัง CLAHE เหลือ `With Helmet` 10 + `Without Helmet` 1
  ต้องตรวจว่ากรอบ `Without Helmet` ที่เพิ่มขึ้นเป็นของจริงหรือ false positive

## ข้อจำกัด

- ไม่มีป้ายกำกับ จึงคำนวณ precision, recall หรือ mAP จากชุดนี้ไม่ได้
- ภาพเพียง 6 ภาพไม่ตัวแทนชุด validation/test
- ค่า confidence 0.25 อาจทำให้เห็นกรอบที่มีคะแนนต่ำและเกิดกรอบซ้อน
- ตัวเลขผล train จาก checkpoint (`precision 0.8946`, `recall 0.8251`,
  `mAP50 0.8986`, `mAP50-95 0.5452`) เป็นค่าที่โมเดลรายงานระหว่าง train
  ไม่ใช่ผลการทดลอง preprocessing รอบนี้

## ขั้นต่อไป

1. ขอ `data.yaml` + ชุด validation/test พร้อมป้ายจากคนที่ 2
2. รัน `src/evaluate_preprocessing.py` กับ validation เพื่อเลือกวิธี
3. รัน test ครั้งเดียวเพื่อรายงานผลสุดท้าย
4. เก็บตัวอย่างตรวจถูก/พลาดจากชุดที่มีป้ายจริง

# ผลการทดลอง — ปรับภาพและวัดผล

โฟลเดอร์นี้เก็บผลของคนที่ 4 ห้ามใส่ตัวเลขที่ยังไม่ได้รันจริง

## โครงสร้าง

```text
results/
├── preprocessing/                 # ภาพก่อน/หลังสำหรับรายงาน
│   └── ..._<method>_compare.jpg
└── evaluation/
    ├── preprocessing_comparison.csv   # ประวัติการรัน (สะสม)
    ├── preprocessing_comparison.md    # สรุปรอบล่าสุด
    └── runs/                          # ไฟล์ที่ Ultralytics สร้าง
```

## 1. ทำภาพก่อน/หลัง (ไม่ต้องมีโมเดล)

```bash
python -m src.make_comparison \
  --input path/to/validation/images \
  --methods clahe median \
  --limit 10
```

ผลจะอยู่ใน `results/preprocessing/` ภาพในมอนทาจถูกย่อเพื่อดูเฉย ๆ ส่วนภาพที่ใช้วัดผลยังคงขนาดเดิม

## 2. เตรียมชุด validation ที่ปรับภาพแล้ว (ไม่ต้องมีโมเดล)

```bash
python -m src.evaluate_preprocessing \
  --data path/to/data.yaml \
  --split val \
  --method clahe \
  --prepare-only
```

ตรวจว่า `data.yaml`, ภาพ และป้ายกำกับที่สร้างขึ้นถูกต้องก่อนรอโมเดล

## 3. วัดผลจริงหลังได้ `best.pt`

```bash
python -m src.evaluate_preprocessing \
  --data path/to/data.yaml \
  --model best.pt \
  --split val \
  --method clahe \
  --conf 0.25 \
  --imgsz 640
```

สคริปต์จะรัน `YOLO.val()` สองรอบด้วยโมเดลไฟล์เดียวกัน:

1. ภาพต้นฉบับ
2. ภาพที่ปรับด้วยวิธีที่เลือก

แล้วเก็บ `preprocessing_comparison.csv` และ `.md` โดยอัตโนมัติ

## ตารางแม่แบบสำหรับส่งต่อคนที่ 5

| การทดลอง | Precision | Recall | mAP50 | mAP50-95 | ชุดภาพ | ไฟล์โมเดล | หมายเหตุ |
|---|---:|---:|---:|---:|---|---|---|
| ภาพเดิม |  |  |  |  | Validation | `best.pt` |  |
| หลังปรับด้วย `...` |  |  |  |  | Validation | `best.pt` |  |
| ผลสุดท้าย |  |  |  |  | Test | `best.pt` | ใช้ test ครั้งเดียว |

## กติกาการเทียบ

- ใช้โมเดลไฟล์เดียวกัน (ตรวจ SHA-256 ในผลลัพธ์)
- ใช้รายชื่อภาพชุดเดียวกัน (ตรวจ `image_list_sha256`)
- ใช้ `conf` และ `imgsz` เดียวกัน
- ต่างกันแค่เปิด/ปิดการปรับภาพ
- ใช้ validation เลือกวิธี และใช้ test รายงานผลสุดท้าย
- ถ้าผลแย่ลง ให้รายงานตามจริงและไม่ตั้งเป็นค่าเริ่มต้น

## ตัวอย่างผลตรวจถูก/พลาด

เก็บภาพที่โมเดลตรวจถูกและตรวจพลาดไว้ใน `results/evaluation/examples/` พร้อมอธิบายสาเหตุ เช่น ภาพมืด หมวกเล็ก ป้ายกำกับไม่ชัด หรือสับสนระหว่างคลาส

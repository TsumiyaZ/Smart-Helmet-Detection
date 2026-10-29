---
title: คนที่ 3 — YOLO Model และ Inference
team-role: model
tags: [smart-helmet, team-task]
status: ยังไม่เริ่ม
---

# คนที่ 3 — YOLO Model และ Inference

> [!abstract] ผลลัพธ์ของบทบาทนี้
> ส่งโมเดลที่ทีมโหลดและเรียกใช้ซ้ำได้ พร้อม checkpoint, คำสั่ง train, class mapping และผล validation ไม่ส่งแค่ภาพ screenshot หรือค่า accuracy ที่สร้างซ้ำไม่ได้

## งานตามวัน

### Day 1
- [ ] ยืนยันเวอร์ชัน framework/YOLO และ dependency ที่ใช้
- [ ] ลองโหลด pretrained checkpoint และ infer ภาพหนึ่งภาพ
- [ ] ตกลง class mapping กับคนที่ 2 และ return format กับคนที่ 1

### Day 2
- [ ] Train/fine-tune เมื่อ dataset config ผ่านการตรวจ
- [ ] บันทึก command, epoch, image size, batch, seed ถ้าตั้งได้, config และ log
- [ ] เก็บ checkpoint ล่าสุดที่ใช้ได้; อย่ารอ train รอบยาวโดยไม่มี fallback

### Day 3
- [ ] เปรียบเทียบ checkpoint และเลือกตัวที่เหมาะจาก validation metric/ตัวอย่าง ไม่เลือกจากภาพเดียว
- [ ] สร้างฟังก์ชัน inference ที่รับภาพ RGB และคืน detections ที่ class/confidence/box แยกชัดเจน
- [ ] ส่ง `best.pt` หรือ path ที่แน่นอน พร้อมวิธีโหลดให้คนที่ 1

### Day 4–5
- [ ] Freeze weight/config สำหรับ demo
- [ ] ตรวจ inference บนภาพที่ไม่ใช้ train และจัดตัวอย่างผลดี/ผลผิดให้คนที่ 4–5
- [ ] ตรวจว่าการใช้ CLAHE ตอน infer ตรงกับวิธีที่ทดลองหรือฝึกไว้

**ส่งมอบ:** weight, inference function, train command/config, validation result และ class names

## รูปแบบ output ที่แนะนำ

```python
{
    "detections": [
        {"class_name": "helmet", "confidence": 0.91, "bbox_xyxy": [x1, y1, x2, y2]}
    ],
    "counts": {"helmet": 1, "no_helmet": 0},
    "annotated_rgb": image_rgb
}
```

ชื่อ class เป็นตัวอย่างเท่านั้น ให้ใช้ค่าจริงจาก dataset และรักษาข้อตกลงใน [[Obsidian-work/03 - สัญญาเชื่อมระบบ]]

## เกณฑ์รับงาน

- [ ] โหลด weight จาก path ที่อธิบายใน README ได้
- [ ] class ID ถูกแปลงเป็น class name ถูกต้อง
- [ ] confidence threshold ส่งเป็น parameter ไม่ hard-code หลายจุด
- [ ] ทดสอบภาพขนาด/สีต่างกันอย่างน้อย 3 กรณี
- [ ] ไม่มีการรายงาน metric บน train set ว่าเป็นผลทดสอบ

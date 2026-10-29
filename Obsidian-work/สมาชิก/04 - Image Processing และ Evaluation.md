---
title: คนที่ 4 — Image Processing และ Evaluation
team-role: preprocessing-evaluation
tags: [smart-helmet, team-task]
status: ยังไม่เริ่ม
---

# คนที่ 4 — Image Processing และ Evaluation

> [!abstract] ผลลัพธ์ของบทบาทนี้
> แสดงให้เห็นการใช้ Image Processing ที่สัมพันธ์กับโจทย์ และวัดผลด้วยวิธีที่ตรวจสอบได้ งานนี้ต้องช่วยทีมตอบว่า preprocessing ช่วยจริงหรือไม่ ไม่ใช่เพิ่มเทคนิคเพื่อให้ครบรายการเฉย ๆ

## งานตามวัน

### Day 1
- [ ] สำรวจความสว่าง/contrast/ความละเอียดของภาพตัวอย่าง
- [ ] เลือกเทคนิคที่จะทดลองหนึ่งอย่าง เช่น CLAHE บนช่อง luminance ใน LAB/YCrCb
- [ ] กำหนดให้ใช้ validation set เดิมในการเปรียบเทียบ raw vs processed

### Day 2
- [ ] ทำ `preprocessing.py` ให้เป็นฟังก์ชันรับ/คืนภาพรูปแบบชัดเจน
- [ ] เทียบตัวอย่างก่อน/หลังและตรวจสี/artefact
- [ ] อย่าบังคับ CLAHE กับ inference ถ้ายังไม่ผ่านการทดลอง

### Day 3
- [ ] รัน model บน validation/test ที่ตกลงกัน ทั้ง original และ processed ตามเวลาที่มี
- [ ] เก็บ precision, recall, mAP ที่ framework ให้ได้; บันทึก split/confidence/config
- [ ] ดู false positive/false negative และจัดตัวอย่างอย่างละกรณี

### Day 4–5
- [ ] สรุปตารางผลและเหตุผลเลือก raw หรือ processed
- [ ] ทำ confusion matrix ถ้า metric/output รองรับ; อธิบาย class-wise ผลลัพธ์
- [ ] ส่งตัวเลขเดียวกันให้คนที่ 5 ใส่รายงานและสไลด์

**ส่งมอบ:** preprocessing function, วิธีใช้, ตารางผลเปรียบเทียบ, metric และตัวอย่าง failure cases

## ข้อควรระวังการประเมิน

- ใช้ validation เพื่อเลือก threshold/แนวทาง และกัน test ไว้ตรวจรอบสุดท้าย
- เปรียบเทียบ raw กับ processed บนภาพชุดเดียวกันและ checkpoint เดียวกัน
- ระบุว่าตัวเลขเป็น mAP50, mAP50-95, precision หรือ recall ให้ชัด ห้ามเขียนแค่ “accuracy” ถ้าเป็น object detection
- ถ้าเวลาไม่พอทำ confusion matrix ให้รายงาน metric ที่มีจริง พร้อมตัวอย่างผิดพลาดแทนการแต่งตัวเลข
- ถ้า CLAHE ทำให้ metric แย่ลง ให้รายงานผลและเลือกไม่เปิดใช้เป็น default ได้

## เกณฑ์รับงาน

- [ ] preprocessing deterministic และรับภาพตามสัญญา interface
- [ ] เขียนชื่อเทคนิคและเหตุผลประกอบ
- [ ] ตารางมีชุดข้อมูล/metric/config ชัดเจน
- [ ] ตัวเลขในรายงานตรงกับ log หรือไฟล์ผลประเมิน

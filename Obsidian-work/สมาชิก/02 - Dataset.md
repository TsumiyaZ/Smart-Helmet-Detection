---
title: คนที่ 2 — Dataset และ Data Quality
team-role: dataset
tags: [smart-helmet, team-task]
status: ยังไม่เริ่ม
---

# คนที่ 2 — Dataset และ Data Quality

> [!abstract] ผลลัพธ์ของบทบาทนี้
> ทำให้ทีมรู้ว่าโมเดลเรียนจากอะไร ข้อมูลมาจากไหน มี label อะไร และผลประเมินเชื่อถือได้แค่ไหน หาก class หรือ annotation ไม่ตรงโจทย์ จะพบก่อนเสียเวลาฝึกโมเดล

## งานตามวัน

### Day 1 — เลือกและตรวจ dataset
- [ ] หา dataset ที่ใช้ได้จริงและตรวจ source/license/เงื่อนไขการเผยแพร่
- [ ] เปิดดู annotation/config เพื่อยืนยันชื่อ class และความหมายของแต่ละ class
- [ ] ตรวจว่าโจทย์ของทีมคือจำแนก `helmet`/`no_helmet` หรือจับคู่ person กับ helmet; แจ้งความเสี่ยงถ้าข้อมูลไม่ตอบโจทย์
- [ ] ส่งภาพตัวอย่างดี/ผิดปกติให้ทั้งทีม

### Day 2 — เตรียมชุดข้อมูล
- [ ] จัดโครงสร้างตามรูปแบบที่ framework ใช้ และตรวจ path ใน config
- [ ] แบ่ง train/validation/test โดยแยก test ไว้ใช้ประเมินรอบสุดท้าย
- [ ] นับจำนวนภาพและ instance ต่อ class; ตรวจ label ว่าง/เกินภาพ/ผิด format
- [ ] ตรวจภาพซ้ำหรือเกือบซ้ำระหว่าง split ถ้าทำได้

### Day 3 — ตรวจคุณภาพระหว่างพัฒนา
- [ ] ตรวจ prediction ผิดตัวอย่างจาก validation แล้วแยกว่าเกิดจาก label, class imbalance, ภาพไกล/มืด หรือโมเดล
- [ ] ถ้าแก้ annotation ให้บันทึกจำนวน/เหตุผลและอย่าแก้ test โดยดูผลเพื่อทำให้ metric สูงขึ้น

### Day 4–5 — สรุปเพื่อส่งงาน
- [ ] ทำตาราง `class | train | val | test`
- [ ] เขียน source, license, วันที่ดาวน์โหลด และข้อจำกัด dataset ลง README/รายงาน
- [ ] จัดตัวอย่างภาพให้ไม่มีข้อมูลที่ทีมไม่มีสิทธิ์เผยแพร่

**ส่งมอบ:** dataset config, สถิติ split/class, ตัวอย่าง annotation, source/license และคำอธิบายข้อจำกัด

## เกณฑ์รับงาน

- [ ] ชื่อ class ใน dataset, model output, UI, README และสไลด์ตรงกัน
- [ ] split แยกกันจริง และ test ไม่ถูกนำไป train
- [ ] ระบุที่มาและ license; ไม่บอกว่าเป็น dataset ของทีมถ้าไม่ได้เก็บเอง
- [ ] แจ้งทีมทันทีถ้าข้อมูลน้อยหรือ label ไม่ครอบคลุม no-helmet

## ข้อมูลให้กรอก

- Dataset URL:
- License/source:
- จำนวนภาพ:
- Class names:
- จำนวน instance ต่อ split/class:
- ข้อจำกัด/ความเสี่ยง:

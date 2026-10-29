---
title: Smart Helmet Detection — Dashboard
tags:
  - computer-vision
  - smart-helmet
  - team-project
status: กำลังดำเนินงาน
team-size: 5
project-days: 5
---

# Smart Helmet Detection — Dashboard

> [!summary] เป้าหมายร่วม
> ใน 5 วัน ส่งมอบระบบตรวจจับการสวมหมวกนิรภัยที่ทำงานครบเส้นทาง: เลือกภาพ/วิดีโอ → preprocessing → YOLO inference → แสดงกรอบและสถิติใน Streamlit → ประเมินผล → ส่ง README, GitHub และ demo

> [!warning] วันนำเสนอ
> เอกสารโจทย์ระบุวันนำเสนอ **2 พฤศจิกายน 2569** หากวันดังกล่าวยังเป็นกำหนดส่ง ให้มี MVP ที่ใช้งานได้ภายใน Day 3 และ freeze ระบบสำหรับนำเสนอไม่เกิน Day 4 ส่วน Day 5 ใช้เป็น buffer/เก็บงานหลังส่ง แผนด้านล่างใช้ Day 1–5 เพื่อให้เลื่อนตามเวลาจริงของทีมได้

## เปิดโน้ตงานของแต่ละคน

- [[Obsidian-work/สมาชิก/01 - แอปและรวมระบบ|คนที่ 1 — Streamlit และรวมระบบ]]
- [[Obsidian-work/สมาชิก/02 - Dataset|คนที่ 2 — Dataset และ Data Quality]]
- [[Obsidian-work/สมาชิก/03 - Model|คนที่ 3 — YOLO และ Inference]]
- [[Obsidian-work/สมาชิก/04 - Image Processing และ Evaluation|คนที่ 4 — Preprocessing และ Evaluation]]
- [[Obsidian-work/สมาชิก/05 - Video Pipeline, QA เอกสาร และ Demo|คนที่ 5 — Video Pipeline, QA, README และ Demo]]
- [[Obsidian-work/02 - ลำดับงานตั้งแต่เริ่มจนส่ง|ลำดับงานตั้งแต่เริ่มจนส่ง]]
- [[Obsidian-work/03 - สัญญาเชื่อมระบบ|ข้อตกลงชื่อไฟล์และรูปแบบข้อมูลระหว่างคน]]

| คน | เจ้าของหลัก | ผลงานที่ต้องส่ง | ผู้รับงานต่อ |
|---|---|---|---|
| **1** | Streamlit UI และ app flow | `app.py`: upload, controls, แสดงผลภาพ/วิดีโอและสถิติ | คนที่ 5 ทดสอบและคนทั้งทีมใช้ demo |
| **2** | Dataset และ data quality | dataset config, class mapping, split, สถิติและแหล่งที่มา | คนที่ 3 train; คนที่ 4 evaluate |
| **3** | YOLO model และ inference | `best.pt`, train config, ฟังก์ชัน predict ภาพ/เฟรม | คนที่ 1 และคนที่ 5 เรียกใช้ |
| **4** | Image Processing และ evaluation | `preprocessing.py`, ตาราง metric, ตัวอย่างข้อผิดพลาด | คนที่ 1 เรียกใช้; คนที่ 5 ใส่รายงาน/สไลด์ |
| **5** | Video pipeline, QA และ release | `video_processing.py`, วิดีโอผลลัพธ์, checklist QA, README และ demo | คนที่ 1 เรียก video pipeline; ทีมส่งงาน |

> [!note] งานวิดีโอของคนที่ 5
> โมเดลตรวจทีละภาพ ส่วนวิดีโอเป็นภาพหลายเฟรมเรียงกัน คนที่ 5 จึงทำโค้ดอ่านเฟรม เรียกโมเดลซ้ำ และเขียนเฟรมผลลัพธ์กลับเป็นคลิป; คนที่ 1 ทำปุ่มอัปโหลด/แสดงคลิป และคนที่ 3 ให้ฟังก์ชันตรวจหนึ่งเฟรม.

**รายชื่อสมาชิก:** คนที่ 1 __________ · คนที่ 2 __________ · คนที่ 3 __________ · คนที่ 4 __________ · คนที่ 5 __________

## ขอบเขตที่ทีมต้องล็อกใน Day 1

1. เลือก dataset หนึ่งชุดที่มีสิทธิ์ใช้และตรวจ label ได้จริง
2. ยืนยันชื่อคลาสจาก dataset ก่อนเขียนโค้ด ห้ามสมมติว่าเป็น `helmet`/`no_helmet` จนกว่าจะตรวจ `data.yaml`/annotation แล้ว
3. ถ้าเป้าหมายคือบอกว่า “ผู้ขับขี่ไม่สวมหมวก” dataset ต้องมี label ที่ตอบคำถามนี้ได้โดยตรง หรือมีทั้ง person และ helmet พร้อมวิธีจับคู่ที่ตรวจสอบได้
4. ใช้ YOLO11 pretrained แล้ว fine-tune เป็น baseline; ห้ามรอ train รอบสมบูรณ์ก่อนเริ่มทำแอป
5. เลือก preprocessing 1 เทคนิคที่อธิบายได้ เช่น CLAHE บนช่องสว่าง; เทียบกับภาพต้นฉบับบน validation set ก่อนใช้จริง
6. MVP ต้องรับภาพและแสดงผลได้ก่อน; เพิ่มวิดีโอสั้นเมื่อ image path เสถียรแล้ว

> [!tip] กติกาเลือกทางที่เร็ว
> ถ้า Day 1 จบแล้วยังหา dataset ที่เหมาะไม่เจอ ให้เลือก dataset สาธารณะที่มี annotation พร้อมและขอบเขตแคบ แทนการเริ่ม annotate จำนวนมากเอง ถ้า train กินเวลานานให้ใช้ checkpoint ที่ได้ผลดีที่สุดจนถึงตอนนั้น แล้วลงทุนเวลาที่เหลือกับ integration และการประเมินที่ถูกต้อง

## ลำดับงานและการส่งต่องาน

| ขั้น | เจ้าของหลัก | ส่งอะไรให้ใคร | เงื่อนไขก่อนส่งต่อ |
|---|---|---|---|
| 1. ยืนยัน dataset/คลาส | คนที่ 2 | ส่ง config, class names, split และตัวอย่างให้คนที่ 3–4 | class ตอบโจทย์ helmet/no-helmet และ path ใช้งานได้ |
| 2. ทำ baseline model | คนที่ 3 | ส่ง weight + inference function ให้คนที่ 1 และผล validation ให้คนที่ 4 | โมเดลโหลดได้และ class IDs ตรงกับ config |
| 3. ทำ preprocessing/วัดผล | คนที่ 4 | ส่งฟังก์ชันให้คนที่ 1 และ metric/ข้อจำกัดให้คนที่ 5 | ทดลองบน split เดิมและเก็บค่าที่มาของตัวเลข |
| 4. ประกอบระบบ | คนที่ 1 + คนที่ 5 | คนที่ 1 ทำ UI; คนที่ 5 ทำ video pipeline และเชื่อม inference ต่อเฟรม | image path ผ่าน; วิดีโอจำกัดขนาด/ความยาวและบันทึกผลได้ |
| 5. ตรวจและแพ็กส่ง | คนที่ 5 เป็น release owner; ทุกคนตรวจส่วนของตัวเอง | ส่งรายการบั๊กกลับเจ้าของส่วน พร้อม README/demo ให้ทีม | วิธีรันผ่าน clean start และเนื้อหาตรงกับผลจริง |

**จังหวะประสานงาน:** นัด sync วันละ 10–15 นาที ให้ทุกคนบอก 1) เสร็จอะไร 2) วันนี้จะส่งอะไรและกี่โมง 3) ติดอะไร/ต้องการใครช่วย หากรอ dependency เกินครึ่งวัน ให้ส่ง mock หรือ contract ที่ชัดเจนเพื่อให้งานส่วนอื่นเดินต่อ

## แผนงาน 5 วัน

### Day 1 — ตกลงโจทย์และเคลียร์ dependencies

- [ ] **คนที่ 1:** ทำหน้า Streamlit โครงเปล่าและตกลง API กับคนที่ 3–4
- [ ] **คนที่ 2:** ส่ง dataset candidate, license/source, ชื่อคลาส, จำนวนข้อมูลและตัวอย่าง annotation
- [ ] **คนที่ 3:** ติดตั้ง/ยืนยัน environment และลอง inference ด้วย pretrained model บนภาพหนึ่งภาพ
- [ ] **คนที่ 4:** ตรวจสภาพภาพและกำหนดวิธีทดลอง preprocessing โดยไม่เปลี่ยน test set
- [ ] **คนที่ 5:** กำหนดขอบเขตวิดีโอ (format, duration, resolution), วาง API ของ video pipeline, สร้าง README skeleton และ test cases
- [ ] **จุดตรวจร่วม:** dataset และ class mapping ถูกยืนยัน; ทุกคนรู้ path/format และ function contract

### Day 2 — ทำ baseline และให้ทุกสายงานทำงานขนาน

- [ ] **คนที่ 1:** upload ภาพ → เรียก inference แบบ placeholder → แสดงภาพผลและสถิติ
- [ ] **คนที่ 2:** จัด train/val/test, ตรวจภาพซ้ำ/label เสีย/ความไม่สมดุล, ส่งชุดพร้อม train
- [ ] **คนที่ 3:** เริ่ม fine-tune, บันทึก config/log/checkpoint และทดลอง inference
- [ ] **คนที่ 4:** ทำ preprocessing แยกเป็นฟังก์ชันและเตรียมตารางเปรียบเทียบ original vs processed
- [ ] **คนที่ 5:** สร้าง `video_processing.py` ที่รับ path + callback สำหรับทำนายทีละเฟรม; ทดลองอ่าน/เขียนวิดีโอสั้นและเตรียม test cases
- [ ] **จุดตรวจร่วม:** อย่างน้อยหนึ่งภาพวิ่งผ่านโมเดลได้จริง มี checkpoint หรือ fallback ที่ใช้ได้

### Day 3 — รวมเป็น MVP และประเมินรอบแรก

- [ ] **คนที่ 1:** เชื่อม model จริงเข้ากับ Streamlit; เพิ่ม confidence threshold และจำนวนที่ตรวจพบ
- [ ] **คนที่ 2:** ยืนยันสถิติ dataset/split และแก้ปัญหา label ที่พบ
- [ ] **คนที่ 3:** ส่ง weight ที่เลือกพร้อมฟังก์ชัน inference และตัวอย่าง output
- [ ] **คนที่ 4:** วัด metric บน validation/test ตามที่แยกไว้และสรุปข้อผิดพลาดสำคัญ
- [ ] **คนที่ 5:** เชื่อม `video_processing.py` กับ inference จริงจากคนที่ 3; ทดสอบวิดีโอสั้นและบันทึกผลลัพธ์
- [ ] **จุดตรวจ MVP:** เปิดแอปได้ → upload ภาพ → เห็นกรอบ/คลาส/confidence → เห็นจำนวน → มี metric เบื้องต้น
- [ ] **หาก deadline คือ 2 พ.ย.:** freeze feature/model ที่จะนำเสนอ เริ่มซ้อม demo วันนี้

### Day 4 — วิดีโอ, hardening และเตรียมนำเสนอ

- [ ] **คนที่ 1:** ทำส่วน UI สำหรับ upload/เริ่มประมวลผล/แสดงวิดีโอผลลัพธ์ โดยเรียก module ของคนที่ 5; ไม่เขียน frame loop ซ้ำ
- [ ] **คนที่ 2:** สรุปแหล่งข้อมูล license จำนวนภาพและ class distribution สำหรับรายงาน/สไลด์
- [ ] **คนที่ 3:** freeze checkpoint/config และตรวจว่าโหลด weight จาก path ที่กำหนดได้
- [ ] **คนที่ 4:** สรุปผล preprocessing, metric, confusion matrix (ถ้ามี) และข้อจำกัด
- [ ] **คนที่ 5:** ทดสอบ video pipeline กับไฟล์ต่าง format/ขนาด, แก้ cleanup/error, ทำ screenshot และร่าง README/demo
- [ ] **จุดตรวจร่วม:** ทดสอบบนเครื่องอย่างน้อย 2 เครื่อง/2 คน; ซ้อมนำเสนอโดยใช้ไฟล์จริง

### Day 5 — ตรวจรับและส่งมอบ

- [ ] **คนที่ 1:** ทดสอบเปิดแอปจาก clean start และ tag/ระบุเวอร์ชันส่ง
- [ ] **คนที่ 2:** ตรวจ dataset documentation และแหล่งอ้างอิง
- [ ] **คนที่ 3:** ตรวจไฟล์ weight, config และวิธีโหลด
- [ ] **คนที่ 4:** ตรวจว่าตัวเลข metric ตรงกันระหว่างผลรัน README และสไลด์
- [ ] **คนที่ 5:** ตรวจ release bundle, README, screenshot, video และไฟล์นำเสนอ; เก็บรายการ known limitations
- [ ] **ทุกคน:** ซ้อม demo รอบสุดท้าย เตรียมภาพ/วิดีโอสำรอง และยืนยันว่าไม่มีไฟล์ส่วนตัวหรือ secret ใน repo

## Definition of Done

- [ ] แอปเริ่มทำงานตาม README บนเครื่องสมาชิกอีกคนได้
- [ ] รับภาพและแสดงผลตรวจจับพร้อมกรอบ, class, confidence และจำนวน
- [ ] วิดีโอสั้นประมวลผลผ่าน `video_processing.py` และ UI แสดง/ดาวน์โหลดผลได้; ถ้าไม่ทันให้ระบุเป็น known limitation และใช้ภาพ demo
- [ ] มี OpenCV preprocessing อย่างน้อย 1 เทคนิคและบอกเหตุผลที่เลือก
- [ ] รายงาน dataset, class, split และ metric จากข้อมูลที่ไม่ใช่ train set
- [ ] มีตัวอย่างผลดีและข้อผิดพลาด ไม่กล่าวอ้างว่า metric ดีเกินหลักฐาน
- [ ] GitHub มี source, requirements, README, screenshot/demo และ weight หรือลิงก์ดาวน์โหลด
- [ ] ทุกคนเข้าใจ flow และตอบได้ว่างานของตัวเองคือส่วนใด

## ช่องสำหรับสถานะทีม

| วันนี้ | งานที่เสร็จ | ปัญหา/ผู้ช่วยที่ต้องการ | งานถัดไป | อัปเดตโดย |
|---|---|---|---|---|
| Day 1 |  |  |  |  |
| Day 2 |  |  |  |  |
| Day 3 |  |  |  |  |
| Day 4 |  |  |  |  |
| Day 5 |  |  |  |  |







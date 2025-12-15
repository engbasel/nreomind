# شرح الباك إند - مشروع NeuroAid 🧠

## 📋 فهرس المحتويات
1. [نظرة عامة](#نظرة-عامة)
2. [معمارية النظام](#معمارية-النظام)
3. [الجزء الأول: API Gateway](#الجزء-الأول-api-gateway)
4. [الجزء الثاني: Flask Main Server](#الجزء-الثاني-flask-main-server)
5. [الجزء الثالث: AI Services](#الجزء-الثالث-ai-services)
6. [كيفية التشغيل](#كيفية-التشغيل)
7. [أسئلة المناقشة المتوقعة](#أسئلة-المناقشة-المتوقعة)

---

## 🎯 نظرة عامة

الباك إند بتاعنا مقسم لـ **4 سيرفرات رئيسية** بتشتغل مع بعض:

```
📱 Flutter App
    ↓
🚪 API Gateway (Port 8080)
    ↓
    ├─→ 🔐 Main Flask Server (Port 5000) - Authentication & Data
    ├─→ 💬 AI Chatbot (Port 5001)
    ├─→ 📊 Stroke Assessment (Port 5002)
    └─→ 🖼️ Image Analysis (Port 5003)
```

---

## 🏗️ معمارية النظام

### ليه استخدمنا أكتر من سيرفر؟

**السبب الرئيسي:** Microservices Architecture

**المميزات:**
- ✅ كل سيرفر مستقل بذاته (لو واحد وقع، الباقي يشتغل)
- ✅ سهولة التطوير (كل واحد يشتغل على جزء)
- ✅ سهولة الصيانة والتحديث
- ✅ أداء أفضل (كل سيرفر متخصص في حاجة واحدة)

---

## 🚪 الجزء الأول: API Gateway

### 📍 المسؤول عن الشرح: [اسم الطالب]

### ما هو الـ Gateway؟
هو **نقطة الدخول الوحيدة** للتطبيق. بدل ما الموبايل يتصل بـ 4 سيرفرات مختلفة، بيتصل بسيرفر واحد بس (الـ Gateway)، والـ Gateway هو اللي يوجه الطلبات للسيرفر الصح.

### الكود الأساسي (gateway.py)

```python
# تعريف السيرفرات المتاحة
SERVICES = {
    'main': {
        'url': 'http://127.0.0.1:5000',
        'prefix': '/api/main'
    },
    'ai_chatbot': {
        'url': 'http://127.0.0.1:5001',
        'prefix': '/api/ai/chat'
    },
    'ai_assessment': {
        'url': 'http://127.0.0.1:5002',
        'prefix': '/api/ai/assessment'
    },
    'ai_image': {
        'url': 'http://127.0.0.1:5003',
        'prefix': '/api/ai/image'
    }
}
```

### كيف يعمل؟

**مثال عملي:**
1. الموبايل يبعت طلب: `POST http://192.168.1.6:8080/api/ai/chat`
2. الـ Gateway يشوف الـ prefix (`/api/ai/chat`)
3. يعرف إن ده للـ Chatbot Service
4. يحول الطلب لـ `http://127.0.0.1:5001/chat`
5. ياخد الرد ويرجعه للموبايل

### المميزات الإضافية:
- **CORS:** بيسمح للموبايل يتصل من أي شبكة
- **Logging:** بيسجل كل الطلبات عشان نعرف مين بعت إيه
- **Error Handling:** لو سيرفر وقع، بيرجع رسالة خطأ واضحة

---

## 🔐 الجزء الثاني: Flask Main Server

### 📍 المسؤول عن الشرح: [اسم الطالب]

### المسؤوليات:
1. **Authentication** (تسجيل دخول وإنشاء حسابات)
2. **إدارة البيانات** (Users, Doctors, Bookings, FAQs)
3. **حفظ نتائج الفحوصات**

### الملفات المهمة:

#### 1. app.py (الملف الرئيسي)
```python
# تسجيل الـ Routes
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(users_bp, url_prefix='/api/users')
app.register_blueprint(doctors_bp, url_prefix='/api/doctors')
app.register_blueprint(bookings_bp, url_prefix='/api/bookings')
app.register_blueprint(scans_bp, url_prefix='/api/scans')
```

#### 2. routes/auth.py (المصادقة)

**التسجيل:**
```python
from werkzeug.security import generate_password_hash
from utils.database import get_users, save_users, get_next_id
from utils.auth import generate_token

@auth_bp.route('/register', methods=['POST'])
def register():
    # 1. استقبال البيانات
    data = request.get_json()
    
    # 2. التحقق من البيانات
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing data'}), 400
    
    # 3. قراءة المستخدمين الحاليين
    users = get_users()
    
    # 4. التحقق من عدم وجود المستخدم
    if any(u['email'] == data['email'] for u in users):
        return jsonify({'error': 'User already exists'}), 400
    
    # 5. تشفير الباسورد
    hashed_password = generate_password_hash(data['password'])
    
    # 6. إنشاء مستخدم جديد
    new_user = {
        'id': get_next_id(users),
        'email': data['email'],
        'password': hashed_password,
        'name': data['name'],
        'phone': data.get('phone', ''),
        'role': 'user',
        'isActive': True,
        'createdAt': datetime.now().isoformat()
    }
    
    # 7. إضافة المستخدم وحفظه في الملف
    users.append(new_user)
    save_users(users)
    
    # 8. إنشاء JWT Token
    token = generate_token(new_user)
    
    # 9. إرجاع الرد (بدون الباسورد)
    user_response = {k: v for k, v in new_user.items() if k != 'password'}
    
    return jsonify({'accessToken': token, 'user': user_response}), 201
```

**تسجيل الدخول:**
```python
from werkzeug.security import check_password_hash

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # 1. قراءة المستخدمين
    users = get_users()
    
    # 2. البحث عن المستخدم
    user = next((u for u in users if u['email'] == data['email']), None)
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # 3. التحقق من الباسورد
    if not check_password_hash(user['password'], data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # 4. إنشاء Token
    token = generate_token(user)
    
    # 5. إرجاع الرد
    user_response = {k: v for k, v in user.items() if k != 'password'}
    return jsonify({'accessToken': token, 'user': user_response})
```

#### 3. routes/scans.py (إدارة الفحوصات)

```python
from utils.database import get_db, save_db, get_next_id

@scans_bp.route('', methods=['GET'])
@require_auth  # يتطلب تسجيل دخول
def get_scans():
    user_id = request.user_id
    
    # قراءة قاعدة البيانات
    db = get_db()
    scans = db.get('scans', [])
    
    # جلب الفحوصات الخاصة بالمستخدم فقط
    user_scans = [s for s in scans if s['userId'] == user_id]
    
    return jsonify(user_scans)

@scans_bp.route('', methods=['POST'])
@require_auth
def create_scan():
    user_id = request.user_id
    data = request.get_json()
    
    # قراءة قاعدة البيانات
    db = get_db()
    scans = db.get('scans', [])
    
    # إنشاء فحص جديد
    new_scan = {
        'id': get_next_id(scans),
        'userId': user_id,
        'result': data['result'],
        'confidence': data['confidence'],
        'findings': data['findings'],
        'imageUrl': data['imageUrl'],
        'createdAt': datetime.now().isoformat()
    }
    
    # إضافة وحفظ
    scans.append(new_scan)
    db['scans'] = scans
    save_db(db)
    
    return jsonify(new_scan), 201
```

### قاعدة البيانات: Local JSON Files

**ليه استخدمنا JSON Files؟**
- ✅ بسيطة وسهلة في التعامل
- ✅ مش محتاجة setup معقد
- ✅ مناسبة للمشاريع الصغيرة والـ Prototyping
- ✅ سهل نقرأها ونعدلها يدوياً
- ✅ مش محتاجة سيرفر database منفصل

**الملفات اللي عندنا (في مجلد `backend/data/`):**
- `users.json` - بيانات المستخدمين (إيميل، باسورد مشفر، اسم، تليفون)
- `db.json` - البيانات الرئيسية (الأطباء، الحجوزات، الفحوصات)
- `faqs.json` - الأسئلة الشائعة

**كيف بنتعامل معاها؟**

```python
# في utils/database.py
import json

def load_json_file(filename):
    """قراءة ملف JSON"""
    with open(f'data/{filename}', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(filename, data):
    """حفظ بيانات في ملف JSON"""
    with open(f'data/{filename}', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_users():
    """جلب كل المستخدمين"""
    return load_json_file('users.json')

def save_users(users):
    """حفظ المستخدمين"""
    save_json_file('users.json', users)
```

**مثال على بيانات في users.json:**
```json
[
  {
    "id": 1,
    "email": "user@example.com",
    "password": "$2b$10$hashed_password_here",
    "name": "أحمد محمد",
    "phone": "+20 100 123 4567",
    "role": "user",
    "isActive": true,
    "createdAt": "2025-12-15T10:30:00"
  }
]
```

---

## 🤖 الجزء الثالث: AI Services

### 📍 المسؤولون عن الشرح: [3 طلاب]

### 3.1 AI Chatbot Service (Port 5001)

**المسؤول:** [اسم الطالب]

**الوظيفة:** محادثة ذكية مع المستخدم عن السكتة الدماغية

**التقنية المستخدمة:** Google Gemini AI

**الكود (ai_services/chatbot/app.py):**

```python
import google.generativeai as genai

# إعداد Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    history = data.get('conversationHistory', [])
    
    # بناء الـ Context
    context = """أنت مساعد طبي متخصص في السكتة الدماغية.
    مهمتك مساعدة المستخدمين بمعلومات دقيقة وواضحة."""
    
    # إرسال للـ AI
    chat = model.start_chat(history=history)
    response = chat.send_message(context + message)
    
    return jsonify({
        'response': response.text,
        'timestamp': datetime.now().isoformat()
    })
```

**كيف يعمل؟**
1. المستخدم يكتب سؤال
2. نبعت السؤال + تاريخ المحادثة لـ Gemini
3. Gemini يرد بإجابة ذكية
4. نرجع الإجابة للمستخدم

---

### 3.2 Stroke Assessment Service (Port 5002)

**المسؤول:** [اسم الطالب]

**الوظيفة:** تقييم خطر الإصابة بالسكتة الدماغية بناءً على بيانات المستخدم

**التقنية:** Machine Learning Model (تم تدريبه مسبقاً)

**البيانات المطلوبة:**
- العمر
- الجنس
- ضغط الدم
- أمراض القلب
- مستوى السكر
- BMI
- التدخين

**الكود:**

```python
import joblib
import numpy as np

# تحميل الموديل المدرب
model = joblib.load('stroke_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # تحضير البيانات
    features = np.array([[
        data['age'],
        1 if data['gender'] == 'male' else 0,
        data['hypertension'],
        data['heartDisease'],
        data['avgGlucoseLevel'],
        data['bmi'],
        # ... باقي الـ features
    ]])
    
    # التنبؤ
    prediction = model.predict_proba(features)[0]
    risk_percentage = prediction[1] * 100
    
    # تحديد مستوى الخطر
    if risk_percentage < 30:
        risk_level = 'low'
    elif risk_percentage < 60:
        risk_level = 'medium'
    else:
        risk_level = 'high'
    
    return jsonify({
        'riskLevel': risk_level,
        'riskPercentage': risk_percentage,
        'recommendations': get_recommendations(risk_level)
    })
```

**كيف تم تدريب الموديل؟**
1. جمع dataset من Kaggle (Stroke Prediction Dataset)
2. تنظيف البيانات
3. تدريب الموديل باستخدام Random Forest / XGBoost
4. حفظ الموديل في ملف `.pkl`

---

### 3.3 Image Analysis Service (Port 5003)

**المسؤول:** [اسم الطالب]

**الوظيفة:** تحليل صور الأشعة للكشف عن السكتة الدماغية

**التقنية:** Deep Learning (CNN - Convolutional Neural Network)

**الكود:**

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# تحميل الموديل
model = load_model('stroke_detection_model.h5')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. استقبال الصورة
    file = request.files['image']
    
    # 2. معالجة الصورة
    image = Image.open(file.stream)
    image = image.resize((224, 224))  # تصغير للحجم المطلوب
    image_array = np.array(image) / 255.0  # Normalization
    image_array = np.expand_dims(image_array, axis=0)
    
    # 3. التنبؤ
    prediction = model.predict(image_array)
    confidence = float(prediction[0][0])
    
    # 4. تحديد النتيجة
    if confidence > 0.7:
        result = 'stroke_detected'
        findings = ['توجد علامات محتملة للسكتة الدماغية']
    else:
        result = 'normal'
        findings = ['الصورة تبدو طبيعية']
    
    # 5. حفظ الصورة
    filename = f"scan-{int(time.time())}.jpg"
    filepath = os.path.join('uploads', filename)
    image.save(filepath)
    
    return jsonify({
        'result': result,
        'confidence': confidence,
        'findings': findings,
        'imageUrl': f'/uploads/{filename}'
    })
```

**كيف تم تدريب الموديل؟**
1. جمع صور أشعة (CT/MRI) من datasets طبية
2. تقسيم الصور: Normal vs Stroke
3. استخدام Transfer Learning (مثل VGG16 أو ResNet)
4. Fine-tuning على dataset الخاص بنا
5. حفظ الموديل في `.h5`

---

## 🚀 كيفية التشغيل

### المتطلبات:
- Python 3.8+
- pip
- Gemini API Key (للـ Chatbot)

### خطوات التشغيل:

#### 1. إعداد البيئة
```bash
cd backend
pip install -r requirements.txt
```

#### 2. إعداد ملفات .env

**للـ Main Server (flask_server/.env):**
```env
PORT=5000
JWT_SECRET=your_secret_key_here
JWT_EXPIRES_IN=7d
MAX_FILE_SIZE=10485760
UPLOAD_PATH=./uploads
```

**للـ Chatbot (ai_services/chatbot/.env):**
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5001
```

#### 3. تشغيل جميع السيرفرات

**Windows:**
```bash
start_all_servers.bat
```

**أو يدوياً:**
```bash
# Terminal 1: Gateway
python gateway.py

# Terminal 2: Main Server
cd flask_server
python app.py

# Terminal 3: Chatbot
cd ai_services/chatbot
python app.py

# Terminal 4: Assessment
cd ai_services/stroke_assessment
python app.py

# Terminal 5: Image Analysis
cd ai_services/stroke_image
python app.py
```

#### 4. التحقق من التشغيل
افتح المتصفح: `http://localhost:8080/health`

يجب أن ترى:
```json
{
  "gateway": "OK",
  "services": {
    "main": {"status": "online"},
    "ai_chatbot": {"status": "online"},
    "ai_assessment": {"status": "online"},
    "ai_image": {"status": "online"}
  }
}
```

---

## 🎓 أسئلة المناقشة المتوقعة

### أسئلة عامة:

**س1: ليه استخدمتوا Microservices بدل Monolithic؟**
- **الإجابة:** عشان كل سيرفر مستقل، لو حصل مشكلة في AI مثلاً، الـ Authentication لسه شغال. وكمان سهل إننا نطور كل جزء لوحده.

**س2: إزاي بتأمنوا الـ API؟**
- **الإجابة:** باستخدام JWT Tokens. كل request محتاج token صحيح، والـ token ده بيتعمل بس لما المستخدم يسجل دخول.

**س3: لو عدد المستخدمين زاد جداً، النظام هيتحمل؟**
- **الإجابة:** ممكن نعمل Horizontal Scaling - نشغل أكتر من نسخة من كل سيرفر ونستخدم Load Balancer.

### أسئلة للـ Gateway:

**س4: ليه محتاجين Gateway أصلاً؟**
- **الإجابة:** عشان نبسط الاتصال. بدل ما الموبايل يعرف 4 عناوين، بيعرف عنوان واحد بس.

**س5: إيه اللي يحصل لو سيرفر من السيرفرات وقع؟**
- **الإجابة:** الـ Gateway بيرجع error 503 (Service Unavailable) مع رسالة واضحة.

### أسئلة للـ Authentication:

**س6: إزاي بتحموا الـ passwords؟**
- **الإجابة:** باستخدام bcrypt للتشفير. مش بنحفظ الباسورد نفسه، بنحفظ hash منه.

**س7: إيه الفرق بين JWT و Session؟**
- **الإجابة:** JWT stateless (مش محتاج نحفظ حاجة في السيرفر)، Session stateful (محتاج نحفظ في السيرفر).

### أسئلة للـ AI Services:

**س8: إزاي الـ Chatbot بيفهم السياق؟**
- **الإجابة:** بنبعت تاريخ المحادثة كلها لـ Gemini، فهو بيفهم السياق من الرسائل السابقة.

**س9: دقة الموديل بتاع الـ Stroke Assessment قد إيه؟**
- **الإجابة:** [حسب الموديل اللي درّبتوه] مثلاً: 85% accuracy على test set.

**س10: الموديل بتاع الصور اتدرب على كام صورة؟**
- **الإجابة:** [حسب الـ dataset] مثلاً: 5000 صورة (2500 normal, 2500 stroke).

**س11: لو الصورة مش واضحة، الموديل بيعمل إيه؟**
- **الإجابة:** بيرجع confidence منخفض وبننصح المستخدم يرفع صورة أوضح.

### أسئلة تقنية متقدمة:

**س12: إزاي بتتعاملوا مع الـ CORS؟**
- **الإجابة:** باستخدام flask-cors، بنسمح لكل الـ origins في development، لكن في production هنحدد الـ domains المسموحة.

**س13: إيه الـ timeout للـ requests؟**
- **الإجابة:** 30 ثانية للـ requests العادية، 60 ثانية للـ AI requests (عشان بتاخد وقت أطول).

**س14: بتحفظوا الصور فين؟**
- **الإجابة:** في مجلد `uploads/scans` على السيرفر، وبنحفظ الـ path في الـ database.

---

## 📊 تقسيم المهام للعرض

### الطالب 1: المقدمة + Gateway
- شرح المعمارية العامة
- شرح دور الـ Gateway
- عرض الكود الأساسي
- **الوقت:** 5-7 دقائق

### الطالب 2: Main Server + Authentication
- شرح Flask Server
- شرح Authentication (JWT)
- **الوقت:** 5-7 دقائق

### الطالب 3: AI Chatbot
- شرح Gemini Integration
- عرض demo للـ chatbot
- **الوقت:** 3-5 دقائق

### الطالب 4: Stroke Assessment
- شرح الـ ML Model
- شرح الـ features
- عرض النتائج
- **الوقت:** 3-5 دقائق

### الطالب 5: Image Analysis
- شرح الـ CNN Model
- شرح معالجة الصور
- عرض demo
- **الوقت:** 3-5 دقائق

### الطالب 6: الختام + Q&A
- ملخص سريع
- التحديات والحلول
- الخطط المستقبلية
- **الوقت:** 2-3 دقائق

---

## 🔧 نصائح للعرض

1. **جهزوا Demo شغال:** أحسن من ألف كلمة
2. **اشرحوا بأمثلة:** "لما المستخدم يعمل كذا، بيحصل كذا"
3. **كونوا جاهزين للأسئلة:** اقروا الكود كويس
4. **اعرضوا الـ Postman:** وروا الـ requests والـ responses
5. **خلوا حد يسأل أسئلة صعبة قبل العرض:** عشان تكونوا جاهزين

---

## 📚 مصادر إضافية

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Google Gemini API](https://ai.google.dev/docs)
- [JWT.io](https://jwt.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---

**بالتوفيق في العرض! 🚀**

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
@auth_bp.route('/register', methods=['POST'])
def register():
    # 1. استقبال البيانات
    data = request.get_json()
    
    # 2. التحقق من البيانات
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing data'}), 400
    
    # 3. تشفير الباسورد
    hashed_password = bcrypt.hashpw(
        data['password'].encode('utf-8'), 
        bcrypt.gensalt()
    )
    
    # 4. حفظ المستخدم في Appwrite
    user = account.create(...)
    
    # 5. إنشاء JWT Token
    token = jwt.encode({
        'userId': user['$id'],
        'exp': datetime.utcnow() + timedelta(days=7)
    }, SECRET_KEY)
    
    return jsonify({'token': token, 'user': user})
```

**تسجيل الدخول:**
```python
@auth_bp.route('/login', methods=['POST'])
def login():
    # 1. التحقق من الإيميل والباسورد
    # 2. إنشاء Session في Appwrite
    # 3. إرجاع Token
```

#### 3. routes/scans.py (إدارة الفحوصات)

```python
@scans_bp.route('', methods=['GET'])
@require_auth  # يتطلب تسجيل دخول
def get_scans():
    user_id = request.user_id
    # جلب كل الفحوصات الخاصة بالمستخدم
    scans = databases.list_documents(
        database_id=DATABASE_ID,
        collection_id=SCANS_COLLECTION_ID,
        queries=[Query.equal('userId', user_id)]
    )
    return jsonify(scans)
```

### قاعدة البيانات: Appwrite

**ليه اخترنا Appwrite؟**
- ✅ Backend as a Service (جاهز ومش محتاج نعمل كل حاجة من الصفر)
- ✅ فيه Authentication جاهز
- ✅ فيه Database جاهز
- ✅ فيه Storage للصور
- ✅ مجاني للمشاريع الصغيرة

**Collections اللي عندنا:**
- `users` - بيانات المستخدمين
- `scans` - نتائج الفحوصات
- `bookings` - الحجوزات
- `doctors` - بيانات الأطباء

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
- حساب Appwrite
- Gemini API Key

### خطوات التشغيل:

#### 1. إعداد البيئة
```bash
cd backend
pip install -r requirements.txt
```

#### 2. إعداد ملفات .env

**للـ Main Server:**
```env
APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1
APPWRITE_PROJECT_ID=your_project_id
APPWRITE_API_KEY=your_api_key
JWT_SECRET=your_secret_key
```

**للـ Chatbot:**
```env
GEMINI_API_KEY=your_gemini_key
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
- شرح Appwrite
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
- [Appwrite Documentation](https://appwrite.io/docs)
- [Google Gemini API](https://ai.google.dev/docs)
- [JWT.io](https://jwt.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---

**بالتوفيق في العرض! 🚀**

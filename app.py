from flask import Flask, request, jsonify
from flask_cors import CORS
import groq
import os
import json
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

# Groq Client
groq_client = groq.Groq(api_key=os.environ.get('GROQ_API_KEY'))

# MailerLite API
MAILERLITE_API_KEY = os.environ.get('MAILERLITE_API_KEY')
MAILERLITE_GROUP_ID = os.environ.get('MAILERLITE_GROUP_ID')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        
        # Validate input
        if not data.get('email') or not data.get('headline') or not data.get('goal'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Analyze with Groq
        analysis = analyze_with_groq(
            headline=data['headline'],
            about=data.get('about', ''),
            goal=data['goal']
        )
        
        # Save to MailerLite
        save_to_mailerlite(
            email=data['email'],
            name=data.get('name', ''),
            analysis=analysis
        )
        
        # Log to file
        log_analysis(data, analysis)
        
        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

def analyze_with_groq(headline, about, goal):
    """تحليل الملف باستخدام Groq"""
    
    prompt = f"""أنت خبير علم نفس LinkedIn. حلل الملف التالي:

**Headline:** {headline}
**About:** {about}
**Goal:** {goal}

أعطِ تحليل نفسي دقيق بناءً على:

1. **Score (0-100):**
   - وضوح القيمة (30%)
   - لغة الأثر vs المهام (30%)
   - التموضع النفسي (40%)

2. **Current Perception:** كيف يُقرأ الملف نفسياً
   (مثال: "موظف يبحث عن وظيفة" / "خبير واثق")

3. **Desired Perception:** ما يجب أن يُقرأ بناءً على الهدف
   (مثال: "قائد استراتيجي" / "مستشار موثوق")

4. **Fatal Error:** خطأ نفسي واحد قاتل
   (مثال: "لغة ضعف، لا لغة قوة")

5. **Quick Fix:** مثال Before/After سريع
   - before: جملة من الـ Headline الحالي
   - after: نسخة محسّنة

أجب بـ JSON فقط، بالعربية:
{{
  "score": 0-100,
  "current_perception": "...",
  "desired_perception": "...",
  "fatal_error": "...",
  "quick_fix": {{
    "before": "...",
    "after": "..."
  }}
}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "أنت خبير علم نفسي LinkedIn. أجب بـ JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    # Parse JSON response
    result = json.loads(response.choices[0].message.content)
    return result

def save_to_mailerlite(email, name, analysis):
    """حفظ Email في MailerLite"""
    
    if not MAILERLITE_API_KEY or not MAILERLITE_GROUP_ID:
        print("MailerLite not configured")
        return
    
    url = f"https://api.mailerlite.com/api/v2/groups/{MAILERLITE_GROUP_ID}/subscribers"
    
    headers = {
        "X-MailerLite-ApiKey": MAILERLITE_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "email": email,
        "name": name,
        "fields": {
            "score": analysis['score'],
            "current_perception": analysis['current_perception'],
            "analyzed_at": datetime.now().isoformat()
        },
        "resubscribe": False,
        "autoresponders": True
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"MailerLite: {response.status_code}")
    except Exception as e:
        print(f"MailerLite Error: {str(e)}")

def log_analysis(data, analysis):
    """حفظ التحليل في ملف"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "email": data['email'],
        "goal": data['goal'],
        "score": analysis['score'],
        "current_perception": analysis['current_perception']
    }
    
    with open('analysis_log.json', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

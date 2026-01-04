from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

# Hugging Face API Key
HF_API_KEY = os.environ.get('HF_API_KEY')

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
        
    except ValueError as e:
        # Specific errors (API key, parsing, etc.)
        print(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        # Generic errors
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def analyze_with_groq(headline, about, goal):
    """تحليل الملف باستخدام Hugging Face"""
    
    # Check if API key exists
    if not HF_API_KEY:
        raise ValueError("Hugging Face API Key not configured. Please add HF_API_KEY to environment variables.")
    
    try:
        # Hugging Face Inference API endpoint
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        
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
  "score": 45,
  "current_perception": "...",
  "desired_perception": "...",
  "fatal_error": "...",
  "quick_fix": {{
    "before": "...",
    "after": "..."
  }}
}}"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 800,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result_text = response.json()
        
        # Extract JSON from response
        if isinstance(result_text, list) and len(result_text) > 0:
            content = result_text[0].get('generated_text', '')
        else:
            content = str(result_text)
        
        # Try to parse JSON
        # Sometimes the model returns text before/after JSON, so we extract it
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
        
        result = json.loads(content)
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API Error: {str(e)}")
        raise ValueError(f"AI analysis failed: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {str(e)}")
        print(f"Response content: {content}")
        raise ValueError(f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        print(f"Hugging Face API Error: {str(e)}")
        raise ValueError(f"AI analysis failed: {str(e)}")

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


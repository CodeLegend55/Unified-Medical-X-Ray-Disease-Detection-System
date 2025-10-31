# 🤖 Google Gemini API Setup Guide

This guide will help you set up Google Gemini API for AI-powered medical report generation in the Unified Medical X-Ray Analysis application.

## 📋 Table of Contents
- [Why Use Gemini API?](#why-use-gemini-api)
- [Getting Your API Key](#getting-your-api-key)
- [Configuration](#configuration)
- [Installation](#installation)
- [Choosing Between APIs](#choosing-between-apis)
- [Troubleshooting](#troubleshooting)

## 🌟 Why Use Gemini API?

Google Gemini offers several advantages for medical report generation:

✅ **Superior Performance**: Gemini-1.5-Pro offers state-of-the-art natural language generation
✅ **Medical Knowledge**: Trained on vast medical literature and clinical documentation
✅ **Faster Response**: Generally faster than Hugging Face inference API
✅ **Generous Free Tier**: Free tier includes 15 requests per minute
✅ **Better Context Understanding**: Excellent at understanding complex medical scenarios
✅ **Easy Integration**: Simple Python SDK with reliable performance

## 🔑 Getting Your API Key

### Step 1: Sign in to Google AI Studio

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. If prompted, accept the Terms of Service

### Step 2: Create an API Key

1. Click on **"Get API Key"** or **"Create API Key"**
2. Select a Google Cloud project or create a new one
3. Click **"Create API Key in new project"** (recommended for first-time users)
4. Your API key will be generated and displayed

### Step 3: Copy Your API Key

1. Click the **Copy** button next to your API key
2. ⚠️ **IMPORTANT**: Store this key securely - treat it like a password!
3. Never commit API keys to version control (Git/GitHub)

## ⚙️ Configuration

### Method 1: Update config.py (Recommended)

1. Open `config.py` in your project
2. Find the Gemini configuration section
3. Replace `"API_KEY_HERE"` with your actual API key:

```python
# ───────────────────────────────────────────────────────────────────────────────
# Google Gemini API Configuration (Optional)
# ───────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "your-actual-gemini-api-key-here"  # Replace this!

# Gemini Model Configuration
GEMINI_MODEL = "gemini-1.5-pro"  # Recommended
```

4. Set the report generation API to Gemini:

```python
# Report Generation API Selection
REPORT_API = "gemini"  # Change this to "gemini"
```

### Method 2: Environment Variables (More Secure)

For production deployments, use environment variables:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then modify `config.py` to read from environment:
```python
import os
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'API_KEY_HERE')
```

## 📦 Installation

Install the required Google Generative AI package:

**Using pip:**
```bash
pip install google-generativeai
```

**Or add to requirements.txt:**
```
google-generativeai>=0.3.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🔄 Choosing Between APIs

You can choose between three report generation methods in `config.py`:

### 1. Google Gemini (Recommended) ⭐

```python
REPORT_API = "gemini"
GEMINI_API_KEY = "your-gemini-key"
GEMINI_MODEL = "gemini-1.5-pro"
```

**Pros:**
- State-of-the-art performance
- Fast response times
- Excellent medical reasoning
- Generous free tier (15 RPM)
- Easy to use

**Best for:** Production use, high-quality reports

### 2. Hugging Face

```python
REPORT_API = "huggingface"
HUGGINGFACE_API_KEY = "your-hf-key"
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
```

**Pros:**
- Access to many open-source models
- Good for experimentation
- Self-hostable models

**Best for:** Research, custom model deployment

### 3. Fallback (No API Required)

```python
REPORT_API = "fallback"
```

**Pros:**
- No API key needed
- Always available
- Fast and reliable

**Cons:**
- Template-based (less detailed)
- No natural language generation

**Best for:** Testing, offline use, demo purposes

## 🏥 Model Selection

Choose the best Gemini model for your needs:

### Gemini-1.5-Pro (Recommended) ⭐
```python
GEMINI_MODEL = "gemini-1.5-pro"
```
- Most capable model
- Best for complex medical reasoning
- Excellent report quality
- **Free tier: 15 requests/minute**

### Gemini-1.5-Flash (Faster)
```python
GEMINI_MODEL = "gemini-1.5-flash"
```
- Faster response time
- Good balance of speed and quality
- Lower cost
- **Free tier: 15 requests/minute**

### Gemini-Pro (Previous Generation)
```python
GEMINI_MODEL = "gemini-pro"
```
- Still very capable
- Stable and reliable
- Good fallback option

## 🔍 Verification

After configuration, verify your setup:

1. **Start the application:**
```bash
python app.py
```

2. **Check the startup logs:**
```
🔧 Report Generation API: GEMINI
──────────────────────────────────────────────────────────────────────
✓ Google Gemini API is valid and accessible
  Model: gemini-1.5-pro
──────────────────────────────────────────────────────────────────────
```

3. **Check the health endpoint:**
```bash
# PowerShell
Invoke-RestMethod http://localhost:5000/health

# Or visit in browser
http://localhost:5000/health
```

4. **Check the API status endpoint:**
```bash
http://localhost:5000/api/status
```

You should see:
```json
{
  "selected_api": "gemini",
  "available": true,
  "status_message": "API key is valid and model is accessible",
  "report_mode": "AI-Generated (Google Gemini)",
  "model_name": "gemini-1.5-pro",
  "api_configured": true
}
```

## ⚠️ Troubleshooting

### Issue: "API key is invalid"

**Solution:**
1. Verify your API key is correct (no extra spaces)
2. Check if the API key is active in [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Try regenerating the API key

### Issue: "google-generativeai package not installed"

**Solution:**
```bash
pip install google-generativeai
```

### Issue: "Model not found"

**Solution:**
- Verify the model name is correct
- Use `"gemini-1.5-pro"` or `"gemini-1.5-flash"` (case-sensitive)
- Check [Google AI Studio](https://ai.google.dev/models/gemini) for available models

### Issue: "Rate limit exceeded"

**Solution:**
- Free tier: 15 requests per minute
- Wait a minute before trying again
- Consider upgrading to paid tier for higher limits
- Add rate limiting in your application

### Issue: "API quota exceeded"

**Solution:**
1. Check your quota at [Google Cloud Console](https://console.cloud.google.com/)
2. Free tier resets monthly
3. Upgrade to paid tier if needed

### Issue: Application falls back to template

**Solution:**
1. Check `config.py`: Ensure `REPORT_API = "gemini"`
2. Verify API key is set correctly
3. Check startup logs for error messages
4. Run validation:
   ```python
   from app import validate_gemini_api
   status = validate_gemini_api()
   print(status)
   ```

## 📊 API Comparison

| Feature | Gemini | Hugging Face | Fallback |
|---------|--------|--------------|----------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Free Tier | 15 RPM | Limited | Unlimited |
| Setup | Easy | Moderate | None |
| Medical Knowledge | Excellent | Good | Basic |
| Cost | Free/Paid | Free/Paid | Free |

## 🔒 Security Best Practices

1. **Never commit API keys to Git:**
   ```bash
   # Add to .gitignore
   config.py
   .env
   ```

2. **Use environment variables in production:**
   ```python
   GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
   ```

3. **Rotate API keys regularly**

4. **Monitor API usage** at [Google Cloud Console](https://console.cloud.google.com/)

5. **Set up billing alerts** to avoid unexpected charges

## 📚 Additional Resources

- [Google AI Studio](https://makersuite.google.com/app/apikey) - Get API keys
- [Gemini API Documentation](https://ai.google.dev/docs) - Official docs
- [Python SDK Documentation](https://ai.google.dev/tutorials/python_quickstart) - Python guide
- [Pricing Information](https://ai.google.dev/pricing) - Pricing details
- [Model Comparison](https://ai.google.dev/models/gemini) - Model specs

## 🎯 Quick Start Summary

1. Get API key: https://makersuite.google.com/app/apikey
2. Install package: `pip install google-generativeai`
3. Update `config.py`:
   ```python
   REPORT_API = "gemini"
   GEMINI_API_KEY = "your-key-here"
   GEMINI_MODEL = "gemini-1.5-pro"
   ```
4. Run: `python app.py`
5. Visit: http://localhost:5000

## 💡 Tips

- Start with `gemini-1.5-pro` for best results
- Switch to `gemini-1.5-flash` if you need faster responses
- Use `fallback` mode for testing without using API quota
- Monitor your usage to stay within free tier limits
- The application automatically falls back to template mode if API fails

## ❓ Need Help?

If you encounter issues:
1. Check the startup logs for error messages
2. Visit `/api/status` endpoint to check API status
3. Review this guide's troubleshooting section
4. Check the [Google AI documentation](https://ai.google.dev/docs)
5. Ensure your Google Cloud project has the Generative AI API enabled

---

**Happy medical imaging analysis! 🏥🤖**

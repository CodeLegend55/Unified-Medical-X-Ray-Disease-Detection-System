# Hugging Face API Setup Guide

This guide explains how to set up and use Hugging Face's Inference API for generating medical reports in the Unified Medical X-Ray Disease Detection System.

## Why Hugging Face Instead of OpenAI?

- **Free Tier Available**: Hugging Face offers a free tier for API usage
- **Open Source Models**: Access to various open-source medical and general language models
- **Flexibility**: Choose from different models based on your needs
- **Privacy**: Can use locally hosted models if needed

## Setup Instructions

### 1. Get Your Hugging Face API Key

1. Visit [Hugging Face](https://huggingface.co/) and create a free account
2. Go to your [Access Tokens page](https://huggingface.co/settings/tokens)
3. Click "New token"
4. Give it a name (e.g., "medical-report-generator")
5. Select "Read" access
6. Click "Generate token"
7. Copy your token (it looks like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### 2. Configure the Application

1. Open `config.py` in your project
2. Replace the placeholder with your actual API key:
   ```python
   HUGGINGFACE_API_KEY = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

### 3. Choose Your Model (Optional)

The default model is `mistralai/Mistral-7B-Instruct-v0.2`, which is excellent for medical text generation.

You can change the model in `config.py` by modifying:
```python
HUGGINGFACE_MODEL = "your-preferred-model-here"
```

#### Recommended Models:

1. **Mistral-7B-Instruct** (Default - Best Overall)
   - Model: `mistralai/Mistral-7B-Instruct-v0.2`
   - Pros: Excellent instruction following, good medical knowledge
   - Free tier: Yes

2. **Llama-2-7B-Chat**
   - Model: `meta-llama/Llama-2-7b-chat-hf`
   - Pros: Strong general knowledge, good formatting
   - Note: May require agreement to Meta's license

3. **Flan-T5-Large**
   - Model: `google/flan-t5-large`
   - Pros: Lightweight, fast, instruction-tuned
   - Cons: Less detailed outputs

4. **BioGPT-Large** (Specialized Medical Model)
   - Model: `microsoft/BioGPT-Large`
   - Pros: Specifically trained on biomedical text
   - Note: May require different prompt formatting

### 4. Install Dependencies

Install the required Hugging Face packages:

```bash
pip install huggingface-hub>=0.20.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 5. Test the Setup

Run the Flask application:
```bash
python app.py
```

You should see:
```
🔗 Hugging Face API: Available
   Model: mistralai/Mistral-7B-Instruct-v0.2
```

## Usage

Once configured, the application will automatically use Hugging Face to generate medical reports when you upload X-ray images.

### API Rate Limits

Hugging Face free tier has rate limits:
- **Free tier**: ~1000 requests per day
- **Inference API**: May have concurrent request limits

If you exceed limits, the application will fall back to the built-in report generator.

### Fallback Behavior

The application has three levels:
1. **Primary**: Hugging Face API (if configured)
2. **Fallback**: Built-in comprehensive report generator
3. **Error Handling**: Detailed error messages if both fail

## Troubleshooting

### Error: "Hugging Face Hub package not available"
**Solution**: Install huggingface-hub
```bash
pip install huggingface-hub
```

### Error: "API rate limit exceeded"
**Solution**: 
- Wait for rate limit to reset (usually 24 hours for free tier)
- Upgrade to Hugging Face Pro for higher limits
- The app will use fallback reports automatically

### Error: "Model not found"
**Solution**: 
- Verify the model name is correct
- Some models require accepting a license on Hugging Face website
- Try a different model from the recommended list

### Error: "Invalid token"
**Solution**:
- Verify your API key is correctly copied
- Ensure no extra spaces in config.py
- Generate a new token if needed

### Reports are too short or incomplete
**Solution**: Adjust parameters in `app.py`:
```python
response = hf_client.text_generation(
    prompt,
    max_new_tokens=1200,  # Increase this
    temperature=0.4,      # Adjust creativity
    top_p=0.95,          # Adjust diversity
)
```

## Advanced Configuration

### Using Local Models (No API Key Needed)

For privacy or offline use, you can run models locally using `transformers`:

```python
from transformers import pipeline

# In app.py, modify the import section:
generator = pipeline('text-generation', model='mistralai/Mistral-7B-Instruct-v0.2')

# Then use it in generate_huggingface_report:
response = generator(prompt, max_new_tokens=800, temperature=0.3)
```

**Note**: This requires significant RAM/VRAM (8GB+ recommended for 7B models)

### Custom Prompts

You can customize the prompt format in `app.py` by modifying the `generate_huggingface_report` function. Different models may work better with different prompt formats.

## Cost Comparison

| Provider | Free Tier | Cost (if exceeding free tier) |
|----------|-----------|-------------------------------|
| Hugging Face | 1000 req/day | ~$0.001 per request |
| OpenAI GPT-3.5 | None | ~$0.002 per request |
| OpenAI GPT-4 | None | ~$0.03 per request |

## Security Best Practices

1. **Never commit API keys to Git**:
   - Add `config.py` to `.gitignore` if it contains real keys
   - Use environment variables for production

2. **Environment Variables** (Production):
   ```python
   import os
   HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', 'your-huggingface-api-key-here')
   ```

3. **Rotate Keys Regularly**: Generate new tokens periodically

## Support

- Hugging Face Documentation: https://huggingface.co/docs
- Inference API Docs: https://huggingface.co/docs/api-inference/
- Model Hub: https://huggingface.co/models

## License

Ensure you comply with the license terms of any model you use:
- Mistral: Apache 2.0
- Llama 2: Meta's Community License
- Flan-T5: Apache 2.0
- BioGPT: MIT License

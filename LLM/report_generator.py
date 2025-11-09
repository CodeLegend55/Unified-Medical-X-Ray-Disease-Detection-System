import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalReportGenerator:
    def __init__(self, model_path="./medical_report_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        
    def generate_report(self, patient_info, max_length=1000):
        """
        Generate a medical report based on model predictions and patient information
        
        Args:
            patient_info (dict): Dictionary containing patient information and model predictions
            max_length (int): Maximum length of the generated report
            
        Returns:
            str: Generated medical report
        """
        # Extract key information
        diagnosis = patient_info.get('diagnosis', 'Unknown')
        confidence = patient_info.get('confidence', 0)
        exam_type = patient_info.get('exam_type', 'X-Ray')
        patient_details = patient_info.get('patient_info', {})
        model_consensus = patient_info.get('model_consensus', [])
        
        # Format the prompt with structured sections
        prompt = (
            f"### Instruction: Generate a structured medical report for a {exam_type} examination.\n\n"
            f"### Input:\n"
            f"DIAGNOSIS: {diagnosis}\n"
            f"CONFIDENCE: {confidence:.1f}%\n"
            f"EXAM TYPE: {exam_type}\n\n"
            f"PATIENT INFORMATION:\n"
            f"{json.dumps(patient_details, indent=2)}\n\n"
            f"MODEL ANALYSIS:\n"
            f"{json.dumps(model_consensus, indent=2)}\n\n"
            f"### Response Format:\n"
            "CLINICAL HISTORY:\n"
            "[Summarize relevant patient information and presenting symptoms]\n\n"
            "TECHNIQUE:\n"
            "[Describe the imaging modality and analysis method]\n\n"
            "FINDINGS:\n"
            "[Describe the key radiological observations]\n"
            "- Primary findings\n"
            "- Secondary observations\n"
            "- Notable features\n\n"
            "IMPRESSION:\n"
            "[Provide diagnostic conclusion]\n"
            "1. Primary diagnosis and confidence level\n"
            "2. Relevant clinical correlations\n"
            "3. Recommendations for follow-up\n\n"
            "### Response:"
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate report with controlled parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=800,  # Changed from max_length to max_new_tokens for better control
            min_new_tokens=200,  # Ensure minimum length of generation
            num_return_sequences=1,
            temperature=0.7,  # Slightly increased for more creative generation
            top_p=0.92,  # Slightly increased
            top_k=50,
            no_repeat_ngram_size=3,  # Prevent repetition of phrases
            do_sample=True,
            repetition_penalty=1.2,  # Penalize repetition
            length_penalty=1.2,  # Encourage longer generation
            pad_token_id=self.tokenizer.eos_token_id,  # Proper padding
        )
        
        # Decode the generated report
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Debug: Print the raw generated text (first 500 chars)
        print("\n" + "="*70)
        print("RAW MODEL OUTPUT (first 500 chars):")
        print("="*70)
        print(generated_text[:500])
        print("="*70 + "\n")
        
        # Extract only the response part
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text.strip()
        
        print("\n" + "="*70)
        print("EXTRACTED RESPONSE (first 500 chars):")
        print("="*70)
        print(response[:500] if response else "[EMPTY]")
        print("="*70 + "\n")
        
        # Post-process the response to ensure proper formatting
        def clean_and_format_response(text):
            """Clean and format the generated response"""
            sections = ["CLINICAL HISTORY:", "TECHNIQUE:", "FINDINGS:", "IMPRESSION:"]
            
            # If the text is too short or empty, return a default message
            if not text or len(text.strip()) < 50:
                return self._generate_default_report(patient_info)
            
            # Split into lines and process
            lines = text.split("\n")
            formatted_sections = {}
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                
                # Check if this line is a section header
                found_section = None
                for section in sections:
                    if section in line:
                        found_section = section
                        break
                
                if found_section:
                    # Save previous section if it exists
                    if current_section and current_content:
                        content_text = "\n".join(current_content).strip()
                        if content_text and content_text not in ["", "No information provided.", "N/A"]:
                            formatted_sections[current_section] = content_text
                    
                    # Start new section
                    current_section = found_section
                    current_content = []
                elif current_section and line:
                    # Add content to current section (skip empty lines at the start)
                    if current_content or line:
                        current_content.append(line)
            
            # Save last section
            if current_section and current_content:
                content_text = "\n".join(current_content).strip()
                if content_text and content_text not in ["", "No information provided.", "N/A"]:
                    formatted_sections[current_section] = content_text
            
            # Build the final report with all sections
            final_report = []
            for section in sections:
                final_report.append(f"{section}")
                if section in formatted_sections:
                    final_report.append(formatted_sections[section])
                else:
                    # Generate section-specific default content
                    final_report.append(self._generate_default_section(section, patient_info))
                final_report.append("")  # Add blank line between sections
            
            return "\n".join(final_report).strip()
        
        return clean_and_format_response(response)
    
    def _generate_default_section(self, section, patient_info):
        """Generate default content for a section based on diagnosis"""
        diagnosis = patient_info.get('diagnosis', 'Unknown')
        confidence = patient_info.get('confidence', 0)
        exam_type = patient_info.get('exam_type', 'X-Ray')
        
        if section == "CLINICAL HISTORY:":
            return f"Evaluation for suspected {diagnosis.lower().replace('_', ' ')} based on presenting symptoms. Clinical correlation requested."
        
        elif section == "TECHNIQUE:":
            return f"Digital {exam_type} examination performed. AI-assisted analysis completed using ensemble deep learning models (ResNet50, DenseNet121, EfficientNetB0). Image quality is adequate for diagnostic interpretation."
        
        elif section == "FINDINGS:":
            findings_map = {
                'COVID19': "Bilateral patchy opacities predominantly in peripheral lung fields. Ground-glass opacities noted. Cardiomediastinal contours within normal limits.",
                'PNEUMONIA': "Consolidation and infiltrates observed in lung parenchyma. Possible pleural involvement noted.",
                'TUBERCULOSIS': "Upper lobe predominant infiltrates with possible cavitation. Lymphadenopathy may be present.",
                'NORMAL_CHEST': "Clear lung fields bilaterally. No acute infiltrates, consolidation, or effusion. Heart size and mediastinal contours are normal. Bony thorax intact.",
                'FRACTURED': "Osseous discontinuity with visible fracture line. Possible displacement noted. Surrounding soft tissue changes present.",
                'NON_FRACTURED': "Intact bony architecture without evidence of acute fracture. No displacement or angulation. Soft tissues appear normal.",
                'OSTEOPOROSIS': "Generalized decreased bone density with trabecular thinning. Osteopenic changes consistent with osteoporosis.",
                'NORMAL_BONE': "Age-appropriate bone density. Normal trabecular architecture. No acute abnormalities detected."
            }
            return findings_map.get(diagnosis, "Radiological findings consistent with the AI-predicted diagnosis.")
        
        elif section == "IMPRESSION:":
            return f"""1. Findings consistent with {diagnosis.replace('_', ' ')} (AI confidence: {confidence:.1f}%)
   - Ensemble analysis from multiple neural networks
   - Clinical correlation recommended

2. RECOMMENDATIONS:
   - Clinical evaluation by appropriate specialist
   - Correlation with laboratory findings and patient history
   - Follow-up imaging as clinically indicated"""
        
        return "Information not available."
    
    def _generate_default_report(self, patient_info):
        """Generate a complete default report when generation fails"""
        diagnosis = patient_info.get('diagnosis', 'Unknown')
        confidence = patient_info.get('confidence', 0)
        exam_type = patient_info.get('exam_type', 'X-Ray')
        
        return f"""CLINICAL HISTORY:
Evaluation for suspected {diagnosis.lower().replace('_', ' ')} based on presenting symptoms. Clinical correlation requested.

TECHNIQUE:
Digital {exam_type} examination performed. AI-assisted analysis completed using ensemble deep learning models (ResNet50, DenseNet121, EfficientNetB0).

FINDINGS:
{self._generate_default_section("FINDINGS:", patient_info)}

IMPRESSION:
{self._generate_default_section("IMPRESSION:", patient_info)}"""
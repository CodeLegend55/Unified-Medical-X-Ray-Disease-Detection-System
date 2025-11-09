import json
import pandas as pd

def create_medical_report_dataset():
    """
    Create a dataset for fine-tuning the LLM model with medical report examples.
    Format: { "instruction": "...", "input": "...", "output": "..." }
    """
    dataset = []
    
    # Example template for medical reports
    conditions = {
        "COVID19": {
            "symptoms": ["fever", "cough", "shortness of breath", "fatigue"],
            "recommendations": ["isolation", "rest", "hydration", "monitor oxygen levels"]
        },
        "PNEUMONIA": {
            "symptoms": ["cough with phlegm", "chest pain", "fever", "difficulty breathing"],
            "recommendations": ["antibiotics", "rest", "hydration", "follow-up chest x-ray"]
        },
        "TUBERCULOSIS": {
            "symptoms": ["persistent cough", "weight loss", "night sweats", "fatigue"],
            "recommendations": ["complete TB treatment course", "isolation", "regular check-ups"]
        },
        "FRACTURED": {
            "symptoms": ["pain", "swelling", "difficulty moving", "visible deformity"],
            "recommendations": ["immobilization", "pain management", "orthopedic consultation"]
        },
        "OSTEOPOROSIS": {
            "symptoms": ["bone pain", "height loss", "stooped posture", "easily fractured bones"],
            "recommendations": ["calcium supplements", "vitamin D", "exercise", "fall prevention"]
        }
    }
    
    # Generate training examples
    for condition, details in conditions.items():
        # Multiple variations of patient profiles
        ages = ["25", "35", "45", "55", "65", "75"]
        severities = ["mild", "moderate", "severe"]
        
        for age in ages:
            for severity in severities:
                # Create input with varying patient information
                patient_info = {
                    "age": age,
                    "condition": condition,
                    "severity": severity,
                    "symptoms": ", ".join(details["symptoms"]),
                    "model_prediction": condition
                }
                
                # Create instruction
                instruction = "Generate a detailed medical report based on the X-ray findings and patient information."
                
                # Create comprehensive medical report
                report = f"""Medical Report
                
Patient Age: {age}
Primary Diagnosis: {condition}
Severity: {severity}

Clinical Findings:
- X-ray Analysis: Consistent with {condition}
- Reported Symptoms: {", ".join(details["symptoms"])}

Recommendations:
{chr(10).join(['- ' + rec for rec in details["recommendations"]])}

Follow-up:
- Schedule follow-up appointment in 2 weeks
- Continue monitoring symptoms
- Seek immediate medical attention if conditions worsen

Additional Notes:
- Patient education provided about the condition
- Treatment plan discussed and agreed upon
"""
                
                # Add to dataset
                dataset.append({
                    "instruction": instruction,
                    "input": json.dumps(patient_info),
                    "output": report
                })
    
    # Save dataset to file
    with open("medical_report_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

if __name__ == "__main__":
    create_medical_report_dataset()
import torch
import logging
import sys
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
from tqdm import tqdm
import psutil

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def prepare_dataset():
    """
    Load and prepare the dataset for training
    """
    try:
        # Check if dataset file exists
        if not os.path.exists("medical_report_dataset.json"):
            logger.error("medical_report_dataset.json not found. Please run train_data.py first.")
            sys.exit(1)
            
        # Load the dataset
        logger.info("Loading dataset from medical_report_dataset.json...")
        with open("medical_report_dataset.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        
        # Split into train and validation
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_dict({
            'instruction': [d['instruction'] for d in train_data],
            'input': [d['input'] for d in train_data],
            'output': [d['output'] for d in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'instruction': [d['instruction'] for d in val_data],
            'input': [d['input'] for d in val_data],
            'output': [d['output'] for d in val_data]
        })
        
        logger.info(f"Dataset prepared: {len(train_data)} training examples, {len(val_data)} validation examples")
        
        return {
            "train": train_dataset,
            "validation": val_dataset
        }
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def format_prompt(example):
    """Format the prompt for training"""
    return f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Response: {example['output']}

"""

def train_model():
    """
    Train the model using the prepared dataset
    """
    try:
        logger.info("Starting model training process...")
        
        # Check for CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model and tokenizer
        # Use a smaller model for CPU training
        model_name = "facebook/opt-125m"  # Much smaller model that can run on CPU
        logger.info(f"Loading model and tokenizer from {model_name}...")
        
        # Load tokenizer with padding token
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=512,
            padding_side="right",
            truncation_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with optimized settings for CPU and use safetensors
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            use_safetensors=True,  # Force use of safetensors
            device_map=None  # Don't use device map for small model
        ).to(device)
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset = prepare_dataset()
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Minimal batch size for CPU
            per_device_eval_batch_size=1,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,  # Keep only best model
            gradient_accumulation_steps=8,  # Increased for stability
            fp16=False,  # Disable mixed precision for CPU
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            learning_rate=5e-5,  # Slightly higher learning rate
            max_grad_norm=0.5,  # Gradient clipping
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            no_cuda=True,  # Force CPU usage
            seed=42
        )
        
        # Tokenization function
        def tokenize_function(examples):
            # Combine instruction, input, and output with clear separators
            full_prompts = []
            for instruction, input_text, output in zip(
                examples['instruction'], 
                examples['input'], 
                examples['output']
            ):
                # Convert input dict to string if necessary
                if isinstance(input_text, dict):
                    input_text = json.dumps(input_text)
                
                # Format prompt
                prompt = f"### Instruction: {instruction}\n\n### Input: {input_text}\n\n### Response: {output}\n\nEND"
                full_prompts.append(prompt)
            
            # Tokenize with padding and truncation
            tokenized = tokenizer(
                full_prompts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None  # Return python lists
            )
            
            # Create attention masks
            attention_mask = tokenized["attention_mask"]
            
            # Create labels (shift tokens for language modeling)
            labels = []
            for i in range(len(tokenized["input_ids"])):
                label = tokenized["input_ids"][i][:]
                label = [-100] * len(label)  # Initialize all as -100 (ignored)
                # Only compute loss on the response part
                response_start = full_prompts[i].find("### Response:") + len("### Response:")
                response_text = full_prompts[i][response_start:]
                response_tokens = tokenizer(response_text, 
                                         truncation=True, 
                                         max_length=512)["input_ids"]
                label[-len(response_tokens):] = response_tokens
                labels.append(label)
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        train_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        val_dataset = dataset["validation"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("Saving model and tokenizer...")
        output_dir = "./medical_report_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("=== Starting Medical Report Generator Training ===")
        
        # Create necessary directories
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./medical_report_model", exist_ok=True)
        
        # Train the model
        success = train_model()
        
        if success:
            logger.info("=== Training completed successfully! ===")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
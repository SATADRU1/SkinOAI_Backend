import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TextModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        try:
            logger.info(f"Using device: {self.device}")
            logger.info("Loading TinyLlama model...")
            
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            self.model.eval()
            logger.info("TinyLlama model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama model: {e}")
            self.model = None
            self.tokenizer = None

    def generate_text(self, disease_pred: str, desc: str) -> str:
        """Generate a treatment plan using TinyLlama."""
        
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.warning("TinyLlama model not loaded, returning fallback response")
            return self._get_fallback_response(disease_pred, desc)
        
        try:
            # Construct the prompt
            prompt = (
                f"<|system|>\n"
                f"You are a medical assistant providing treatment advice for skin conditions.\n"
                f"<|user|>\n"
                f"Diagnosed condition: {disease_pred}\n"
                f"Patient symptoms: {desc if desc else 'No additional symptoms reported'}\n\n"
                f"Please provide a brief treatment plan with 3-4 recommendations.\n"
                f"<|assistant|>\n"
            )

            # Tokenize and generate response
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    min_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
            
            # Add disclaimer if not present
            if "dermatologist" not in response.lower():
                response += "\n\nDisclaimer: Please consult with a dermatologist for proper diagnosis and treatment."
            
            return response if response else self._get_fallback_response(disease_pred, desc)

        except Exception as e:
            logger.error(f"Error generating text with TinyLlama: {e}")
            return self._get_fallback_response(disease_pred, desc)
    
    def _get_fallback_response(self, disease_pred: str, desc: str) -> str:
        """Generate a comprehensive fallback response with specific medical advice"""
        
        # Specific treatment recommendations based on condition
        treatments = {
            "Eczema": [
                "Apply fragrance-free moisturizer immediately after bathing",
                "Use mild, soap-free cleansers",
                "Consider over-the-counter hydrocortisone cream for inflammation",
                "Identify and avoid known triggers (stress, allergens, irritants)",
                "Wear soft, breathable cotton clothing"
            ],
            "Atopic Dermatitis": [
                "Maintain consistent skincare routine with hypoallergenic products",
                "Apply topical corticosteroids as directed by healthcare provider",
                "Use lukewarm water for bathing and limit bath time",
                "Consider antihistamines for itching relief",
                "Implement stress management techniques"
            ],
            "Psoriasis": [
                "Apply moisturizers containing ceramides or urea",
                "Consider coal tar preparations for scaling",
                "Use salicylic acid products to remove scales",
                "Maintain healthy lifestyle with regular exercise",
                "Limit alcohol consumption and manage stress"
            ],
            "Acne/Pimples": [
                "Use gentle, non-comedogenic cleansers twice daily",
                "Apply benzoyl peroxide or salicylic acid treatments",
                "Avoid touching or picking at affected areas",
                "Use oil-free, non-comedogenic moisturizers",
                "Consider retinoid treatments for persistent cases"
            ],
            "Melanoma": [
                "Seek immediate dermatological evaluation",
                "Perform regular self-examinations using ABCDE criteria",
                "Use broad-spectrum SPF 30+ sunscreen daily",
                "Avoid UV exposure during peak hours (10 AM - 4 PM)",
                "Consider professional skin mapping and monitoring"
            ],
            "Basal Cell Carcinoma": [
                "Schedule prompt dermatological consultation",
                "Protect area from further sun exposure",
                "Use broad-spectrum sunscreen with SPF 30 or higher",
                "Avoid picking or scratching the affected area",
                "Consider Mohs surgery for complete removal"
            ],
            "Squamous Cell Carcinoma": [
                "Seek immediate medical evaluation",
                "Protect from UV radiation with clothing and sunscreen",
                "Avoid immunosuppressive factors when possible",
                "Monitor for signs of growth or changes",
                "Discuss treatment options including surgical removal"
            ],
            "Nevus": [
                "Monitor for changes in size, color, or texture",
                "Perform monthly self-examinations",
                "Use sun protection to prevent changes",
                "Schedule annual dermatological screenings",
                "Document appearance with photos for comparison"
            ],
            "Seborrheic Keratosis": [
                "No treatment required unless cosmetically bothersome",
                "Avoid irritation from clothing or jewelry",
                "Monitor for unusual changes in appearance",
                "Consider removal if frequently irritated",
                "Protect from sun exposure to prevent new lesions"
            ],
            "Warts Molluscum": [
                "Avoid touching or scratching to prevent spread",
                "Keep affected area clean and dry",
                "Consider over-the-counter wart treatments with salicylic acid",
                "Boost immune system with proper nutrition and rest",
                "Practice good hygiene to prevent transmission"
            ]
        }
        
        # Get specific recommendations or use general ones
        specific_treatments = treatments.get(disease_pred, [
            "Keep the affected area clean and dry",
            "Apply gentle, fragrance-free moisturizer",
            "Avoid scratching or irritating the area",
            "Monitor for changes in appearance",
            "Protect from sun exposure with appropriate clothing and sunscreen"
        ])
        
        # Generate comprehensive response
        response = f"Professional Treatment Recommendations for {disease_pred}:\n\n"
        
        for i, treatment in enumerate(specific_treatments, 1):
            response += f"{i}. {treatment}\n"
        
        # Add lifestyle recommendations
        response += "\nGeneral Skin Health Guidelines:\n"
        response += "• Maintain a balanced diet rich in vitamins A, C, and E\n"
        response += "• Stay hydrated with adequate water intake\n"
        response += "• Get sufficient sleep for skin repair and regeneration\n"
        response += "• Manage stress through relaxation techniques\n"
        
        # Add important disclaimer
        response += "\n⚠️ IMPORTANT MEDICAL DISCLAIMER:\n"
        response += "This AI-generated advice is for informational purposes only and should not replace professional medical consultation. "
        response += "Please consult with a board-certified dermatologist for accurate diagnosis, personalized treatment plans, and proper medical supervision. "
        response += "Seek immediate medical attention if you notice rapid changes, bleeding, or concerning symptoms."
        
        return response

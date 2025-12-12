import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}
        print(f"Device selected: {self.device}")

    def load_model(self, model_name):
        """
        Loads the model and processor based on the model_name.
        implements caching to avoid reloading.
        """
        if model_name in self.models:
            return self.models[model_name], self.processors[model_name]

        print(f"Loading model: {model_name}...")
        
        if "blip" in model_name.lower():
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        elif "vit-gpt2" in model_name.lower():
            processor = ViTImageProcessor.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            processor = (processor, tokenizer)
        
        else:
            raise ValueError(f"Model {model_name} not supported.")

        self.models[model_name] = model
        self.processors[model_name] = processor
        print(f"Model {model_name} loaded successfully.")
        
        return model, processor
import torch
from PIL import Image

class CaptionGenerator:
    def __init__(self, model, processor, device=None):
        self.model = model
        self.processor = processor
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def generate_caption(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        if "Blip" in type(self.model).__name__:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption

        elif "VisionEncoderDecoder" in type(self.model).__name__:
            feature_extractor, tokenizer = self.processor
            
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return preds[0].strip()

        else:
            return "Error: Unknown model architecture."
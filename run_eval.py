import sys
import os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_loader import ModelLoader
from model.caption_generator import CaptionGenerator
from evaluation.metrics import MetricsCalculator

def load_flickr8k_data(data_dir, limit=None):
    captions_file = os.path.join(data_dir, "captions.txt")
    images_dir = os.path.join(data_dir, "Images")

    if not os.path.exists(captions_file) or not os.path.exists(images_dir):
        print(f"❌ Error: Could not find Flickr8k files in {data_dir}")
        print("Ensure you have 'captions.txt' and an 'Images/' folder.")
        sys.exit(1)

    print("📖 Reading captions.txt...")
    df = pd.read_csv(captions_file)
    df.columns = [c.strip() for c in df.columns]
    grouped = df.groupby('image')['caption'].apply(list).reset_index()

    dataset = []
    for idx, row in grouped.iterrows():
        if limit and idx >= limit:
            break
            
        img_name = row['image']
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            dataset.append({
                "image_path": img_path,
                "captions": row['caption']
            })
    
    return dataset

def run_flickr_evaluation(model_name, data_dir, limit=50):
    data = load_flickr8k_data(data_dir, limit)
    print(f"✅ Loaded {len(data)} images for evaluation.")

    loader = ModelLoader()
    model, processor = loader.load_model(model_name)
    generator = CaptionGenerator(model, processor)

    predictions = []
    references = []

    print(f"🚀 Generating captions for {len(data)} images...")
    
    for item in tqdm(data):
        try:
            image = Image.open(item['image_path'])
            pred = generator.generate_caption(image)
            
            predictions.append(pred)
            references.append(item['captions']) 
            
        except Exception as e:
            print(f"Error processing {item['image_path']}: {e}")

    print("\nComputing metrics (BLEU, METEOR, ROUGE)...")
    calculator = MetricsCalculator()
    results = calculator.compute_metrics(predictions, references)

    print("\n" + "="*40)
    print(f"📊 Flickr8k Evaluation Report")
    print("="*40)
    print(f"Model:   {model_name}")
    print(f"Samples: {len(data)}")
    print("-" * 20)
    print(f"BLEU:    {results['BLEU']:.4f}")
    print(f"METEOR:  {results['METEOR']:.4f}")
    print(f"ROUGE-L: {results['ROUGE-L']:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--data_dir", type=str, default="data/flicker8k", help="Path to folder containing Images/ and captions.txt")
    parser.add_argument("--limit", type=int, default=50, help="How many images to evaluate (set 0 for all)")
    
    args = parser.parse_args()
    
    base_path = os.path.dirname(os.path.dirname(__file__))
    full_data_path = os.path.join(base_path, args.data_dir)
    limit = args.limit if args.limit > 0 else None

    run_flickr_evaluation(args.model, full_data_path, limit)
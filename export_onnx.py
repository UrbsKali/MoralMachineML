import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Define the model architecture (Must match training)
class TrolleyModel(nn.Module):
    def __init__(self, num_features, num_countries, emb_dim=16):
        super(TrolleyModel, self).__init__()
        
        # Embedding for user country
        self.country_emb = nn.Embedding(num_countries, emb_dim)
        
        # Shared feature extractor (Siamese-like structure)
        # Input dim: num_features + emb_dim
        input_dim = num_features + emb_dim
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
        
    def forward(self, x_a, x_b, country_idx):
        # Get country embedding
        c_emb = self.country_emb(country_idx) # [batch, emb_dim]
        
        # Concatenate country info to both options
        a_input = torch.cat([x_a, c_emb], dim=1)
        b_input = torch.cat([x_b, c_emb], dim=1)
        
        # Compute scores for both options
        score_a = self.feature_net(a_input)
        score_b = self.feature_net(b_input)
        
        # Logits: score_b - score_a
        logits = score_b - score_a
        return logits

def main():
    print("Loading data to determine country mapping...")
    # Load dataset to get countries 
    # Use PairedResponses.csv if available as it's smaller, else SharedResponses
    if os.path.exists('./dataset/PairedResponses.csv'):
        df = pd.read_csv('./dataset/PairedResponses.csv')
        # Based on notebook, we used 'UserCountry3_A' and filled NaNs with 'Unknown'
        # The notebook mentions: paired_df['UserCountry3'] = paired_df['UserCountry3_A'].fillna('Unknown')
        countries = df['UserCountry3_A'].fillna('Unknown').astype(str).unique()
    else:
        # Fallback (might be slow)
        print("PairedResponses.csv not found, checking existing notebook logic...")
        # If we can't find the file, we can't accurately reconstruct indices without the exact encoder state.
        # Assuming the user has run the notebook and the file exists.
        return 

    # Fit encoder
    country_encoder = LabelEncoder()
    country_encoder.fit(countries)
    n_countries = len(country_encoder.classes_)
    
    # Save country mapping
    mapping = {country: int(idx) for country, idx in zip(country_encoder.classes_, country_encoder.transform(country_encoder.classes_))}
    with open('UI/country_mapping.json', 'w') as f:
        json.dump(mapping, f)
    print(f"Country mapping saved to UI/country_mapping.json ({n_countries} countries)")

    # Initialize Model
    # Features = 20 chars + 2 context = 22
    num_features = 22 
    model = TrolleyModel(num_features, n_countries)
    
    # Load Weights
    model_path = 'UI/trolley_model_acc0_7.pth'
    if not os.path.exists(model_path):
        # Check root
        model_path = 'trolley_model_acc0_7.pth'
        
    print(f"Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        # Sometimes keys don't match if saved differently. 
        print(f"Error loading state dict: {e}")
        return

    model.eval()

    # Create dummy input for export
    # x_a: [1, 22], x_b: [1, 22], country: [1]
    dummy_x_a = torch.randn(1, 22)
    dummy_x_b = torch.randn(1, 22)
    dummy_country = torch.tensor([0], dtype=torch.long)

    import onnx
    
    # Export to ONNX
    onnx_path = "UI/model.onnx"
    torch.onnx.export(
        model, 
        (dummy_x_a, dummy_x_b, dummy_country), 
        onnx_path, 
        verbose=False,
        input_names=['input_a', 'input_b', 'country_idx'],
        output_names=['logits'],
        dynamic_axes={'input_a': {0: 'batch_size'}, 'input_b': {0: 'batch_size'}, 'country_idx': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )

    print(f"Checking if model was saved with external data...")
    # Load the model back to ensure it's self-contained
    onnx_model = onnx.load(onnx_path)
    
    # Check if we need to repack
    # We just force save it again without external data (default behavior for small models)
    # This inlines any external data if present
    onnx.save(onnx_model, onnx_path)
    
    # Clean up .data file if it exists, as we want a single file
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"Removed external data file {data_file}, model inlined to {onnx_path}")
    else:
        print(f"Model exported to {onnx_path} (self-contained)")

if __name__ == "__main__":
    main()

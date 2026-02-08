import os
import tempfile
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath

class Predictor(BasePredictor):
    def setup(self):
        """加载模型到内存"""
        import torch
        from demucs.pretrained import get_model
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.models = {}
        
        for model_name in ["htdemucs", "htdemucs_ft", "htdemucs_6s"]:
            try:
                print(f"Loading model: {model_name}...")
                self.models[model_name] = get_model(model_name)
                self.models[model_name].to(self.device)
                self.models[model_name].eval()
                print(f"✓ Loaded model: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        
        print(f"Models loaded: {list(self.models.keys())}")
    
    def predict(
        self,
        audio: CogPath = Input(description="Input audio file (WAV, MP3, FLAC, etc.)"),
        model: str = Input(
            description="Demucs model to use",
            default="htdemucs_ft",
            choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"]
        ),
        stem: str = Input(
            description="Which stem to extract",
            default="vocals",
            choices=["vocals", "drums", "bass", "other", "all"]
        ),
        shifts: int = Input(
            description="Number of random shifts (higher = better quality, slower)",
            default=1,
            ge=0,
            le=10
        ),
    ) -> dict:
        """Run audio source separation"""
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.audio import save_audio
        
        separator = self.models.get(model)
        if separator is None:
            raise ValueError(f"Model {model} not loaded")
        
        print(f"Loading audio: {audio}")
        wav, sr = torchaudio.load(str(audio))
        wav = wav.to(self.device)
        
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        
        wav = wav.unsqueeze(0)
        print(f"Audio shape: {wav.shape}, sample rate: {sr}")
        
        print(f"Separating with {model}, shifts={shifts}...")
        with torch.no_grad():
            sources = apply_model(
                separator, 
                wav, 
                device=self.device,
                shifts=shifts,
                split=True,
                overlap=0.25,
                progress=True
            )
        
        # 确保移到 CPU
        sources = sources.squeeze(0).cpu()
        print(f"Sources device: {sources.device}")
        
        source_names = separator.sources
        print(f"Source names: {source_names}")
        
        output_dir = tempfile.mkdtemp()
        outputs = {}
        
        for i, name in enumerate(source_names):
            if stem != "all" and name != stem:
                continue
            output_path = os.path.join(output_dir, f"{name}.wav")
            
            # 再次强制确保是 CPU tensor (防御性编程)
            source_wav = sources[i].cpu()
            
            save_audio(source_wav, output_path, samplerate=sr)
            outputs[name] = CogPath(output_path)
            print(f"Saved: {name}.wav")
        
        if stem == "vocals" or stem == "all":
            vocals_idx = source_names.index("vocals") if "vocals" in source_names else None
            if vocals_idx is not None:
                # 在 CPU 上操作
                no_vocals = sources.clone()
                no_vocals[vocals_idx] = 0
                no_vocals_mix = no_vocals.sum(dim=0)
                
                output_path = os.path.join(output_dir, "no_vocals.wav")
                save_audio(no_vocals_mix, output_path, samplerate=sr)
                outputs["no_vocals"] = CogPath(output_path)
                print("Saved: no_vocals.wav")
        
        return outputs

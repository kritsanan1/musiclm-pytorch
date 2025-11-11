<img src="./musiclm.png" width="450px"></img>

## MusicLM - Pytorch

Implementation of <a href="https://google-research.github.io/seanet/musiclm/examples/">MusicLM</a>, Google's new SOTA model for music generation using attention networks, in Pytorch.

They are basically using text-conditioned <a href="https://github.com/lucidrains/audiolm-pytorch">AudioLM</a>, but surprisingly with the embeddings from a text-audio contrastive learned model named <a href="https://arxiv.org/abs/2208.12415">MuLan</a>. MuLan is what will be built out in this repository, with AudioLM modified from the other repository to support the music generation needs here.

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication with the <a href="https://laion.ai/">LAION</a> community

<a href="https://www.youtube.com/watch?v=jTrYIGxOuKQ">What's AI by Louis Bouchard</a>

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their <a href="https://huggingface.co/docs/accelerate/index">accelerate</a> training library

## Usage

```install
$ pip install musiclm-pytorch
```

## Usage

`MuLaN` first needs to be trained

```python
import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# get a ton of <sound, text> pairs and train

wavs = torch.randn(2, 1024)
texts = torch.randint(0, 20000, (2, 256))

loss = mulan(wavs, texts)
loss.backward()

# after much training, you can embed sounds and text into a joint embedding space
# for conditioning the audio LM

embeds = mulan.get_audio_latents(wavs)  # during training

embeds = mulan.get_text_latents(texts)  # during inference
```

To obtain the conditioning embeddings for the three transformers that are a part of `AudioLM`, you must use the `MuLaNEmbedQuantizer` as so

```python
from musiclm_pytorch import MuLaNEmbedQuantizer

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

# now say you want the conditioning embeddings for semantic transformer

wavs = torch.randn(2, 1024)
conds = quantizer(wavs = wavs, namespace = 'semantic') # (2, 8, 1024) - 8 is number of quantizers
```

To train (or finetune) the three transformers that are a part of `AudioLM`, you simply follow the instructions over at `audiolm-pytorch` for training, but pass in the `MulanEmbedQuantizer` instance to the training classes under the keyword `audio_conditioner`

ex. `SemanticTransformerTrainer`

```python
import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
).cuda()

trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
    folder ='/path/to/audio/files',
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1
)

trainer.train()
```

After much training on all three transformers (semantic, coarse, fine), you will pass your finetuned or trained-from-scratch `AudioLM` and `MuLaN` wrapped in `MuLaNEmbedQuantizer` to the `MusicLM`

```python
# you need the trained AudioLM (audio_lm) from above
# with the MulanEmbedQuantizer (mulan_embed_quantizer)

from musiclm_pytorch import MusicLM

musiclm = MusicLM(
    audio_lm = audio_lm,                 # `AudioLM` from https://github.com/lucidrains/audiolm-pytorch
    mulan_embed_quantizer = quantizer    # the `MuLaNEmbedQuantizer` from above
)

music = musiclm('the crystalline sounds of the piano in a ballroom', num_samples = 4) # sample 4 and pick the top match with mulan
```

## Advanced Features

### Variable Length Audio Support

The AudioSpectrogramTransformer now supports variable length audio with automatic masking:

```python
from musiclm_pytorch import AudioSpectrogramTransformer

# Create transformer with variable length support
audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    support_variable_length = True,
    max_audio_length = 16000 * 30  # 30 seconds at 16kHz
)

# Use with different length audio
wavs = torch.randn(2, 12000)  # Different lengths
audio_lengths = torch.tensor([12000, 8000])  # Actual lengths

embeddings = audio_transformer(wavs, audio_lengths=audio_lengths)
```

### OpenCLIP Integration

MuLaN can now be used with OpenCLIP for audio-text contrastive learning:

```python
from musiclm_pytorch import create_mulan_open_clip_model

# Create MuLaN model optimized for OpenCLIP
mulan_openclip = create_mulan_open_clip_model(
    dim = 512,
    depth = 6,
    heads = 8,
    use_case = 'music',  # 'music', 'speech', or 'general'
    audio_sample_rate = 16000
)

# Encode audio and text
audio_embeddings = mulan_openclip.encode_audio(wavs, audio_lengths)
text_embeddings = mulan_openclip.encode_text(texts)

# Get similarity scores
logits_per_audio, logits_per_text = mulan_openclip.get_similarity(wavs, texts, audio_lengths)
```

### Optimized Spectrogram Parameters

Get optimal spectrogram parameters for different use cases:

```python
from musiclm_pytorch import get_optimal_spectrogram_params, create_optimized_audiospectrogram_transformer

# Get optimal parameters for music
params = get_optimal_spectrogram_params(audio_sample_rate=16000, use_case='music')

# Create optimized transformer
transformer = create_optimized_audiospectrogram_transformer(
    dim = 512,
    depth = 6,
    use_case = 'music',
    audio_sample_rate = 16000
)
```

### Advanced Training Features

The trainer now supports comprehensive training optimizations:

```python
from musiclm_pytorch.trainer import MuLaNTrainer

# Create trainer with advanced features
trainer = MuLaNTrainer(
    mulan = mulan,
    dataset = dataset,
    use_mixed_precision = True,  # FP16 training
    use_ema = True,  # Exponential moving averages
    lr_scheduler_type = 'cosine',  # Cosine annealing
    warmup_steps = 1000,
    weight_decay = 0.01,
    compile_model = True,  # PyTorch 2.0 compilation
    max_checkpoints = 5,
    checkpoint_every = 5000
)

# Train with validation and comprehensive logging
trainer.train(num_epochs = 10, with_validation = True)
```

### Comprehensive Evaluation Metrics

Evaluate your model with detailed metrics:

```python
from musiclm_pytorch.evaluation import MusicLMEvaluator

# Create evaluator
evaluator = MusicLMEvaluator(
    mulan = mulan,
    device = 'cuda'
)

# Evaluate model
texts = ["A calm piano melody", "Upbeat electronic music"]
generated_audio = [torch.randn(1, 16000), torch.randn(1, 16000)]

results = evaluator.comprehensive_evaluation(
    texts = texts,
    generated_audio = generated_audio,
    reference_audio = None,
    audio_lengths = [16000, 16000]
)

print("Audio Quality Metrics:", results['audio_quality'])
print("Text-Audio Alignment:", results['text_audio_alignment'])
print("Music Structure Analysis:", results['music_structure'])
```

### Inference Optimization and Deployment

Optimize models for fast inference and deployment:

```python
from musiclm_pytorch.inference import (
    InferenceConfig, OptimizedMuLaN, BatchInferenceEngine,
    ONNXExporter, DeploymentPackager
)

# Create optimized inference config
config = InferenceConfig(
    use_quantization = True,  # Model quantization
    use_mixed_precision = True,  # FP16 inference
    batch_size = 4,
    memory_efficient = True,
    compile_model = True
)

# Optimize model for inference
optimized_mulan = OptimizedMuLaN(mulan, config)

# Create batch inference engine
engine = BatchInferenceEngine(
    model = mulan,
    config = config
)

# Process multiple audio files efficiently
texts = ["Music description 1", "Music description 2", "Music description 3"]
audio_outputs = engine.process_text_to_audio_batch(
    texts = texts,
    audio_shapes = [(80, 1024), (80, 1024), (80, 1024)]
)

# Export to ONNX for deployment
onnx_exporter = ONNXExporter(config)
onnx_exporter.export_mulan_to_onnx(
    mulan = mulan,
    export_path = "mulan_model.onnx"
)

# Create deployment package
deployment_packager = DeploymentPackager(config)
deployment_packager.create_deployment_package(
    models = {'mulan': mulan},
    package_path = "musiclm_deployment_package",
    include_examples = True
)
```

### Multi-GPU and Distributed Training

Train on multiple GPUs with distributed training:

```python
from musiclm_pytorch.distributed import (
    DistributedMuLaNTrainer, DistributedConfig,
    run_distributed_training, create_distributed_config
)

# Create distributed config
distributed_config = DistributedConfig(
    world_size = 4,  # 4 GPUs
    use_ddp = True,
    use_amp = True,  # Mixed precision
    gradient_accumulation_steps = 2,
    gradient_clipping = 1.0,
    sync_batchnorm = True
)

# Create distributed trainer
trainer = DistributedMuLaNTrainer(
    mulan = mulan,
    dataset = dataset,
    distributed_config = distributed_config,
    lr = 1e-4,
    weight_decay = 0.01
)

# Or use the convenient function
results = run_distributed_training(
    model = mulan,
    dataset = dataset,
    num_epochs = 10,
    batch_size = 32,
    learning_rate = 1e-4,
    world_size = 4
)
```

### Benchmarking and Speed Testing

Benchmark inference performance:

```python
from musiclm_pytorch.inference import benchmark_inference_speed

# Benchmark different configurations
results = benchmark_inference_speed(
    model = mulan,
    config = config,
    num_runs = 100,
    warmup_runs = 10
)

print(f"Average inference time: {results['average_time_ms']:.2f}ms")
print(f"Samples per second: {results['samples_per_second']:.2f}")
```

### Advanced Training Features

The trainer now supports comprehensive training optimizations:

```python
from musiclm_pytorch.trainer import MuLaNTrainer

# Create trainer with advanced features
trainer = MuLaNTrainer(
    mulan = mulan,
    dataset = dataset,
    use_mixed_precision = True,  # FP16 training
    use_ema = True,  # Exponential moving averages
    lr_scheduler_type = 'cosine',  # Cosine annealing
    warmup_steps = 1000,
    weight_decay = 0.01,
    compile_model = True,  # PyTorch 2.0 compilation
    max_checkpoints = 5,
    checkpoint_every = 5000
)

# Train with validation and comprehensive logging
trainer.train(num_epochs = 10, with_validation = True)
```

### Comprehensive Evaluation Metrics

Evaluate your model with detailed metrics:

```python
from musiclm_pytorch.evaluation import MusicLMEvaluator

# Create evaluator
evaluator = MusicLMEvaluator(
    mulan = mulan,
    device = 'cuda'
)

# Evaluate model
texts = ["A calm piano melody", "Upbeat electronic music"]
generated_audio = [torch.randn(1, 16000), torch.randn(1, 16000)]

results = evaluator.comprehensive_evaluation(
    texts = texts,
    generated_audio = generated_audio,
    reference_audio = None,
    audio_lengths = [16000, 16000]
)

print("Audio Quality Metrics:", results['audio_quality'])
print("Text-Audio Alignment:", results['text_audio_alignment'])
print("Music Structure Analysis:", results['music_structure'])
```

### Inference Optimization and Deployment

Optimize models for fast inference and deployment:

```python
from musiclm_pytorch.inference import (
    InferenceConfig, OptimizedMuLaN, BatchInferenceEngine,
    ONNXExporter, DeploymentPackager
)

# Create optimized inference config
config = InferenceConfig(
    use_quantization = True,  # Model quantization
    use_mixed_precision = True,  # FP16 inference
    batch_size = 4,
    memory_efficient = True,
    compile_model = True
)

# Optimize model for inference
optimized_mulan = OptimizedMuLaN(mulan, config)

# Create batch inference engine
engine = BatchInferenceEngine(
    model = mulan,
    config = config
)

# Process multiple audio files efficiently
texts = ["Music description 1", "Music description 2", "Music description 3"]
audio_outputs = engine.process_text_to_audio_batch(
    texts = texts,
    audio_shapes = [(80, 1024), (80, 1024), (80, 1024)]
)

# Export to ONNX for deployment
onnx_exporter = ONNXExporter(config)
onnx_exporter.export_mulan_to_onnx(
    mulan = mulan,
    export_path = "mulan_model.onnx"
)

# Create deployment package
deployment_packager = DeploymentPackager(config)
deployment_packager.create_deployment_package(
    models = {'mulan': mulan},
    package_path = "musiclm_deployment_package",
    include_examples = True
)
```

### Multi-GPU and Distributed Training

Train on multiple GPUs with distributed training:

```python
from musiclm_pytorch.distributed import (
    DistributedMuLaNTrainer, DistributedConfig,
    run_distributed_training, create_distributed_config
)

# Create distributed config
distributed_config = DistributedConfig(
    world_size = 4,  # 4 GPUs
    use_ddp = True,
    use_amp = True,  # Mixed precision
    gradient_accumulation_steps = 2,
    gradient_clipping = 1.0,
    sync_batchnorm = True
)

# Create distributed trainer
trainer = DistributedMuLaNTrainer(
    mulan = mulan,
    dataset = dataset,
    distributed_config = distributed_config,
    lr = 1e-4,
    weight_decay = 0.01
)

# Or use the convenient function
results = run_distributed_training(
    model = mulan,
    dataset = dataset,
    num_epochs = 10,
    batch_size = 32,
    learning_rate = 1e-4,
    world_size = 4
)
```

### Benchmarking and Speed Testing

Benchmark inference performance:

```python
from musiclm_pytorch.inference import benchmark_inference_speed

# Benchmark different configurations
results = benchmark_inference_speed(
    model = mulan,
    config = config,
    num_runs = 100,
    warmup_runs = 10
)

print(f"Average inference time: {results['average_time_ms']:.2f}ms")
print(f"Samples per second: {results['samples_per_second']:.2f}")
```



## Citations

```bibtex
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
```

```bibtex
@article{Huang2022MuLanAJ,
    title   = {MuLan: A Joint Embedding of Music Audio and Natural Language},
    author  = {Qingqing Huang and Aren Jansen and Joonseok Lee and Ravi Ganti and Judith Yue Li and Daniel P. W. Ellis},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.12415}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

```bibtex
@article{Liu2022PatchDropoutEV,
    title   = {PatchDropout: Economizing Vision Transformers Using Patch Dropout},
    author  = {Yue Liu and Christos Matsoukas and Fredrik Strand and Hossein Azizpour and Kevin Smith},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.07220}
}
```

```bibtex
@misc{liu2021swin,
    title   = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    author  = {Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
    year    = {2021},
    eprint  = {2111.09883},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{gilmer2023intriguing
    title  = {Intriguing Properties of Transformer Training Instabilities},
    author = {Justin Gilmer, Andrea Schioppa, and Jeremy Cohen},
    year   = {2023},
    status = {to be published - one attention stabilization technique is circulating within Google Brain, being used by multiple teams}
}
```

```bibtex
@inproceedings{Shukor2022EfficientVP,
    title   = {Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment},
    author  = {Mustafa Shukor and Guillaume Couairon and Matthieu Cord},
    booktitle = {British Machine Vision Conference},
    year    = {2022}
}
```

```bibtex
@inproceedings{Zhai2023SigmoidLF,
    title   = {Sigmoid Loss for Language Image Pre-Training},
    author  = {Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
    year    = {2023}
}
```

*The only truth is music.* - Jack Kerouac

*Music is the universal language of mankind.* - Henry Wadsworth Longfellow

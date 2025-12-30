# Ollama Setup Guide for Proverbs Clustering

This guide will help you set up Ollama for high-quality cluster title generation.

## What is Ollama?

Ollama is a tool that lets you run large language models locally on your machine. It's:
- **Free** and open source
- **Private** - your data never leaves your computer
- **Fast** - optimized for consumer hardware
- **Easy** - simple command-line interface

## Installation

### macOS

```bash
brew install ollama
```

Or download from: https://ollama.com/download

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download the installer from: https://ollama.com/download

## Starting Ollama

After installation, start the Ollama service:

```bash
ollama serve
```

Leave this terminal window open while using the clustering pipeline. Ollama runs in the background and listens for API requests.

## Pulling Models

Pull one or more models for title generation:

### Recommended: Llama 3 (4.7GB)

```bash
ollama pull llama3
```

**Best for**: Excellent quality, balanced speed

### Alternative: Mistral (4.1GB)

```bash
ollama pull mistral
```

**Best for**: Very good quality, slightly faster than llama3

### Lightweight: Phi-3 Mini (2.3GB)

```bash
ollama pull phi3:mini
```

**Best for**: Faster generation, good quality, lower resource usage

## Testing Your Setup

### Quick Test

```bash
# Test that Ollama is running
curl http://localhost:11434/api/tags

# Test text generation
ollama run llama3 "Summarize this: The sluggard does not plow in the autumn"
```

### Python Test

```python
from cluster_titles import initialize_title_generator

# Initialize with Ollama
generator = initialize_title_generator(backend='ollama', model='llama3')

# Test title generation
test_verses = [
    "The sluggard does not plow in the autumn; he will seek at harvest and have nothing.",
    "The desire of the sluggard kills him, for his hands refuse to labor."
]

titles = generator.generate_all_titles(test_verses)
print(titles)
```

Expected output:
```python
{
    'title_1word': 'Laziness',
    'title_3words': 'Laziness Brings Poverty',
    'title_5words': 'Consequences Of Laziness And Sloth'
}
```

## Using Ollama in the Pipeline

### New Clustering Experiments

Ollama is now the default backend. Just run:

```python
python main.py
```

Or in code:

```python
from clustering_pipeline import run_clustering_pipeline

experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    k_range=(10, 41),
    title_backend='ollama',  # This is now the default
    title_model='llama3'      # This is now the default
)
```

### Regenerating Titles for Existing Experiments

Use the regeneration script:

```bash
# Preview changes for single cluster
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --preview 0

# Dry run to see what would change
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --dry-run

# Regenerate titles for one experiment
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --model llama3

# Regenerate titles for all experiments
python regenerate_titles.py \
    --all \
    --model llama3
```

## Switching Between Models

### Use Mistral Instead

```python
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    title_backend='ollama',
    title_model='mistral'
)
```

### Use Phi-3 for Faster Generation

```python
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    title_backend='ollama',
    title_model='phi3:mini'
)
```

### Fallback to Transformers (Flan-T5)

If Ollama isn't available:

```python
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    title_backend='transformers',
    title_model='google/flan-t5-small'
)
```

## Performance Comparison

| Model | Size | Speed (per cluster) | Quality | RAM Usage |
|-------|------|---------------------|---------|-----------|
| Flan-T5-small | 77M | 2-5s | Poor ⭐ | 500MB |
| Phi-3 Mini | 2.3GB | 1-3s | Good ⭐⭐⭐ | 4GB |
| Mistral | 4.1GB | 3-8s | Excellent ⭐⭐⭐⭐⭐ | 6GB |
| Llama 3 | 4.7GB | 3-8s | Excellent ⭐⭐⭐⭐⭐ | 8GB |

*Performance measured on M1 MacBook Pro

## Troubleshooting

### "Cannot connect to Ollama"

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found"

**Solution**: Pull the model first:
```bash
ollama pull llama3
```

### "Out of memory"

**Solutions**:
1. Use a smaller model: `phi3:mini`
2. Close other applications
3. Process fewer k values at once: `--k 19 23` instead of all

### Slow generation

**Solutions**:
1. Use `phi3:mini` for faster generation
2. Ensure Ollama is using your GPU (automatic on Apple Silicon, CUDA on NVIDIA)
3. Reduce parallelism (built-in for sequential processing)

### Poor title quality

If titles are still poor:
1. Check that you're using `llama3` or `mistral`, not `flan-t5-small`
2. Verify model was pulled correctly: `ollama list`
3. Try a different model: `mistral` vs `llama3`
4. Check the example output in "Testing Your Setup" above

## Advanced Configuration

### Custom Ollama Port

If running Ollama on a different port:

```python
from cluster_titles import OllamaBackend

backend = OllamaBackend(model='llama3', base_url='http://localhost:8080')
```

### Running Ollama on a Remote Server

```python
backend = OllamaBackend(model='llama3', base_url='http://your-server:11434')
```

### Adjusting Temperature

For more creative titles (higher temperature) or more conservative titles (lower temperature):

```python
titles = generator.generate_all_titles(verses, temperature=0.5)  # default is 0.3
```

## Resource Requirements

### Minimum
- 8GB RAM
- 2GB free disk space
- CPU: Any modern processor

### Recommended
- 16GB RAM
- 10GB free disk space (for multiple models)
- Apple Silicon, or NVIDIA GPU with CUDA

## Next Steps

1. ✅ Install Ollama
2. ✅ Pull a model (`ollama pull llama3`)
3. ✅ Test your setup
4. ✅ Run the pipeline or regenerate existing titles
5. 📊 Analyze the improved cluster titles!

## Questions?

- Ollama Documentation: https://ollama.com/docs
- Ollama GitHub: https://github.com/ollama/ollama
- Available Models: https://ollama.com/library



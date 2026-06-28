# Weight Files

The Qwen3.5-2B-Base metadata files in this directory are committed for inspection:

```text
config.json
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
model.safetensors.index.json
README.md
LICENSE
```

The actual model weight file is not committed:

```text
model.safetensors-00001-of-00001.safetensors
```

It is about 4.55GB, which is too large for a normal GitHub repository.

The local downloaded checkpoint currently lives at:

```text
/workspace/hf-models/Qwen3.5-2B-Base
```

Training scripts should reference that local directory, or download the model from Hugging Face when it is missing.

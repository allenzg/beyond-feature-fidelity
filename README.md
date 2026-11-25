# Beyond Feature Fidelity

**Recovery and Readout Adaptation in Multimodal Large Language Models**  
Zhiliang Zhu, jiazheng Huang, Chubin Peng, Yang Zijie, Puyue Wang, Yuquan Xue, Dawei Zhao

[Paper](docs/paper.pdf) · [Project page](https://allenzg.github.io/beyond-feature-fidelity/) · [OpenReview](https://openreview.net/forum?id=1ki2Qs08aF) · [Citation](docs/citation.bib)

We compare raw and recovered visual features under frozen and adapted replacement readouts for four multimodal language models. The main table contains nine benchmarks. The static project page includes all 144 scores and links to a machine-readable copy.

## Run a feature-level example

From this directory, with Python 3.12 and the companion model directory alongside the checkout, run

```sh
python -m pip install 'torch>=2.6'
python check_features.py --weights ../huggingface --family llava15 --condition m6
python scripts/iclr_v2/test_eval_gqa.py
```

The example applies the released recovery module and projector to eight synthetic 1,024-dimensional tokens on CPU. For LLaVA-1.5, the expected output shape is (1, 8, 4096). The companion release supplies the LLaVA-1.5 weights. The other families’ configurations are included for researchers providing their own checkpoints. M1 and M4 use a frozen readout with raw and recovered inputs respectively. M7 and M6 adapt the readout to raw and recovered inputs respectively.

## Training and evaluation

The projector trainer is in `scripts/iclr_controls/train_projector_control.py`. Run scripts from the checkout root. The JSON configurations under `configs/iclr_v2` identify the target caches, split IDs, initialization weights and training schedule for each family. Copy the model repository’s `data` and `outputs` subdirectories into the checkout to use these original relative paths.

The full image pipeline additionally requires locally obtained image data and base models. Use the original GQA balanced question files and images with the GQA manifest builder. Build DINOv3 features with `build_dino_cache.py` and target tokens with `extract_native_tokens.py`. The trainer expects the manifests and caches specified by the selected configuration. Raw datasets, historical caches and base language models are obtained separately. The included evaluation launcher covers GQA. The nine-benchmark table on the website is transcribed from the manuscript.

```sh
python -m pip install -r requirements.txt
python -m scripts.iclr_v2.build_gqa_official_manifests --help
python -m scripts.iclr_v2.build_dino_cache --help
python -m scripts.iclr_v2.extract_native_tokens --help
python -m scripts.iclr_controls.train_projector_control --config configs/iclr_v2/llava15/llava15_m6_v2.json --seed 123
python -m scripts.iclr_v2.generate_gqa_predictions --model-family llava15 --config configs/iclr_v2/llava15/llava15_m6_v2.json --output-jsonl predictions.jsonl --summary-json summary.json
```

The prediction launcher reads local backbone snapshots under `models/` and accepts model paths on the command line. The optional GPU pipeline requires a Transformers build exposing Qwen3_5ForConditionalGeneration and the selected family’s native model support. Feature-level checks use PyTorch alone.

## Release contents

- [Weights and checkpoint selection](docs/weights.md)
- [Native CLIP follow-up](supplementary/native-clip/README.md)
- [Main-table scores](docs/results.json)
- `docs/` is a build-free GitHub Pages site. Its paths are relative to the repository site root.

## License

The project code is available under [Apache-2.0](LICENSE). The authors release the [paper](docs/paper.pdf) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

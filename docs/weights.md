# LLaVA-1.5 checkpoints

The [Hugging Face model repository](https://huggingface.co/allenzhengzz/beyond-feature-fidelity) contains four files. For Recovered–Adapted inference, load the shared DINOv3 recovery module and the M6 projector. P1 is the clean-trained projector shared by M1 and M4. M7 supplies the direct raw-adaptation comparison.

The [inventory](weights.json) contains exact model-repository paths, file sizes and checksums. See the [code example](../README.md#run-a-feature-level-example).

# Source Code Documentation

## Modeling Files

The `Modeling/` folder contains the main model-level modifications. When importing the model, you need to use the corresponding modeling file based on your acceleration method:

### Qwen Model Variants

- **FastV, Random, Frame methods**: Use `modeling_qwen2_5_omni_fastv.py`
  ```python
  from Modeling.modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
  ```

- **DART method**: Use `modeling_qwen2_5_omni_dart.py`
  ```python
  from Modeling.modeling_qwen2_5_omni_dart import Qwen2_5OmniForConditionalGeneration
  ```

### Phi-4 Model

The same pattern applies to Phi-4 models. Select the appropriate modeling file based on your acceleration strategy.

## Directory Structure

- `Modeling/` - Qwen and Phi4MM model modifications
- `Qwen_2.5_Omni/` - Qwen Omni evaluation scripts
- `Phi4MM/` - Phi-4 multimodal model scripts
- `Aero-1/` - Aero-1 model evaluation
- `Voxtral/` - Voxtral model evaluation
- `Segement/` - Audio segmentation and task generation utilities
- `analyse_audio_duration/` - Audio duration analysis tools

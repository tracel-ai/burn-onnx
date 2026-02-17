#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "onnxruntime>=1.18.0",
#   "transformers>=4.44.0",
#   "numpy",
#   "torch",
#   "onnxscript",
# ]
# ///

import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir, process_model


SUPPORTED_MODELS = {
    'qwen1.5-0.5b': {
        'hf_name': 'Qwen/Qwen1.5-0.5B',
        'display_name': 'Qwen1.5 0.5B',
        'seq_length': 32,
    },
    'qwen2.5-0.5b': {
        'hf_name': 'Qwen/Qwen2.5-0.5B',
        'display_name': 'Qwen2.5 0.5B',
        'seq_length': 32,
    },
    'qwen3-0.6b': {
        'hf_name': 'Qwen/Qwen3-0.6B',
        'display_name': 'Qwen3 0.6B',
        'seq_length': 32,
    },
}


def safe_name(model_name):
    """Sanitize model name for use in filenames (dots break Rust Path operations)."""
    return model_name.replace('.', '_')


def download_and_convert_model(model_name, output_path):
    """Download Qwen model from HuggingFace and export to ONNX format."""
    import torch

    model_config = SUPPORTED_MODELS[model_name]
    hf_name = model_config['hf_name']
    seq_length = model_config['seq_length']
    display_name = model_config['display_name']

    print(f"Downloading {display_name} model from HuggingFace...")

    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()

    print(f"  Model loaded (vocab_size={config.vocab_size})")

    # Wrap model to return only logits tensor
    class LogitsWrapper(torch.nn.Module):
        def __init__(self, causal_lm):
            super().__init__()
            self.causal_lm = causal_lm

        def forward(self, input_ids):
            return self.causal_lm(input_ids).logits

    wrapper = LogitsWrapper(model)
    wrapper.eval()

    print("Exporting to ONNX format...")

    dummy_input_ids = torch.randint(0, config.vocab_size, (1, seq_length))

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids,),
            output_path,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'},
            },
            opset_version=16,
            do_constant_folding=True,
        )

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create ONNX file at {output_path}")

    print(f"  ONNX model saved to: {output_path}")


def generate_test_data(model_path, output_path, model_name):
    """Generate test input/output data and save as PyTorch tensors."""
    import torch
    import onnxruntime as ort

    print("\nGenerating test data...")

    model_config = SUPPORTED_MODELS[model_name]
    seq_length = model_config['seq_length']

    np.random.seed(42)
    batch_size = 1

    # Use vocab_size 151936 for random token IDs (common across Qwen models)
    input_ids = np.random.randint(0, 151936, size=(batch_size, seq_length), dtype=np.int64)

    print(f"  Input shapes:")
    print(f"    input_ids: {input_ids.shape}")

    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {'input_ids': input_ids})

    logits = outputs[0]

    test_data = {
        'input_ids': torch.from_numpy(input_ids),
        'logits': torch.from_numpy(logits),
    }

    torch.save(test_data, output_path)

    print(f"  Test data saved to: {output_path}")
    print(f"    logits shape: {logits.shape}")


def main():
    parser = argparse.ArgumentParser(description='Qwen Model Preparation Tool')
    parser.add_argument('--model', type=str, default='qwen2.5-0.5b',
                        choices=list(SUPPORTED_MODELS.keys()),
                        help=f'Qwen model to download and prepare (default: qwen2.5-0.5b)')
    parser.add_argument('--list', action='store_true',
                        help='List all supported models')

    args = parser.parse_args()

    if args.list:
        print("Supported Qwen models:")
        for model_id, config in SUPPORTED_MODELS.items():
            print(f"  - {model_id:20s} ({config['display_name']})")
        return

    model_name = args.model
    display_name = SUPPORTED_MODELS[model_name]['display_name']

    print("=" * 60)
    print(f"{display_name} Model Preparation Tool")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir("qwen")

    sname = safe_name(model_name)
    original_path = artifacts_dir / f"{sname}.onnx"
    processed_path = artifacts_dir / f"{sname}_opset16.onnx"
    test_data_path = artifacts_dir / f"{sname}_test_data.pt"

    if processed_path.exists() and test_data_path.exists():
        print(f"\nAll files already exist for {display_name}:")
        print(f"  Model: {processed_path}")
        print(f"  Test data: {test_data_path}")
        print("\nNothing to do!")
        return

    if not original_path.exists() and not processed_path.exists():
        print(f"\nStep 1: Downloading and converting {display_name} model...")
        download_and_convert_model(model_name, original_path)

    if not processed_path.exists():
        print("\nStep 2: Processing model...")
        process_model(original_path, processed_path, target_opset=16)

        if original_path.exists():
            original_path.unlink()

    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(processed_path, test_data_path, model_name)

    print("\n" + "=" * 60)
    print(f"{display_name} model preparation completed!")
    print(f"  Model: {processed_path}")
    print(f"  Test data: {test_data_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

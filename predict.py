import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CNNTransformer, ctc_greedy_decode, ctc_greedy_decode_constrained


IMG_H = 64
IMG_W = 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory containing .jpg images')
    parser.add_argument('--output-dir', required=True, help='Directory to write CSV files')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return parser.parse_args()


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    inv_vocab_raw = checkpoint['inv_vocab']
    inv_vocab = {int(k): v for k, v in inv_vocab_raw.items()}
    num_classes = checkpoint['num_classes']

    model = CNNTransformer(
        num_classes=num_classes,
        d_model=256,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    return model, inv_vocab


def run_inference(model, inv_vocab, transform, input_dir, device, constrained=True):
    rows = []
    input_dir = Path(input_dir)

    with torch.no_grad():
        for img_path in sorted(input_dir.glob('*.jpg')):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            log_probs = model(img_tensor)

            if constrained:
                pred = ctc_greedy_decode_constrained(log_probs, inv_vocab)[0]
            else:
                pred = ctc_greedy_decode(log_probs, inv_vocab)[0]

            rows.append({'image_id': img_path.name, 'plate_text': pred})

    return rows


def write_csv(output_dir, strategy_name, rows):
    output_path = Path(output_dir) / f'{strategy_name}.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'plate_text'])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_dir = Path(__file__).resolve().parent
    checkpoint_path = submission_dir / 'best.pt'

    transform = build_transform()
    model, inv_vocab = load_model(checkpoint_path, device)

    # strategy 1: constrained decoding (main strategy)
    rows_constrained = run_inference(
        model, inv_vocab, transform, args.input_dir, device, constrained=True
    )
    write_csv(args.output_dir, 'transformer_constrained', rows_constrained)

    # strategy 2: plain greedy decoding (for comparison)
    rows_greedy = run_inference(
        model, inv_vocab, transform, args.input_dir, device, constrained=False
    )
    write_csv(args.output_dir, 'transformer_greedy', rows_greedy)

    # save summary
    summary = {
        'checkpoint': str(checkpoint_path),
        'device': args.device,
        'strategies': ['transformer_constrained', 'transformer_greedy']
    }
    summary_path = output_dir / 'strategies_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'Done. Wrote {len(rows_constrained)} predictions per strategy to {output_dir}')


if __name__ == '__main__':
    main()

import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from alpr.paths import OCR_OUTPUT_DIR, DEEPSEEK_OCR_WEIGHTS_DIR

class OCR:
    def __init__(self, cp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(cp_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(cp_path, trust_remote_code=True)

    def inference(self, img_path):
        self.model.eval()
        prompt = "<image>\nFree OCR."
        OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        res = self.model.infer(self.tokenizer,
                               prompt=prompt, image_file=img_path, output_path=str(OCR_OUTPUT_DIR), image_size=640, base_size=640,
                               crop_mode=False, save_results=False, eval_mode=True)
        return res

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, default="")

    return parser.parse_args()

def main():
    args = get_args()

    img_path = args.img_path
    ocr = OCR(DEEPSEEK_OCR_WEIGHTS_DIR)

    result = ocr.inference(img_path)
    print(result)

if __name__ == "__main__":
    main()

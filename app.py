#!/usr/bin/env python3
"""
Fully Automated Real Estate Photo Enhancer SaaS
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoRE")

# Simple paid users list (replace with real database + Stripe later)
PAID_USERS = set()

class AutoEnhancer:
    def __init__(self):
        self.presets = {
            "Interior": {"strength": 1.10, "warm": 0.07, "sky": 0.10},
            "Exterior": {"strength": 1.20, "warm": 0.05, "sky": 0.25},
            "Twilight": {"strength": 1.30, "warm": 0.12, "sky": 0.20},
            "Luxury":   {"strength": 1.15, "warm": 0.09, "sky": 0.15},
        }

    def process(self, img: np.ndarray, preset: str = "Interior", is_paid: bool = False) -> np.ndarray:
        try:
            cfg = self.presets.get(preset, self.presets["Interior"])
            result = img.copy()

            result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
            result = self._white_balance(result)
            result = self._exposure(result, cfg["strength"])
            result = self._clahe(result, cfg["strength"])
            result = self._temperature(result, cfg["warm"])
            result = self._bright_boost(result, cfg["sky"])
            result = self._sharpen(result, 0.55)

            result = np.clip(result, 0, 255).astype(np.uint8)

            if not is_paid:
                result = self._watermark(result)

            return result
        except Exception as e:
            logger.error(e)
            return img

    def _white_balance(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] -= (np.mean(lab[:, :, 1]) - 128) * 0.7
        lab[:, :, 2] -= (np.mean(lab[:, :, 2]) - 128) * 0.7
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _exposure(self, img, s):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.22 * s), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _clahe(self, img, s):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2 + s * 0.5, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def _temperature(self, img, amount):
        res = img.astype(np.float32)
        res[:, :, 2] = np.clip(res[:, :, 2] * (1 + amount), 0, 255)
        res[:, :, 0] = np.clip(res[:, :, 0] * (1 - amount * 0.35), 0, 255)
        return res.astype(np.uint8)

    def _bright_boost(self, img, amount):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2]
        mask = np.clip((v - 150) / 100, 0, 1)
        hsv[:, :, 2] = np.clip(v + mask * 45 * amount, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _sharpen(self, img, amount):
        blur = cv2.GaussianBlur(img, (0, 0), 1.6)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

    def _watermark(self, img):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        txt = Image.new("RGBA", pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        text = "FREE PREVIEW • Upgrade to remove"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        x = pil.width - (bbox[2] - bbox[0]) - 12
        y = pil.height - (bbox[3] - bbox[1]) - 12
        draw.rectangle([x-6, y-4, x + (bbox[2]-bbox[0]) + 6, y + (bbox[3]-bbox[1]) + 4], fill=(0, 0, 0, 120))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))
        return cv2.cvtColor(np.array(Image.alpha_composite(pil, txt).convert("RGB")), cv2.COLOR_RGB2BGR)


engine = AutoEnhancer()

def enhance_photo(image, preset, user_email):
    if image is None:
        return None, "Please upload a photo."

    is_paid = user_email.strip().lower() in PAID_USERS if user_email else False

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = engine.process(img_bgr, preset=preset, is_paid=is_paid)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    if is_paid:
        status = "✅ Paid account – clean image delivered"
    else:
        status = "🔒 Free preview (watermarked). Upgrade to remove watermark."

    return result_rgb, status


with gr.Blocks(title="Auto Real Estate Enhancer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Automated Real Estate Photo Enhancer
    Upload → Automatic professional enhancement → Download
    **Free tier has watermark. Paid removes it.**
    """)

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Upload Property Photo")
            preset = gr.Radio(["Interior", "Exterior", "Twilight", "Luxury"], value="Interior", label="Type")
            email = gr.Textbox(label="Your Email (for paid accounts)", placeholder="agent@example.com")
            btn = gr.Button("Enhance Automatically", variant="primary")

        with gr.Column():
            image_out = gr.Image(label="Enhanced Result")
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(enhance_photo, inputs=[image_in, preset, email], outputs=[image_out, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
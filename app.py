import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Temporary paid users list (we will replace this with real payments later)
PAID_USERS = set()

class PhotoEnhancer:
    def __init__(self):
        self.presets = {
            "Interior": {"strength": 1.13, "warm": 0.08, "sky": 0.08},
            "Exterior": {"strength": 1.23, "warm": 0.05, "sky": 0.27},
            "Twilight": {"strength": 1.33, "warm": 0.14, "sky": 0.20},
            "Luxury":   {"strength": 1.17, "warm": 0.10, "sky": 0.13},
        }

    def enhance(self, img, preset="Interior", is_paid=False):
        try:
            cfg = self.presets.get(preset, self.presets["Interior"])
            result = img.copy()

            # Light denoising
            result = cv2.fastNlMeansDenoisingColored(result, None, 3.5, 3.5, 7, 21)

            # White balance
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:, :, 1] -= (np.mean(lab[:, :, 1]) - 128) * 0.65
            lab[:, :, 2] -= (np.mean(lab[:, :, 2]) - 128) * 0.65
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

            # Exposure / brightness
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.23 * cfg["strength"]), 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Local contrast (CLAHE)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.3 + cfg["strength"] * 0.4, tileGridSize=(8, 8))
            l = clahe.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            # Slight warmth
            res = result.astype(np.float32)
            res[:, :, 2] = np.clip(res[:, :, 2] * (1 + cfg["warm"]), 0, 255)
            res[:, :, 0] = np.clip(res[:, :, 0] * (1 - cfg["warm"] * 0.35), 0, 255)
            result = res.astype(np.uint8)

            # Boost bright areas (sky / windows)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            v = hsv[:, :, 2]
            mask = np.clip((v - 148) / 100, 0, 1)
            hsv[:, :, 2] = np.clip(v + mask * 42 * cfg["sky"], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Light sharpening
            blur = cv2.GaussianBlur(result, (0, 0), 1.5)
            result = cv2.addWeighted(result, 1.45, blur, -0.45, 0)

            result = np.clip(result, 0, 255).astype(np.uint8)

            if not is_paid:
                result = self.add_watermark(result)

            return result
        except Exception as e:
            print("Enhancement error:", e)
            return img

    def add_watermark(self, img):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        txt_layer = Image.new("RGBA", pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        text = "Preview • Upgrade to download clean version"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        x = pil.width - (bbox[2] - bbox[0]) - 14
        y = pil.height - (bbox[3] - bbox[1]) - 12

        draw.rectangle(
            [x - 7, y - 5, x + (bbox[2] - bbox[0]) + 7, y + (bbox[3] - bbox[1]) + 5],
            fill=(0, 0, 0, 130)
        )
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 210))

        combined = Image.alpha_composite(pil, txt_layer)
        return cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)


enhancer = PhotoEnhancer()


def process_image(image, preset, email):
    if image is None:
        return None, "Please upload a photo first."

    is_paid = False
    if email and email.strip().lower() in PAID_USERS:
        is_paid = True

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = enhancer.enhance(img_bgr, preset=preset, is_paid=is_paid)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    if is_paid:
        status = "Clean version ready (no watermark)"
    else:
        status = "Free preview (watermarked). Upgrade to remove watermark."

    return result_rgb, status


# Interface
with gr.Blocks(
    title="Real Estate Photo Enhancer",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1050px !important}"
) as demo:

    gr.Markdown(
        """
        # Real Estate Photo Enhancer
        Upload a property photo and get a cleaner, brighter version in seconds.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Property Photo",
                height=360
            )
            preset = gr.Radio(
                choices=["Interior", "Exterior", "Twilight", "Luxury"],
                value="Exterior",
                label="Photo Type"
            )
            email = gr.Textbox(
                label="Email (only needed for paid accounts)",
                placeholder="you@email.com"
            )
            btn = gr.Button("Enhance Photo", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Enhanced Result", height=360)
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(
        fn=process_image,
        inputs=[image_input, preset, email],
        outputs=[image_output, status]
    )

    gr.Markdown(
        """
        ---
        Free version includes a small watermark.  
        Paid version removes the watermark so you can use the photos on listings.
        """
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

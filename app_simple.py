import os
import subprocess

import gradio as gr
import imageio

OUTPUT_DIR = 'outputs/app_simple'
EHM_TRACKER_DIR = 'EHM-Tracker'


def predict_overlay(image_path):
    """Track a single image with EHM-Tracker and return the mesh render overlaid on it."""
    if image_path is None:
        raise gr.Error("Please upload an image first.")

    name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.abspath(OUTPUT_DIR)
    viz_fp = os.path.join(output_dir, name, 'viz_tracking.mp4')

    if not os.path.exists(viz_fp):
        subprocess.run(
            ['python', '-m', 'src.tracking_single_image',
             '-i', os.path.abspath(image_path), '-o', output_dir, '--save_vis_video'],
            cwd=EHM_TRACKER_DIR, check=True,
        )
        if not os.path.exists(viz_fp):
            raise gr.Error("Tracking failed: no visualization was produced. Check the console logs.")

    # The viz video's single frame is [input | mesh render | overlay] side by side.
    frame = imageio.get_reader(viz_fp).get_data(0)
    panel_w = frame.shape[1] // 3
    return frame[:, 2 * panel_w:]


demo = gr.Interface(
    fn=predict_overlay,
    inputs=gr.Image(label="Input Image", type="filepath"),
    outputs=gr.Image(label="EHM Mesh Overlay"),
    title="EHM Single-Image Tracker",
    description="Upload a single image to get the predicted EHM 3D mesh rendered on top of it.",
    flagging_mode="never",
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo.launch(server_name="0.0.0.0", allowed_paths=[".", OUTPUT_DIR])

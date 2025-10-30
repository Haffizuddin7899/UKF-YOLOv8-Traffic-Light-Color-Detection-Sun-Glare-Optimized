# 🚦 UKF + YOLOv8 Traffic Light Color Detection — Sun-Glare Optimized 🌤️  
**Robust Traffic Light Color Detection for Autonomous Vehicle Systems**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Kalman Filter](https://img.shields.io/badge/UKF-FilterPy-orange.svg)](https://filterpy.readthedocs.io/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-yellow.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📖 Project Overview
This repository provides an **optimized Unscented Kalman Filter (UKF)** framework integrated with **YOLOv8** for **traffic light color detection** in autonomous vehicle systems — specifically designed to handle **sun-glare and lighting variations**.

The model compensates for sunlight effects that typically distort color readings by incorporating:
- **Dynamic Window Sizing (DWS)** for local region normalization, and  
- **Adaptive UKF optimization**, which dynamically adjusts noise, weights, and gain based on real-time sunlight intensity.

---

## 🌟 Key Features
✅ **YOLOv8-based Detection** — Efficient, real-time traffic light localization.  
✅ **Dynamic Window Sizing (DWS)** — Stabilizes ROI-based color detection.  
✅ **UKF Optimization** — Integrates nonlinear sunlight compensation in the state transition.  
✅ **Adaptive Weighting Scheme** — Adjusts sigma-point parameters (α, β, κ) dynamically.  
✅ **Sunlight Compensation** — Adds sunlight as an extra state variable.  
✅ **Time-based Analysis** — Frame timestamps for temporal accuracy plots.  
✅ **Visualization Ready** — Includes accuracy, error, and noise reduction plots.

---

## 🧩 Repository Structure
```
.
├── README.md
├── notebooks/
│   └── ukf_yolo_traffic_light_colab.ipynb     # Main Colab notebook (full implementation)
├── data/
│   └── vedois.zip                             # Example video (user-provided)

```

---

## ⚙️ Dependencies
Install the following Python packages:
```bash
pip install ultralytics filterpy opencv-python matplotlib numpy
```

### Requirements Summary
| Package | Purpose |
|----------|----------|
| `ultralytics` | YOLOv8 inference |
| `filterpy` | Unscented Kalman Filter |
| `opencv-python` | Frame extraction & image processing |
| `matplotlib` | Plotting visual results |
| `numpy` | Array operations |

---

## 🚀 Getting Started (Google Colab)
1. Open `notebooks/ukf_yolo_traffic_light_colab.ipynb` in **Google Colab**.  
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install dependencies:
   ```python
   !pip install ultralytics filterpy opencv-python matplotlib numpy
   ```
4. Upload your video file (e.g., `vedois.zip`) to Drive and update its path:
   ```python
   zip_path = '/content/drive/MyDrive/Test1/vedois.zip'
   ```
5. Run all cells sequentially — outputs include frame extraction, YOLOv8 inference, UKF optimization, and visualization plots.

---

## 🧠 How It Works — System Design
### 1. YOLOv8 + Dynamic Window Sizing (DWS)
- Detects traffic lights in each frame.
- Crops the ROI and resizes it to a fixed DWS patch (e.g., 32×32).
- Computes mean color for dominant color classification (Red/Yellow/Green).

### 2. Adaptive UKF Filtering
- **State Vector:** `[R, G, B, SunlightImpact]`
- **Measurement Vector:** `[R, G, B]`
- **Transition Function (fx):** Integrates nonlinear sunlight compensation:
  \[
  sunlight\_effect = e^{-0.05 \cdot sunlight\_intensity}
  \]
  Adjusted RGB = RGB − sunlight_effect

### 3. Dynamic Noise and Gain
- Measurement & process noise scale with sunlight intensity.
- Kalman gain adaptively decreases during high glare to reduce bias.

### 4. Visualization & Evaluation
- Generates 3 key plots:
  - Accuracy Comparison
  - Error Rate Over Time
  - Noise Reduction Over Time
- Additional per-color plots (Red, Yellow, Green) over time.

---

## 🔧 Parameter Configuration
| Parameter | Description | Default |
|------------|--------------|----------|
| `dws_size` | Dynamic window size | 32 |
| `confidence_threshold` | YOLO detection confidence | 0.3 |
| `alpha, beta, kappa` | UKF sigma-point parameters | 0.1, 2.0, 0 |
| `base_noise` | Initial noise scaling factor | 0.1 |
| `sunlight_intensity` | Randomized in notebook (replace with real input) | 0–1 |

💡 *For real deployment, replace random sunlight with an image-based or sensor-based estimator.*

---

## 📊 Output Visualizations
1. **Accuracy Comparison** – YOLOv8+DWS vs YOLOv8+DWS+UKF  
2. **Error Rate Comparison** – Frame-based error rate tracking  
3. **Noise Reduction Curve** – Impact of UKF smoothing  
4. **Per-Color Densities** – Red, Yellow, Green over frame indices  

All plots are saved automatically in the `/results/plots` directory.

---

## 🧪 Evaluation Metrics (Recommended)
| Metric | Description |
|---------|--------------|
| **Precision / Recall / F1** | Detection performance per color class |
| **RMSE** | Error between ground truth and UKF estimates |
| **Noise Suppression Ratio** | Improvement in variance after UKF |
| **Latency (FPS)** | Real-time performance indicator |

---

## ⚠️ Limitations
- Simulated sunlight intensity — must be replaced with real measurements for deployment.
- YOLOv8n weights are generic; for better accuracy, fine-tune on traffic-light datasets.
- Overexposed or saturated pixels limit correction accuracy.
- Computational overhead from per-frame UKF updates in real-time systems.

---

## 🔮 Future Enhancements
- Integrate sunlight intensity estimation using image histogram analysis.
- Implement real-time object tracking (e.g., SORT or ByteTrack) for frame continuity.
- Extend UKF to include camera pose or distance factors.
- Deploy lightweight model variants for embedded vehicle systems.

---

## 🧰 Example Usage Snippet
```python
# Run YOLOv8 + UKF pipeline
for frame_file in sorted(os.listdir(frames_dir)):
    frame_path = os.path.join(frames_dir, frame_file)
    color_density = yolo_inference_with_dws(frame_path)
    sunlight_intensity = np.random.uniform(0, 1)  # replace with real estimator
    filtered_output = apply_ukf(color_density, sunlight_intensity)
```

---

## 🪪 License
This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation


---

## 👤 Author
**HAFFIZUDDIN**  
*Software Engineering Student, Karakorum International University, Gilgit GB*  
📧 Contact: [haffizuddin7899@gmail.com]  
💻 GitHub: [https://github.com/Haffizuddin7899]

---

### 💡 Acknowledgements
This project builds on:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FilterPy Library](https://filterpy.readthedocs.io/)
- OpenCV & NumPy for image and numerical computation.

---
> “Making autonomous systems see clearly — even in the glare of the sun.”

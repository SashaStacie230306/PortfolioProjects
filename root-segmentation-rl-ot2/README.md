# Root Segmentation & Reinforcement Learning Control of the OT-2 Robot  

This project demonstrates a full pipeline from **biological image analysis** to **robotics control**:

- **Root Segmentation (Computer Vision)**  
  - U-Net model trained on seedling images.  
  - Post-processing: skeletonization, length extraction, growth zone analysis.  

- **Error Analysis**  
  - Patch-based inference for large images.  
  - Automated evaluation of segmentation quality and measurement consistency.  

- **Robotic Control (Simulation)**  
  - Custom Gym environment for OT-2 pipette robot.  
  - Implemented both **PID controller** and **PPO reinforcement learning** agent.  

---

## Results at a Glance

- U-Net segmentation successfully extracts fine root structures.  
- Error analysis detects root length and zones accurately.  
- PID control: stable convergence to goals.  
- PPO RL agent: learned precise movements (≤ 1 mm error) in simulation.  

---

## Repository Organization

- `notebooks/` → Jupyter notebooks (data prep, training, inference, analysis).  
- `scripts/` → Reusable Python modules (segmentation, RL, PID, simulation).  
- `results/` → Example predictions, figures, and demo GIFs.  
- `docs/` → Reports, training summaries, and presentation slides.  

---

⚡ *From plant roots to robot actions: bridging computer vision and lab automation.*

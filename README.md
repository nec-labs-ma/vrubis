# VRUBIS 

We propose VRUBIS, a Benchmark on Intersection Safety for Vulnerable Road Users (VRUs), designed to advance robust, real-time perception and forecasting methods for urban traffic intersections. Built on the Intersection Safety Challenge (ISC) hosted in 2024 by U.S. Department of Transportation (USDOT), which our team (NEC Labs America, NEC America, and University of Hawaiʻi) won, VRUBIS provides the community with a fully-labeled, multi-modal dataset including 1.3M RGB and thermal images, 151k LiDAR scans, 75k radar frames, traffic-signal metadata, and over 4.5M 2D/3D bounding boxes, trajectories, and conflict annotations. The dataset captures a wide range of everyday interactions involving real vehicles, pedestrians, and cyclists, alongside a set of realistic simulated collision and near collision scenarios between real vehicles and heated, articulated, mobile dummies. We release detailed annotations for detection, classification, tracking, forecasting, and conflict prediction, across all modalities. Our winning ISC solution will be open-sourced as a baseline. VRUBIS is the first benchmark to unify multi-modal perception and multi-agent prediction for VRU safety at intersections—addressing critical gaps in existing datasets by combining real-world and simulated high-risk scenarios with rich sensor diversity, and enabling impactful, safety-critical deployment in real-world urban environments.
- images can be downloaded by running `aws s3 cp s3://udsot-data/images <root>/ --no-sign-request`
- annotations can be downloaded by running `aws s3 cp s3://udsot-data/images <root>/ --recursive --no-sign-request`

```bash
## 📁 Directory Structure

vrubis/
│
├── annotations/                # Contains COCO-style annotation JSON files
│   ├── train_thermal_camera.json
│   ├── train_visual_camera.json
│   ├── val_thermal_camera.json
│   ├── val_visual_camera.json
│
├── images/                     # Images used for training/validation
│   └── Runs_*/                 # Images grouped by run, per camera
│       ├── Runs_001/
│       │   ├── VisualCamera1/
│       │   ├── VisualCamera2/
│       │   ├── ThermalCamera2/
│       │   └── ...
│       ├── Runs_002/
│       │   └── ...
│       └── ...

```
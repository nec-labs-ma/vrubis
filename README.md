# VRUBIS 

We propose VRUBIS, a Benchmark on Intersection Safety for Vulnerable Road Users (VRUs), designed to advance robust, real-time perception and forecasting methods for urban traffic intersections. Built on the Intersection Safety Challenge (ISC) hosted in 2024 by U.S. Department of Transportation (USDOT), which our team (NEC Labs America, NEC America, and University of Hawaiʻi) won, VRUBIS provides the community with a fully-labeled, multi-modal dataset including 1.3M RGB and thermal images, 151k LiDAR scans, 75k radar frames, traffic-signal metadata, and over 4.5M 2D/3D bounding boxes, trajectories, and conflict annotations. The dataset captures a wide range of everyday interactions involving real vehicles, pedestrians, and cyclists, alongside a set of realistic simulated collision and near collision scenarios between real vehicles and heated, articulated, mobile dummies. We release detailed annotations for detection, classification, tracking, forecasting, and conflict prediction, across all modalities. Our winning ISC solution will be open-sourced as a baseline. VRUBIS is the first benchmark to unify multi-modal perception and multi-agent prediction for VRU safety at intersections—addressing critical gaps in existing datasets by combining real-world and simulated high-risk scenarios with rich sensor diversity, and enabling impactful, safety-critical deployment in real-world urban environments.

## Dataset format
```
Training Data
│   ├── Run_48
│   │   ├── Lidar1_Run_48.pcap
│   │   ├── Lidar2_Run_48.pcap
│   │   ├── ThermalCamera1_Run_48.mp4
│   │   ├── ThermalCamera2_Run_48.mp4
│   │   ├── ThermalCamera3_Run_48.mp4
│   │   ├── ThermalCamera4_Run_48.mp4
│   │   ├── ThermalCamera5_Run_48.mp4
│   │   ├── VisualCamera1_Run_48.mp4
│   │   ├── VisualCamera2_Run_48.mp4
│   │   ├── VisualCamera3_Run_48.mp4
│   │   ├── VisualCamera4_Run_48.mp4
│   │   ├── VisualCamera5_Run_48.mp4
│   │   ├── VisualCamera6_Run_48.mp4
│   │   ├── VisualCamera7_Run_48.mp4
│   │   ├── VisualCamera8_Run_48.mp4
│   │   ├── ISC_Run_48_ISC_all_timing.csv
│   │   ├── ThermalCamera1_Run_48_frame-timing.csv
│   │   ├── ThermalCamera2_Run_48_frame-timing.csv
│   │   ├── ThermalCamera3_Run_48_frame-timing.csv
│   │   ├── ThermalCamera4_Run_48_frame-timing.csv
│   │   ├── ThermalCamera5_Run_48_frame-timing.csv
│   │   ├── VisualCamera1_Run_48_frame-timing.csv
│   │   ├── VisualCamera2_Run_48_frame-timing.csv
│   │   ├── VisualCamera3_Run_48_frame-timing.csv
│   │   ├── VisualCamera4_Run_48_frame-timing.csv
│   │   ├── VisualCamera5_Run_48_frame-timing.csv
│   │   ├── VisualCamera6_Run_48_frame-timing.csv
│   │   ├── VisualCamera7_Run_48_frame-timing.csv
│   │   └── VisualCamera8_Run_48_frame-timing.csv
|   |...
|...
```

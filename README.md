# DriveAI-A-Vision-Powered-Prototype-for-Self-Driving-Car-Navigation

DriveAI is a vision-based autonomous driving prototype using Raspberry Pi and a remote server. It reduces reliance on expensive sensors like LiDAR by using camera-based perception. The system demonstrates lane keeping and object detection in controlled environments.

Raspberry Pi 
i) Real-time video capture
ii) Lane detection using Two-Zone Weighted Centroid algorithm
iii) Low-latency steering control

Remote Server (Laptop)
i) Object detection using YOLOv8n
ii) Processing of traffic elements (vehicles, obstacles, etc.)

Challenges & Solutions
Processing latency → Solved using edge–cloud split architecture
Lighting variations → Handled using HSV-based thresholding
Network instability → Implemented failsafe steering mode
Real-time constraints → Lightweight YOLOv8n model used on server

Tech Stack
Python
OpenCV
Raspberry Pi
YOLOv8n (Ultralytics)
Socket streaming
NumPy




# OneAPI-AI-ML-Models

Hey so here I have did 5 different problems that I developed it in Intel OneAPI.

# i)Lane detection using hough transform method with a lane image

      This repository contains code to perform lane detection on an image using the Hough Transform method. Lane detection is a crucial component of many autonomous vehicles and driver assistance systems. The Hough Transform allows us to detect lines within an image, which is especially useful for identifying lane markings on the road.
      
      Steps Involved:
      ->Loading and Displaying the Image:
      
      We start by loading an image using the mpimg.imread function and displaying it using Matplotlib's imshow. This gives us a visual representation of the road scene we want to process.
      
      ->Converting to Grayscale:
      
      To simplify further processing, we convert the colored road image to grayscale using OpenCV's cvtColor function. Grayscale images contain only intensity information, making it easier to detect edges.
      
      ->Edge Detection using Canny Algorithm:
      
      The Canny edge detection algorithm is applied to the grayscale image. This algorithm highlights edges by detecting rapid changes in intensity. We use cv2.Canny to get the edge-detected image.
      
      ->Region of Interest Selection:
      
      Not all parts of the image are relevant for lane detection. We define a region of interest (ROI) using a polygon that covers the area of the road where lane markings are expected. The region_of_interest function is used to extract the edge-detected image within this region.
      
      ->Hough Transform for Line Detection:
      
      The Hough Transform is a technique to detect lines in an image. We apply the cv2.HoughLinesP function to detect lines in the region of interest. Parameters such as rho, theta, and thresholds are used to control the detection sensitivity.
      
      ->Drawing Detected Lines:
      
      The lines detected by the Hough Transform are drawn on the original road image. The draw_lines function takes care of this step, overlaying the detected lane lines on the road scene.
      
      ->Displaying the Final Result:
      
      Finally, we display the road image with the detected lane lines using Matplotlib's imshow. This gives us a visual representation of how well the lane detection algorithm performs.
      
      ->Usage:
      To use this lane detection method on your own images, follow these steps:
      
      Replace 'lane3.jpg' with the path to your input image.
      Adjust parameters like thresholds, line lengths, and gap sizes as needed.
      Then you will be good to go


# ii)malaria deetction using cnn with cell images


# iii)Anomaly detection using alibi detect in clothe materials
      This repository demonstrates anomaly detection in clothing material images using the Alibi Detect library. Anomaly detection is essential for identifying unusual patterns or outliers in data, making it valuable for quality control in manufacturing and ensuring product consistency.
      
      ->Workflow Overview:
      Dataset Preparation:
      
      The dataset comprises images of clothing materials. Good (normal) images are used for training and validation, while any image differing significantly from the norm is considered an outlier.
      
      ->Data Preprocessing:
      
      Images are loaded, resized, and normalized to prepare them for model training. Both good and bad (outlier) images are processed for training and evaluation.
      
      ->CNN Architecture Definition:
      
      A Convolutional Neural Network (CNN) architecture is designed to encode and decode images. The encoder network learns to represent images in a lower-dimensional space, and the decoder network reconstructs the original images from these encoded representations.
      
      ->Outlier Detector Creation:
      
      An OutlierVAE detector from Alibi Detect is utilized. This detector employs a Variational Autoencoder (VAE) architecture for anomaly detection. It is trained on the good images and aims to reconstruct them accurately.
      
      ->Training the Detector:
      
      The detector is trained on the good images dataset. The training process involves optimizing the VAE's parameters to minimize the reconstruction error.
      
      ->Setting the Threshold:
      
      The threshold for outlier detection is defined either manually or using the infer_threshold function, which sets the threshold based on the percentage of instances considered outliers.
      
      ->Anomaly Detection and Visualization:
      
      The trained detector is used to predict anomalies in both the test set and specific outlier images. Visualization functions are used to display instance scores and feature scores, aiding in anomaly identification.

# iv)cats vs dog using cnn with images

# v)Predicting whether a customer will stay or not in bank based on csv data using ANN

# How OneAPI helped me
Using a windows laptop with a mx graphic card training the malaria dataset with 30K images would have been a tiring process but with intel oneAPI it saved really a lot of time and its almost equal to the time run in mac whcih makes worrying about my laptop hardware incompatibility reduntant.
Despite working on a Windows laptop with an MX graphics card and 8GB of RAM, Intel OneAPI provided me with substantial benefits and optimizations that greatly enhanced my project development and execution. Here's how Intel OneAPI supported me:

- **Optimized Libraries and Frameworks:** Intel OneAPI's optimized libraries and frameworks effectively utilized my MX graphics card and CPU, accelerating computations. For instance, in my CNN projects like malaria detection and cats vs. dogs classification, the deep learning frameworks leveraged the MX graphics card, improving training and inference speeds.

- **Parallelism and Multithreading:** Intel OneAPI's tools facilitated parallelism and multithreading, making the most of my CPU's cores for concurrent tasks. This was crucial for projects with large datasets, enhancing data processing speed.

- **Performance Optimization:** With Intel OneAPI, I had access to tools for analyzing and optimizing code performance. These tools identified bottlenecks and improved memory usage, optimizing hardware resources and overall performance.

- **Memory Management:** Given my 8GB RAM, efficient memory management was key. Intel OneAPI's tools helped optimize memory usage, enhancing stability.

- **Hardware-Aware Development:** OneAPI's emphasis on hardware-aware development allowed me to tailor applications to my hardware, even with its limitations.

- **Ease of Development:** Intel OneAPI's consistent development environment streamlined transitions between projects, eliminating the need to learn new tools each time.

- **Access to Tutorials and Documentation:** Intel OneAPI's rich resources, including tutorials and documentation, aided troubleshooting, skill enhancement, and resource utilization.

In summary, Intel OneAPI's optimizations accelerated project development, making the most of my laptop's hardware. Despite hardware constraints, I experienced noticeable improvements in project performance and development efficiency.

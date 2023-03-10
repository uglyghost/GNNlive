We have provided the experimental results data and the complete version of the code for everyone to verify. In addition, the training dataset will be provided later.


# GL360: Graph Representation Learning based FoV Prediction for 360° Live Video Streaming

## 1. Abstract
With the increased bandwidth demand, providing 360° live video streaming with premium quality significantly challenges the current video system. The key to alleviate this bandwidth pressure is focusing the limited network resources on the user's region of interest (also termed Field-of-View, FoV) instead of delivering the whole parts of the frames with equal presentations. 
 The online content generated in 360° live video coupling with the high dynamic of user viewport locations, however, makes the accuracy FoV prediction quite  difficult due to the lack of pre-knowledge on the user viewing behaviors. 
 In this paper, we propose a novel 360° transmission framework with a lightweight Graph Representation Learning based FoV prediction called GL360. 
    First, we model the relationship and interplay between users and tiles of the panoramic video with the dynamic heterogeneous relational graph convolutional network (RGCN) employed for the representation learning of the user and tile embedding.
    Second, we propose an online dynamic heterogeneous graph learning (DHGL) based algorithm to timely capture the time-varying features of the user's viewing behaviors  within limited prior knowledge.
    Further, we design an FoV-aware content delivery algorithm that allows the edge servers to determine the video tiles' resolution for each accessed user.
    Additionally, experimental results based on real traces demonstrate how our solution outperforms state-of-the-art solutions in FoV prediction and network performance.

 (1-1-Conan Gore Fly)  |  (1-2-Front)  |  (1-3-Help)
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-1](pic/1-1-Front.gif "1-1-Front") |  ![alt-text-2](pic/1-2-Front.gif "1-2-Front") |  ![alt-text-2](pic/1-3-Help.gif "1-3-Help")

  (1-4-Conan Weird Al)  |  (1-5-Tahiti Surf)  |  (1-6-Falluja)
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-1](pic/viewport_result.gif "title-7") |  ![alt-text-2](pic/viewport_result.gif "title-8") |  ![alt-text-2](pic/1-6-Falluja.gif "1-6-Falluja")

  (1-7-Cooking Battle)  |  (1-8-Football)  |  (1-9-Rhinos)
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-1](pic/1-7-Cooking.gif "1-7-Cooking Battle") |  ![alt-text-2](pic/viewport_result.gif "title-5") |  ![alt-text-2](pic/viewport_result.gif "title-6")

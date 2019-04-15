---
title: SoccerNet A Scalable Dataset for Event Spotting in Soccer Videos
subtitle: CVPR Workshop 2018
layout: publication
modal-id: 1
date: 2018-04-13
img: SoccerNet.png
thumbnail: SoccerNet-thumbnail.png
alt: image-alt
project-date: April 2018
paper: arXiv
paper_link: https://arxiv.org/abs/1804.04527
yt_link: x4E3DPy84xM
conference: CVPR Workshop on Computer Vision in Sports 2018
conference_link: http://www.vap.aau.dk/cvsports/

---

In this paper, we introduce SoccerNet, a benchmark for action spotting in soccer videos. The dataset is composed of 500 complete soccer games from six main European leagues, covering three seasons from 2014 to 2017 and a total duration of 764 hours. A total of 6,637 temporal annotations are automatically parsed from online match reports at a one minute resolution for three main classes of events (Goal, Yellow/Red Card, and Substitution). As such, the dataset is easily scalable. These annotations are manually refined to a one second resolution by anchoring them at a single timestamp following well-defined soccer rules. With an average of one event every 6.9 minutes, this dataset focuses on the problem of localizing very sparse events within long videos. We define the task of spotting as finding the anchors of soccer events in a video. Making use of recent developments in the realm of generic action recognition and detection in video, we provide strong baselines for detecting soccer events. We show that our best model for classifying temporal segments of length one minute reaches a mean Average Precision (mAP) of 67.8%. For the spotting task, our baseline reaches an Average-mAP of 49.7% for tolerances δ ranging from 5 to 60 seconds.



```
@article{giancola2018soccernet,
  title={SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos},
  author={Giancola, Silvio and Amine, Mohieddine and Dghaily, Tarek and Ghanem, Bernard},
  journal={arXiv preprint arXiv:1804.04527},
  year={2018}
}
```
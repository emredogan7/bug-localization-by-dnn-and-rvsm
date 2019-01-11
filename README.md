# Bug Localization from Bug Reports

- This study and implementation is adapted from the study [Bug Localization with Combination of Deep Learning and Information Retrieval](https://ieeexplore.ieee.org/document/7961519)


## Dataset

- For our implementation, the dataset of *Eclipse UI Platform* is used.
	- The source code of the project can be found [here](https://github.com/eclipse/eclipse.platform.ui).
	- The bug dataset can be accessed from [here](https://github.com/logpai/bugrepo/tree/master/EclipsePlatform).


## Approach

- In previous studies, a cosine similartiy based information retrieval model ,rVSM, has been used and resulted with good top-k accuracy results. In our case, rVSM approach is combined with some other metadata and fed to a deep neural network to conclude withg a relevancy score between a bug report and a source code file. This final relevany scores between all bug reports and source files are kept and top-k accuracy results for k=1,5,10,20 are calculated. In the original study, top-20 accuracy is found to be about 85% where our implementation achieves a 79% top-20 accuracy. 

- The top-k accruacy results for different k values from the original study & our study can be seen observed from the figures below.

Original Study            	    |  Our Implementation
:------------------------------:|:------------------------------:
![](./Results/origResults.png)  |  ![](./Results/ourResults.png)


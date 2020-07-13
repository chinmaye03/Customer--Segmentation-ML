## Project:Customer Segmentation using Machine learning
  

**Customer Segmentation can be a powerful means to identify unsatisfied customer needs. This technique can be used by companies to outperform the competition by developing uniquely appealing products and services.**
 
 
 **Customer Segmentation is the subdivision of a market into discrete customer groups that share similar characteristics. Customer Segmentation can be a powerful means to identify unsatisfied customer needs. Using the above data companies can then outperform the competition by developing uniquely appealing products and services.**

**The most common ways in which businesses segment their customer base are:**

1.Demographic information, such as gender, age, familial and marital status, income, education, and occupation.

2.Geographical information, which differs depending on the scope of the company. For localized businesses, this info might pertain to specific towns or counties. 
   For larger companies, it might mean a customer’s city, state, or even country of residence.

3.Psychographics, such as social class, lifestyle, and personality traits.

4.Behavioral data, such as spending and consumption habits, product/service usage, and desired benefits.

**Advantages of Customer Segmentation:**

* Determine appropriate product pricing.

* Develop customized marketing campaigns.

* Design an optimal distribution strategy.

* Choose specific product features for deployment.

* Prioritize new product development efforts.
## K Means Clustering Algorithm

**K-Means clustering is a type of unsupervised learning. The main goal of this algorithm to find groups in data and the number of groups is represented by K. It is an iterative procedure where each data point is assigned to one of the K groups based on feature similarity.**
 
 
 *->k-means is  one of  the simplest unsupervised  learning  algorithms  that  solve  the well  known clustering problem. The procedure follows a simple and  easy  way  to classify a given data set  through a certain number of  clusters (assume k clusters) fixed apriori. The  main  idea  is to define k centers, one for each cluster. These centers  should  be placed in a cunning  way  because of  different  location  causes different  result. So, the better  choice  is  to place them  as  much as possible  far away from each other. The  next  step is to take each point belonging  to a  given data set and associate it to the nearest center. When no point  is  pending,  the first step is completed and an early group age  is done. At this point we need to re-calculate k new centroids as barycenter of  the clusters resulting from the previous step. After we have these k new centroids, a new binding has to be done  between  the same data set points  and  the nearest new center. A loop has been generated. As a result of  this loop we  may  notice that the k centers change their location step by step until no more changes  are done or  in  other words centers do not move any more. Finally, this  algorithm  aims at  minimizing  an objective function know as squared error function given by:*


![](https://c02d4336-a-62cb3a1a-s-sites.googlegroups.com/site/dataclusteringalgorithms/k-means-clustering-algorithm/kmeans.JPG?attachauth=ANoY7coTkP_Wv8ivAY15ZIPbsUnMa0La6fGUAT2SWk0IL96-Th97hLchdWx0aIWR7RlFmgR2jCLWJQ80sGZmTLn1olQIxV-qwIgtfF_TmFnSB4jACVmhf1fq7v_iFtPR6ERXz29XCXX6lhFn0FLwTdrUSktpvNBDoKnl5fZ8fDD8lrUZscr90aPxeGFgWVXSelk5gD-2scbRme75ojRZbiVHsp_pEADfrrQ2TyTRjHMHnu1ft5By-krMwJLZ2rmkFBXSRSqoA4W_UGFm8BAA8KtBV4wFUzI32g%3D%3D&attredirects=0)


where,
                           
                           ‘||xi - vj||’ is the Euclidean distance between xi and vj.
                           
                           ‘ci’ is the number of data points in ith cluster.
                           
                           ‘c’ is the number of cluster centers
 


## Implementing K Means Algorithm

This project is a part of the Mall Customer Segmentation Data competition held on Kaggle.

The dataset can be downloaded from the kaggle website which can be found here[](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

**The steps involved for implementing K means are:**
1. Importing required libraries.
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
```


2. Accessing the dataset

3. Analyzing the dataset

4.Implementing K means algorithm

5.Visualizing the Clusters



```markdown


# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/chinmaye03/Customer--Segmentation-ML/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

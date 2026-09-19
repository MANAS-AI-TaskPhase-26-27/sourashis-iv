# ML Fundamentals and Data Preprocessing

## ML Basics Overview
- AI $\supset$ ML $\supset$ DL
- **Machine learning**: We can _train_ a model _(mathematical relationship)_ on a dataset resembling the real world, e.g. _translation apps, autonomous vehicles_.
- **AI Inference**: The model applies the _patterns_ learned from the training data to _make predictions_.
- **Supervised learning**: requires _labeled data_ and a human in the loop.
- **Unsupervised learning**: uses _unlabeled data_ and _discovers structure_ on its own, e.g. _anomaly detection_.
- **Reinforcement learning**: optimizes through _trial and error_.
- **Semi-supervised learning**: some labeled and mostly unlabeled
    - **Cons**: noise & bias sensitivity, complex (combining data)
- **Generative AI**
    - Initially trained by UL, where it learns to mimic the data.
    - Later, it can be fine-tuned using SL or RL on specific data.
    - **LLMs** are based on _transformer architectures_.

---

## Supervised Learning
- The data already contains the correct answer, and the tasks are well-defined.
- **Ensemble learning**: Multiple models are trained together, and the results are aggregated to find the best approach.
    - The individual algorithms, called **weak learners** or **base models** have a **bias-variance tradeoff** _(simple-complex)_,
      which this sort of learning solves.
- **Cons**: human bias, overfitting (too closely tailored to the training data)
- Can be used in recommendation engines, e.g. ranking the recommendations.

### Types
- **Regression**
    - Predicts a numeric value.
    - **_Types/Algorithms_**: linear, lasso, ridge, polynomial
- **Classification**
    - Predicts if something belongs in a category.
    - The features need to be categorized.
    - **_Types_**: binary, multi-class, multi-label _(one item $\rightarrow$ multiple labels)_
    - **_Algorithms_**: linear classifiers, support vector machines (SVMs), decision trees, k-nearest neighbor (KNN), logistic regression, random forest.

### Concepts
#### Data
- Datasets are made up of _examples_ $\rightarrow$ _size_) where **features** ($\rightarrow$ _diversity_) are the _inputs_ and **labels** are the _intended end-results_.
- If a feature has no causal relationship to the label, it doesn't help.
  Features can be added or removed based on how effective they are in predictions.
- They can be **labeled (ground truth)** or **unlabeled**.
#### Training
- Based on the **loss function** _(deviation from the actual value)_, the model gradually _updates_ its solution.
- The gradient of the function decides the direction of the update.
#### Evaluation
- We test the model with data with only features. If not good enough, can go back to training with relevant features.
#### Inference
- Predictions on unlabeled data

### Optimization Algorithms
- **Gradient descent** family, e.g. _stochastic GD (SGD)_
- Learning algorithms while training neural networks, etc.
- **Naive Bayes**: Presence of one feature doesn't affect the effect of the other in the probability of an outcome,
  and they have equal effects.
- **Linear regression**: b/w a continuous dependent and $\geq$ 1 independent variables.
- **Non-linear regression**
- **Logistic regression**: binary
- **Polynomial regression**: exponential
- **SVM**: separates the classes by a **decision boundary / hyperplane**, and tries to adjust it to _maximize the distance_.
- **KNN**: classifies datapoints based on their proximity to other datapoints, and they naturally separate out when plotted.
- **Random forest _(both C&R)_**: uncorrelated decision trees merged to reduce variance and increase accuracy.
    - The decision trees are _if-then-else trees_ that break down the dataset into chunks.
---

## Self-Supervised Learning
- The model generates its own _implicit_ and _pseudo_-labels, and uses that for training and evaluation.
  However, it needs pretext tasks to learn how to label in the first place.
- Widespread use in _computer vision_ and _NLP_
- **Transfer learning**: commonly associated, which is a pretrained model being made to do a downstream task.
- They learn rich, transferable features that can be _fine-tuned_ for domain-specific tasks.
- **Cons**: compute-intensive, noise & bias sensitivity, pretext tasks need expertise

---

## Unsupervised Learning
- **Pros**: flexible (since autonomous), scalable (since unlabeled)
- **Cons**: imprecise outcomes (no evaluation), noise & bias sensitivity
- Used in _data mining_.
- **Applications**: computer vision (object recognition), anomaly detection
- Human intervention is required to validate the output and _make sense of it_.
- There's a lack of transparency in the _basis of clustering_.

### Clustering
- The user doesn't define the categories, but can attempt to name later.

#### Exclusive & Overlapping
- **K-means** is used for exclusive clustering, where k groups are made based on the distance of the datapoint from the centroid of the group.
- **Soft or fuzzy** k-means is used for overlapping clustering.

#### Hierarchical Cluster Analysis (HCA)
- Visualized by a _dendogram_.
- **Agglomerative** _(bottoms-up)_
    - The distance b/w clusters is determined by:
        - **Ward's linkage**: increase in sum of squared datapoints
        - **Average linkage**: mean distance b/w every combination of datapoints
        - **Complete/maximum linkage**: maximum distance
        - **Single/minimum linkage**: minimum distance
    - Euclidean distance is usually used, but others, such as Manhattan distance can also be used.
- **Divisive** _(top-down)_: not commonly used

#### Probabilistic, e.g. the Gaussian Mixture Models (GMMs)
- An unspecified number of **Gaussian / normal / probability distribution functions**.
- If the _mean or variance_ are known, we can find which Gaussian a datapoint belongs to.
- Since we don't know these variables, we assume a latent variable representing the label.
- The **Expectation-Maximization (EM)** algorithm is used, where it randomly guesses the parameters of each Gaussian.
    - Expectation guesses probabilities, while Maximization updates the parameters of the Gaussians.
    - This process is carried out 'til the parameters stop changing.

### Association Rules
- Used to find **relationships** between datapoints, e.g., for **cross-selling**, which is how often one datapoint is expected appear with reference to another datapoint.
- **Algorithms**: Apriori, Eclat, FP-Growth

#### Apriori Algorithms
- Finds the frequent itemsets/combinations.

### Dimensionality reduction:
- Reduces the complexity as in data preprocessing, data compression, etc.
  to the most crucial features. This preserves accuracy while increasing efficiency.
- **_Examples_**: Principal Component Analysis (PCA), encoders

#### Principal Component Analysis (PCA)
- A linear transformation is applied to the data, and depending on the dimension of the dataset,
  it yields n orthogonal principal components. The components with low variance are discarded.
- Besides reducing redundant data, it can also compress data.
- The principal components are **hybrid features**.

#### Singular Value Decomposition (SVD)
- Factorizes a matrix into three low-rank matrices.

#### Autoencoders
- Uses neural networks, where the input layer is encoded to **hidden layers**, where the **bottleneck** is the layer with the fewest neurons/nodes.
- The output layer is **reconstructed** from the hidden layers, and the **reconstruction error** is noted.
- The weights in the network are adjusted to find the **optimal bottleneck**.

---

## Reinforcement Learning
- Takes an **action** based on its **state**, and there's either **reward** or **penalty**, which fine-tunes the model **policy**.
- Balances _exploration_ and _exploitation_, e.g. a self-driving car.
- _RL with Human Feedback (RLHF)_ is emerging
- **Pros**: can solve complex tasks
- **Cons**: inconsistent in the beginning, environmental data needs, reward hacking
---

## Deep Learning

### Neural Networks
- Processes training data with layers of nodes that mimic the human brain.
- **Node**: has _inputs, weights, bias/threshold, output_.
  If the output exceeds the threshold, it fires the next layer.

---

## Data Preprocessing

#### New
- `sns.hisplot(arr/Series, ax: Axes, kde=False, legend=True, label=)`
    - **Kernel Density Estimate (KDE)**: continuous curve from a histogram
- `sns.boxplot()` shows the outliers.
- `df.corr()` shows correlation b/w columns.
- `sns.heatmap(df, annot=False)`
- `df/Series.quantile(%ile/iterable[%iile])` or `np.percentile(Series/df, %ile/iterable[%ile]))`

### Steps
- **Sanity check**
    - null, duplicates, garbage values
- **Exploratory Data Analysis (EDA)**
    - Describe the data.
    - Use _histogram_ to understand the distribution.
    - Use _boxplot_ to identify outliers.
    - Use _scatter_ plot to understand the relationship b/w the features and the label.
    - Use _heatmap_ to understand correlation.
- **Missing values**
    - Fill mean/median/mode or impute
- **Outlier treatment**
    - Done on continuous numerical data and not on categorical or discrete variables or the label.
    - If there are many outliers, don't treat them.
    - Interquartile clipping $\rightarrow [q1 - \lambda \Delta, q3 + \lambda \Delta]$
- **Duplicate & garbage value treatment**
- **Encoding**

### Sanity Check & Missing Values
- If undefined, then `NaN` makes sense; otherwise, guess by other values in that _column_ or _row_ $\rightarrow$ **imputation**
- `dropna(axis=0, inplace=False)`
- `fillna(val/Series/dict, method='bfill/ffill', axis=0)`
    - `dict` as in _column $\rightarrow$ val_
    - `df.fillna(method='bfill')` ~ `df.bfill()`
- If the %age of missing values $\geq$ 50%, _delete the column_.
- `sklearn.impute.KNNImputer().fit_transform(DataFrame, n_neighbors=5)` to impute select columns by taking the average of the neighbors.
- `object` columns can have garbage values $\rightarrow$ perform `value_counts()` on them to detect.
- For _binary features_, can add a third value for `NaN` or a new column `feature_is_nan`, or fill it with the **mode**.

### Scaling

#### Normalization (Min-Max Scaling)
- $$\frac{x - x_{min}}{x{max} - x{min}}$$
- We've gotta scale related _continuous_ numeric data to fit in the same range in distance-based algorithms like KNN and SVM.
- `mlxtend.preprocessing.minmax_scaling(arr/df/Series, columns: iterable, min_val=0, max_val=1)`
- **Tip**: only pass the intended columns as the positional argument.

#### Standardization (Z-Score Scaling)
- $$\frac{x - x_{mean}}\sigma$$
- Centers the data to a _mean of 0_ and _standard deviation of 1_.
- Useful for _linear regression, logistic regression, PCA_.

## Power Transformation
- **Normal distribution** (aka **bell curve / Gaussian**): _Equal no._ of observations are on either side of the _mean_.
  `Mean = Median` and more _observations are closer_ to the mean.
- Algorithms like _linear discriminant analysis (LDA)_ and _Gaussian naive Bayes_ assume normal distribution.
- Both the shape and range change. _Correlation b/w features_ and the _relative spacing b/w points_ stay unchanged.
- Mean and median of the data change.
- `scipy.stats.boxcox(arr/Series)[0]`
    - All values must be _positive_.
    - For zero-valued features, add a small **constant**.
- **Tip**: only normalize the **skewed features**.

### Parsing Dates
- `pd.to_datetime(Series, format=, infer_datetime_format=False)`
    - `format`: _%d, %m, %y, %Y_

### Character Encodings
- `str.encode/bytes.decode(encoding, errors='strict/ignore/replace')` returns `bytes`
    - **UTF-8** uses a variable length of 1&#x2013;4 bytes.
- `charset_normalizer.detect(str)`: usually a part of the file read as bytes

### Inconsistent Data Entry
- _`str` methods_ and `fuzzywuzzy.process.extract(str, arr/Series, limit=5, scorer=`
    - `scorer=fuzzywuzzy.fuzz.token_sort_ratio` gives a score b/w 1 and 100.
- Change the values in close matches to the correct value.

### Encoding
- **One hot encoding**: `pd.get_dummies(df/Series, drop_first=True, dtype=bool)`
  - for **textual** **nominal data**
- **Label encoding**:
  - Convert to `category` type and `.cat.codes`
  - for **ordinal data**
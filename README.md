# FRUITS
(**F**eature Ext**R**action **U**sing **IT**erated **S**ums) <br>

## Idea
The purpose of this repository is to find a suitable way for using iterated sums signatures in order to extract features from time series data.
<br>
We do this by first creating an instance of the class `FeatureExtractor`. The User now has a lot of options to customize this extractor.
There are three steps the extractor is going to do where the user is able to make changes.
- preparing the data: `DataPreparateur`<br>
  This step can be just the identity function so we use each time series without any changes for the next step. Another option is to calculate the increments `x_i-x_{i-1}` first.
- calculating the iterated sums: `SummationIterator[s]`
  ...by specifying a bunch of SummationIterators first. Mainly one wants to use the default concatenations of compositions (e.g. `[11][1223]`) to calculate the iterated sums sigature (i.e. `<[11][1223], ISS(x)>`). More iterators lead to more features extracted at the end.
- extracting the featuers of the iterated sums: `Feature[s]`
  The user can now add a bunch of pre-defined feature extraction functions to the class. They will be applied to each one of the iterated sums from the previous step. The total number of features is `[number of feature functions added] * [number of SummationIterators added]`.
```python
X = numpy.array([...]) # think of a 3 dimensional time series dataset

featex = fruits.FeatureExtractor()

featex.set_data_preperateur(fruits.preparateurs.INC)

summation_iterators = fruits.iterators.generate_random_concatenations(number=10, dim=3, 
                        max_letter_weight=3, max_concatenation_length=5)

for it in summation_iterators:
  featex.add_summation_iterator(it)
  
featex.add_feature(fruits.features.PPV)
featex.add_feature(fruits.features.MAX)

# 20 features for each time series
extracted_features = featex(X)
```

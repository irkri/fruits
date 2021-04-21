# FRUITS
(**F**eature Ext**R**action **U**sing **IT**erated **S**ums) <br>

## Idea
The purpose of this repository is to find a suitable way for using iterated sums signatures in order to extract features from time series data.
<br>
We do this by first creating an instance of the class `Fruit`. The User now has a lot of options to customize this class.
There are three steps this feature extractor is going to do where the user is able to make changes.
- preparing the data: `DataPreparateur`<br>
  This step can be just the identity function so we use each time series without any changes for the next step. Another option is to calculate the increments `x_i-x_{i-1}` first.
- calculating the iterated sums: `SummationIterator[s]`<br>
  ...by specifying a bunch of SummationIterators first. Mainly one wants to use the default concatenations of compositions (e.g. `[11][1223]`) to calculate the iterated sums sigature (i.e. `<[11][1223], ISS(x)>`). More iterators lead to more features extracted at the end.
- extracting the features of the iterated sums: `FeatureFilter[s]`<br>
  The user can now add a bunch of pre-defined feature extraction functions to the class. They will be applied to each one of the iterated sums from the previous step. The total number of features is `[number of FeatureFilters added] * [number of SummationIterators added]` for one time series.
```python
X = numpy.array([...]) # think of a 3 dimensional time series dataset

featex = fruits.Fruit()

featex.add(fruits.preparateurs.INC)

summation_iterators = fruits.iterators.generate_random_concatenations(number=10, dim=3, 
                        max_letter_weight=3, max_concatenation_length=5)

# add the ten iterators to the class instance
featex.add(it)

# choose from a variety of FeatureFilters 
featex.add(fruits.features.PPV)
featex.add(fruits.features.MAX)

# 2*10=20 features for each time series
extracted_features = featex(X)
```

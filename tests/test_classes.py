# vim: tabstop=4 noexpandtab
import sys
sys.path.insert(0, "../")
import numpy as np
import unittest
import fruits

class TestFruit(unittest.TestCase):

	X_1 = np.array([[[-4,0.8,0,5,-3], [2,1,0,0,-7]],
					[[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]])

	def test_preparateurs(self):
		prep = fruits.preparateurs.DataPreparateur()

		X_1_1 = fruits.preparateurs.INC(True).prepare(self.X_1)
		X_1_2 = fruits.preparateurs.INC(zero_padding=False).prepare(self.X_1)

		self.assertTrue(np.allclose(X_1_1, 
						np.array([[[0,4.8,-0.8,5,-8], [0,-1,-1,0,-7]],
								  [[0,3,-6,4,-6], [0,4,-3,3.5,-7.5]]])))
		self.assertTrue(np.allclose(X_1_2, 
						np.array([[[-4,4.8,-0.8,5,-8], [2,-1,-1,0,-7]],
								  [[5,3,-6,4,-6], [-5,4,-3,3.5,-7.5]]])))

	def test_iterators(self):
		# word [11122][122222][11]
		word1 = [[lambda X: X[0, :]**3, lambda X: X[1, :]**2],
				 [lambda X: X[0, :], lambda X: X[1, :]**5],
				 [lambda X: X[0, :]**2]]

		# word [22][112][2221]
		word2 = [[lambda X: X[1, :]**2],
				 [lambda X: X[0, :]**2, lambda X: X[1, :]],
				 [lambda X: X[1, :]**3, lambda X: X[0, :]]]

		it1 = fruits.iterators.SummationIterator("word 1")
		it1.append(*word1)
		it2 = fruits.iterators.SummationIterator("word 2")
		it2.append(*word2)

		sit1 = fruits.iterators.SimpleWord("[11122][122222][11]")
		sit2 = fruits.iterators.SimpleWord("[22][112][2221]")

		result_fast = fruits.core.ISS(self.X_1, [sit1, sit2])
		result_slow = fruits.core.ISS(self.X_1, [it1, it2])

		self.assertTrue(np.allclose(result_slow, result_fast))

		# -------------------------------------------------------------

		w1 = fruits.iterators.SimpleWord("[1]")
		w2 = fruits.iterators.SimpleWord("[2]")
		w3 = fruits.iterators.SimpleWord("[11]")
		w4 = fruits.iterators.SimpleWord("[12]")
		w5 = fruits.iterators.SimpleWord("[1][1]")
		w6 = fruits.iterators.SimpleWord("[1][2]")

		r1 = fruits.core.ISS(self.X_1, [w1, w2, w3, w4, w5, w6])

		self.assertTrue(np.allclose(r1[:,0,:], 
									np.array([[-4,-3.2,-3.2,1.8,-1.2],
											  [5,13,15,21,21]])))
		self.assertTrue(np.allclose(r1[:,1,:], 
									np.array([[2,3,3,3,-4],
											  [-5,-6,-10,-10.5,-18.5]])))
		self.assertTrue(np.allclose(r1[:,2,:], 
									np.array([[16,16.64,16.64,41.64,50.64],
											  [25,89,93,129,129]])))
		self.assertTrue(np.allclose(r1[:,3,:], 
									np.array([[-8,-7.2,-7.2,-7.2,13.8],
											  [-25,-33,-41,-44,-44]])))
		self.assertTrue(np.allclose(r1[:,4,:], 
									np.array([[16,13.44,13.44,22.44,26.04],
											  [25,129,159,285,285]])))
		self.assertTrue(np.allclose(r1[:,5,:], 
									np.array([[-8,-11.2,-11.2,-11.2,-2.8],
											  [-25,-38,-98,-108.5,-276.5]])))

		# -------------------------------------------------------------

		# word: [relu(0)][relu(1)]
		relu_iterator = fruits.iterators.SummationIterator("relu collection")
		relu_iterator.append([lambda X: X[0, :]*(X[0, :]>0)], 
							 [lambda X: X[1, :]*(X[1, :]>0)])
		mix = [relu_iterator, fruits.iterators.SimpleWord("[111]")]

		mix_result = fruits.core.ISS(self.X_1, mix)

		self.assertTrue(np.allclose(mix_result,
									np.array([[[0,0.8,0.8,0.8,0.8],
											   [-64,-63.488,-63.488,61.512,
											   	34.512]],
											  [[0,0,0,0,0],
											   [125,637,645,861,861]]])))

	def test_sieves(self):
		sieve = fruits.features.FeatureSieve()

		ppv_1 = fruits.features.PPV(quantile=0, 
									constant=True).sieve(self.X_1[0])
		ppv_2 = fruits.features.PPV(quantile=0.5, constant=False,
									sample_size=1).sieve(self.X_1[1])

		self.assertTrue(np.allclose(ppv_1, np.array([3/5,4/5])))
		self.assertTrue(np.allclose(ppv_2, np.array([1,0])))

		maxi = fruits.features.MAX.sieve(self.X_1[0])
		mini = fruits.features.MIN.sieve(self.X_1[1])

		self.assertTrue(np.allclose(maxi, np.array([5,2])))
		self.assertTrue(np.allclose(mini, np.array([0,-8])))

		ppv_c = fruits.features.PPV_connected(quantile=0, constant=True).sieve(self.X_1[0])
		np.testing.assert_allclose( ppv_c, [0.8,0.4] )

	def test_fruit(self):
		featex = fruits.Fruit()

		featex.add(fruits.preparateurs.INC(zero_padding=False))

		featex.add(fruits.iterators.generate_words(1,3,5))
		featex.add(fruits.iterators.generate_random_words(12,2,3,5))
		self.assertEqual(len(featex.get_summation_iterators()), 375)

		featex.add(fruits.features.PPV(quantile=0, constant=True))
		featex.add(fruits.features.PPV(quantile=0.2, constant=False, 
									   sample_size=1))
		featex.add(fruits.features.MAX)
		featex.add(fruits.features.MIN)

		self.assertEqual(featex.nfeatures(), 1500)

		features = featex(self.X_1)

	def test_multiple_branches(self):
		featex = fruits.Fruit()

		w1 = fruits.iterators.SimpleWord("[1]")
		w2 = fruits.iterators.SimpleWord("[2]")
		w3 = fruits.iterators.SimpleWord("[11]")
		w4 = fruits.iterators.SimpleWord("[12]")
		w5 = fruits.iterators.SimpleWord("[1][1]")
		w6 = fruits.iterators.SimpleWord("[1][2]")

		featex.add(w1, w2, w3)
		featex.add(fruits.features.MAX)
		featex.start_new_branch()
		featex.add(w4, w5, w6)
		featex.add(fruits.features.MIN)

		self.assertEqual(featex.nfeatures(), 6)

		features = featex(self.X_1)
		#print(features)
		self.assertEqual(features.shape, (2, 6))

		self.assertTrue(np.allclose(features,
									np.array([[1.8,3,50.64,-8,13.44,-11.2],
											  [21,-5,129,-44,25,-276.5]])))

if __name__=='__main__':
	unittest.main()

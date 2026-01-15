import sklearn.datasets
from data_loaders.synthetic import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader

class blobs_generator(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                 num_samples=200,
                 blobs_features=2,
                 **kwargs):
        self.num_samples = num_samples
        self.blobs_features = blobs_features
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         dataset_name='Blobs Synthetic',
                         **kwargs)
        
    def load_data(self):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_blobs,
                                        samples=self.num_samples,
                                        test=False,
                                        n_features=self.blobs_features)
        return data    


if __name__ == "__main__":
    loader = blobs_generator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
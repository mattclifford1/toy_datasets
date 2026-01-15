import sklearn.datasets
from data_loaders.synthetic import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader

class moons_generator(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                 num_samples=200,
                 moons_noise=0.2,
                 **kwargs):
        self.num_samples = num_samples
        self.moons_noise = moons_noise
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         dataset_name='Moons Synthetic',
                         **kwargs)
        
    def load_data(self):
        '''
        sample from the half moons data distribution
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_moons,
                                        samples=self.num_samples,
                                        test=False,
                                        noise=self.moons_noise)
        return data    


if __name__ == "__main__":
    loader = moons_generator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
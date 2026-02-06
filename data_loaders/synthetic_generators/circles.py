import sklearn.datasets
from data_loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader

class circles_generator(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 train_size=0.5,
                 num_samples=200,
                 circles_noise=0.2,
                 **kwargs):
        self.num_samples = num_samples
        self.circles_noise = circles_noise
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Circles Synthetic',
                         **kwargs)
        
    def load_data(self):
        '''
        sample from the circles data distribution
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_circles,
                                        samples=self.num_samples,
                                        test=False,
                                        noise=self.circles_noise,
                                        factor=0.8)
        return data    


if __name__ == "__main__":
    loader = circles_generator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
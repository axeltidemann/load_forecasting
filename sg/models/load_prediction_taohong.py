import load_prediction
import taohong

class VanillaModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        pass
    
    def _get_transform(self):
        return taohong.vanilla

    
if __name__ == '__main__':
    load_prediction.run(VanillaModelCreator)

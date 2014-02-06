class Model(object):
    """A class that holds all the properties necessary for a model to
    be employed in the GA search for optimal parameters."""

    def __init__(self, name, genes, error_func, transformer, loci):
        self._name = name
        self._genes = genes
        self._error_func = error_func
        self._transformer = transformer
        self._loci = loci
        self._dataset = None
        self._day = None
        self._preprocessors = None
        self._postprocessors = None

    @property
    def name(self):
        return self._name
    
    def get_day(self):
        return self._day
    def set_day(self, day):
        self._day = day
    day = property(get_day, set_day)

    def get_loci(self):
        return self._loci
    def set_loci(self, loci):
        self._loci = loci
    loci = property(get_loci, set_loci)

    def get_genes(self):
        return self._genes
    def set_genes(self, genes):
        self._genes = genes
    genes = property(get_genes, set_genes)

    def get_error_func(self):
        return self._error_func
    def set_error_func(self, error_func):
        self._error_func = error_func
    error_func = property(get_error_func, set_error_func)

    def get_preprocessors(self):
        return self._preprocessors
    def set_preprocessors(self, preprocessors):
        self._preprocessors = preprocessors
    preprocessors = property(get_preprocessors, set_preprocessors)

    def get_transformer(self):
        return self._transformer
    def set_transformer(self, transformer):
        self._transformer = transformer
    transformer = property(get_transformer, set_transformer)

    def get_postprocessors(self):
        return self._postprocessors
    def set_postprocessors(self, postprocessors):
        self._postprocessors = postprocessors
    postprocessors = property(get_postprocessors, set_postprocessors)

    def get_genome(self):
        return self._genome
    def set_genome(self, genome):
        self._genome = genome
    genome = property(get_genome, set_genome)

    def get_dataset(self):
        return self._dataset
    def set_dataset(self, dataset):
        self._dataset = dataset
    dataset = property(get_dataset, set_dataset)

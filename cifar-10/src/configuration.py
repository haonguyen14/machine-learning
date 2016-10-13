
class Configuration(object):

    def __init__(
        self,
        input_size,
        output_size,
        examples_per_epoches,
        batch_size,
        a_function
    ):

        self.input_size = input_size
        self.output_size = output_size

        self.examples_per_epoches = examples_per_epoches
        self.batch_size = batch_size

        self.a_function = a_function

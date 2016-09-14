
class Configuration(object):

    def __init__(
        self,
        input_size,
        output_size,
        batch_size,
        num_epoch,
        a_function
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.a_function = a_function

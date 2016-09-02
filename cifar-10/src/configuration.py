
class Configuration(object):

    def __init__(
        self,
        input_size,
        output_size,
        batch_size,
        num_epoch
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
